# -*- coding: utf-8 -*-
# OTOFON + FONALİZ ENTEGRE TARAMA SCRIPT'İ

# --- 1. ADIM: Kütüphaneleri Import Etme ---
import pandas as pd
import numpy as np
import time
import gspread
import pytz
import os
import json
import sys
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from tefas import Crawler
from tqdm import tqdm
import concurrent.futures
import traceback
import warnings
# Duygu Analizi için eklenen kütüphaneler
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings('ignore')

# --- Sabitler ve Yapılandırma ---
GSPREAD_CREDENTIALS_SECRET = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
SHEET_ID = '1hSD4towyxKk9QHZFAcRlXy9NlLa_AyVrB9Jsy86ok14' # OtoFon Google Sheet ID'si
WORKSHEET_NAME_WEEKLY = 'haftalık'
WORKSHEET_NAME_FONALIZ = 'Fonanaliz' # Yeni çalışma sayfası adı
TIMEZONE = pytz.timezone('Europe/Istanbul')
MAX_WORKERS = 10

# --- Duygu Analizi Modeli ---
try:
    print("Duygu analizi modeli ve tokenizer yükleniyor...")
    MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print("✅ Duygu analizi modeli başarıyla yüklendi.")
except Exception as e:
    print(f"❌ Duygu analizi modeli yüklenirken hata oluştu: {e}")
    tokenizer = None
    model = None

# --- Yardımcı Fonksiyonlar ---
def google_sheets_auth():
    print("\nGoogle Sheets için kimlik doğrulaması yapılıyor...")
    try:
        if not GSPREAD_CREDENTIALS_SECRET:
            print("❌ Hata: GCP_SERVICE_ACCOUNT_KEY secret bulunamadı.")
            sys.exit(1)
        creds_json = json.loads(GSPREAD_CREDENTIALS_SECRET)
        gc = gspread.service_account_from_dict(creds_json)
        print("✅ Kimlik doğrulama başarılı.")
        return gc
    except Exception as e:
        print(f"❌ Kimlik doğrulama sırasında hata oluştu: {e}")
        traceback.print_exc()
        sys.exit(1)

try:
    tefas_crawler_global = Crawler()
    print("TEFAS Crawler başarıyla başlatıldı.")
except Exception as e:
    print(f"TEFAS Crawler başlatılırken hata: {e}")
    tefas_crawler_global = None

def load_takasbank_fund_list():
    print(f"Takasbank'tan güncel fon listesi yükleniyor...")
    try:
        df_excel = pd.read_excel(TAKASBANK_EXCEL_URL, engine='openpyxl')
        df_data = df_excel[['Fon Adı', 'Fon Kodu']].copy()
        df_data['Fon Kodu'] = df_data['Fon Kodu'].astype(str).str.strip().str.upper()
        df_data.dropna(subset=['Fon Kodu'], inplace=True)
        df_data = df_data[df_data['Fon Kodu'] != '']
        print(f"✅ {len(df_data)} adet fon bilgisi okundu.")
        return df_data
    except Exception as e:
        print(f"❌ Takasbank Excel yükleme hatası: {e}")
        return pd.DataFrame()

def get_price_on_or_before(df_fund_history, target_date: date):
    if df_fund_history is None or df_fund_history.empty or target_date is None: return np.nan
    df_filtered = df_fund_history[df_fund_history['date'] <= target_date].copy()
    if not df_filtered.empty: return df_filtered.sort_values(by='date', ascending=False)['price'].iloc[0]
    return np.nan

def calculate_change(current_price, past_price):
    if pd.isna(current_price) or pd.isna(past_price) or past_price is None or current_price is None: return np.nan
    try:
        current_price_float, past_price_float = float(current_price), float(past_price)
        if past_price_float == 0: return np.nan
        return ((current_price_float - past_price_float) / past_price_float) * 100
    except (ValueError, TypeError): return np.nan

def fetch_data_for_fund_parallel(args):
    fon_kodu, start_date_overall, end_date_overall = args
    global tefas_crawler_global
    if tefas_crawler_global is None: return fon_kodu, None, pd.DataFrame()
    
    try:
        df = tefas_crawler_global.fetch(
            start=start_date_overall.strftime("%Y-%m-%d"),
            end=end_date_overall.strftime("%Y-%m-%d"),
            name=fon_kodu,
            columns=["date", "price", "market_cap", "number_of_investors", "title"]
        )
        if df.empty: return fon_kodu, None, None
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        fon_adi = df['title'].iloc[0] if not df.empty and 'title' in df.columns else fon_kodu
        return fon_kodu, fon_adi, df.sort_values(by='date').reset_index(drop=True)
    except Exception as e:
        print(f"\n❌ Hata: Fon '{fon_kodu}' için veri çekilemedi. Detay: {e}")
        return fon_kodu, None, None

# --- DUYGU ANALİZİ BÖLÜMÜ ---
def get_kap_news(fon_kodu):
    try:
        url = f"https://www.kap.org.tr/tr/fon-bilgileri/genel/{fon_kodu}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        news_items = soup.select('div.w-col.w-col-11 a')
        return [item.text.strip() for item in news_items[:5]] # Son 5 haberi al
    except requests.exceptions.RequestException as e:
        # print(f"KAP Haberleri alınamadı ({fon_kodu}): {e}")
        return []

def analyze_sentiment(texts, tokenizer, model):
    if not texts or tokenizer is None or model is None:
        return {'label': 'N/A', 'score': 0.0}
    
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Ortalama skoru hesapla
        avg_prob = probs.mean(dim=0)
        sentiment_label = model.config.id2label[avg_prob.argmax().item()]
        sentiment_score = avg_prob.max().item()
        
        return {'label': sentiment_label, 'score': round(sentiment_score, 2)}
    except Exception:
        return {'label': 'Hata', 'score': 0.0}

# --- FONALİZ BÖLÜMÜ ---
def hesapla_metrikler(df_fon_fiyat):
    if df_fon_fiyat is None or len(df_fon_fiyat) < 10: return None
    df_fon_fiyat['daily_return'] = df_fon_fiyat['price'].pct_change()
    df_fon_fiyat = df_fon_fiyat.dropna()
    if df_fon_fiyat.empty: return None
    getiri = (df_fon_fiyat['price'].iloc[-1] / df_fon_fiyat['price'].iloc[0]) - 1
    volatilite = df_fon_fiyat['daily_return'].std() * np.sqrt(252)
    ortalama_gunluk_getiri = df_fon_fiyat['daily_return'].mean()
    sharpe_orani = (ortalama_gunluk_getiri / df_fon_fiyat['daily_return'].std()) * np.sqrt(252) if df_fon_fiyat['daily_return'].std() != 0 else 0
    negatif_getiriler = df_fon_fiyat[df_fon_fiyat['daily_return'] < 0]['daily_return']
    if negatif_getiriler.empty or negatif_getiriler.std() == 0:
        sortino_orani = 0
    else:
        downside_deviation = negatif_getiriler.std() * np.sqrt(252)
        sortino_orani = (ortalama_gunluk_getiri * 252) / downside_deviation if downside_deviation != 0 else 0
    return {
        'Getiri (%)': round(getiri * 100, 2),
        'Standart Sapma (Yıllık %)': round(volatilite * 100, 2),
        'Sharpe Oranı (Yıllık)': round(sharpe_orani, 2),
        'Sortino Oranı (Yıllık)': round(sortino_orani, 2),
        'Piyasa Değeri (TL)': df_fon_fiyat['market_cap'].iloc[-1],
        'Yatırımcı Sayısı': df_fon_fiyat['number_of_investors'].iloc[-1]
    }

def run_fonaliz_scan_to_gsheets(fon_listesi: list, gc):
    print("\n" + "="*40)
    print("     AŞAMA 2: FONALİZ RİSK VE DUYGU ANALİZİ BAŞLATILIYOR")
    print(f"     {len(fon_listesi)} adet filtrelenmiş fon analiz edilecek...")
    print("="*40)
    
    if not fon_listesi:
        print("ℹ️ Fonaliz için filtreden geçen fon bulunamadı. İşlem atlanıyor.")
        return

    ANALIZ_SURESI_AY = 3
    end_date = datetime.now(TIMEZONE).date()
    start_date = end_date - pd.DateOffset(months=ANALIZ_SURESI_AY)
    
    tasks = [(fon_kodu, start_date, end_date) for fon_kodu in fon_listesi]
    analiz_sonuclari = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, task): task[0] for task in tasks}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon), total=len(tasks), desc=" Fonaliz Risk ve Duygu Analizi")

        for future in progress_bar:
            fon_kodu, fon_adi, data = future.result()
            if data is not None:
                metrikler = hesapla_metrikler(data)
                if metrikler:
                    # Duygu Analizi Entegrasyonu
                    haberler = get_kap_news(fon_kodu)
                    duygu_sonucu = analyze_sentiment(haberler, tokenizer, model)
                    
                    sonuc = {
                        'Fon Kodu': fon_kodu, 
                        'Fon Adı': fon_adi, 
                        **metrikler,
                        'Duygu Etiketi': duygu_sonucu['label'],
                        'Duygu Skoru': duygu_sonucu['score']
                    }
                    analiz_sonuclari.append(sonuc)

    if not analiz_sonuclari:
        print("\n--- SONUÇ: Fonaliz için analiz edilecek yeterli veri bulunamadı. ---")
        return

    df_sonuc = pd.DataFrame(analiz_sonuclari)
    sutun_sirasi = [
        'Fon Kodu', 'Fon Adı', 'Yatırımcı Sayısı', 'Piyasa Değeri (TL)', 
        'Duygu Etiketi', 'Duygu Skoru', # Yeni sütunlar
        'Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)', 'Getiri (%)', 'Standart Sapma (Yıllık %)'
    ]
    df_sonuc = df_sonuc[sutun_sirasi]
    df_sonuc_sirali = df_sonuc.sort_values(by=['Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)'], ascending=[False, False])

    print(f"\n✅ Fonaliz tamamlandı. Sonuçlar Google Sheets'teki '{WORKSHEET_NAME_FONALIZ}' sayfasına yazılıyor...")
    try:
        spreadsheet = gc.open_by_key(SHEET_ID)
        try:
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME_FONALIZ)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME_FONALIZ, rows="1000", cols=30) # Sütun sayısı artırıldı
        
        worksheet.clear()
        df_sonuc_sirali = df_sonuc_sirali.replace([np.inf, -np.inf], np.nan).fillna('')
        worksheet.update([df_sonuc_sirali.columns.values.tolist()] + df_sonuc_sirali.values.tolist())
        
        body_resize = {"requests": [{"autoResizeDimensions": {"dimensions": {"sheetId": worksheet.id, "dimension": "COLUMNS"}}}]}
        spreadsheet.batch_update(body_resize)
        print("✅ Google Sheets güncellendi ve sütunlar yeniden boyutlandırıldı.")
    except Exception as e:
        print(f"❌ Google Sheets'e yazma hatası (Fonaliz): {e}")

# --- HAFTALIK TARAMA FONKSİYONU ---
def run_weekly_scan(num_weeks: int):
    start_time_main = time.time()
    today = datetime.now(TIMEZONE).date()
    all_fon_data_df = load_takasbank_fund_list()

    if all_fon_data_df.empty:
        print("❌ Taranacak fon listesi alınamadı. İşlem durduruldu.")
        return pd.DataFrame()

    print("\n" + "="*40)
    print("     AŞAMA 1: HAFTALIK İVMELENME TARAMASI BAŞLATILIYOR")
    print("="*40)

    genel_veri_cekme_baslangic_tarihi = today - timedelta(days=(num_weeks * 7) + 21)
    tasks = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, today) for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]
    
    weekly_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in tasks}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon), total=len(tasks), desc="Haftalık Tarama")

        for future in progress_bar:
            fon_kodu, _, fund_history = future.result()
            if fund_history is None or fund_history.empty: continue

            weekly_changes = []
            current_week_end_date = today
            for _ in range(num_weeks):
                current_week_start_date = current_week_end_date - timedelta(days=7)
                price_end = get_price_on_or_before(fund_history, current_week_end_date)
                price_start = get_price_on_or_before(fund_history, current_week_start_date)
                weekly_changes.append(calculate_change(price_end, price_start))
                current_week_end_date = current_week_start_date
            
            if len(weekly_changes) == num_weeks and all(pd.notna(c) for c in weekly_changes):
                weekly_results.append({'Fon Kodu': fon_kodu, 'Hafta_1_Getiri': weekly_changes[0], 'Hafta_2_Getiri': weekly_changes[1]})

    results_df = pd.DataFrame(weekly_results)
    print(f"\n✅ Haftalık tarama tamamlandı. Toplam Süre: {time.time() - start_time_main:.2f} saniye")
    return results_df

# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    print("Entegre OtoFon+Fonaliz Script'i Başlatıldı.")
    
    gc_auth = google_sheets_auth()
    if not gc_auth:
        sys.exit(1)

    # 1. Adım: Haftalık taramayı çalıştır
    haftalik_sonuclar_df = run_weekly_scan(num_weeks=2)

    if haftalik_sonuclar_df.empty:
        print("Haftalık tarama sonucu boş. İşlem sonlandırılıyor.")
        sys.exit(0)

    # 2. Adım: Sonuçları filtrele
    print("\nFiltreleme uygulanıyor: Son 2 haftanın toplam getirisi >= %2")
    
    # NaN değerleri 0 ile doldurarak hataları önle
    haftalik_sonuclar_df['Toplam_Getiri'] = haftalik_sonuclar_df['Hafta_1_Getiri'].fillna(0) + haftalik_sonuclar_df['Hafta_2_Getiri'].fillna(0)
    
    filtrelenmis_df = haftalik_sonuclar_df[haftalik_sonuclar_df['Toplam_Getiri'] >= 2].copy()
    
    if filtrelenmis_df.empty:
        print("Filtreyi geçen fon bulunamadı. İşlem sonlandırılıyor.")
        sys.exit(0)
        
    filtrelenmis_fon_listesi = filtrelenmis_df['Fon Kodu'].tolist()
    print(f"✅ Filtreleme tamamlandı. {len(filtrelenmis_fon_listesi)} fon Fonaliz için seçildi.")
    print("Filtrelenen Fonlar:", filtrelenmis_fon_listesi)

    # 3. Adım: Fonaliz'i filtrelenmiş liste ile çalıştır
    run_fonaliz_scan_to_gsheets(filtrelenmis_fon_listesi, gc_auth)

    print("\n--- Tüm işlemler tamamlandı ---")
        
    print("\nScript başarıyla tamamlandı.")