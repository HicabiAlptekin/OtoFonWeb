# -*- coding: utf-8 -*-
# ENTEGRE FON TARAMA VE ANALİZ ARACI (OtoFon + Fonaliz + Trend Analizi)
# v3.0 - tefas-crawler v0.6.0+ API + 3 Günlük Trend Analizi + Google Sheets
#
# Bu script, GitHub Actions'da çalışarak:
# 1. Takasbank'tan fon listesini alır
# 2. TEFAS API'sinden (v0.6.0+) fon verilerini çeker
# 3. Haftalık getiri analizi yapar
# 4. 3 günlük kısa trend analizi yapar
# 5. Fonaliz risk/getiri metriklerini hesaplar
# 6. Tüm sonuçları Google Sheets'e yazar

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

warnings.filterwarnings('ignore')

# --- SABİTLER ---
GSPREAD_CREDENTIALS_SECRET = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
SHEET_ID = '1hSD4towyxKk9QHZFAcRlXy9NlLa_AyVrB9Jsy86ok14'
WORKSHEET_MANUAL = 'veriler'
WORKSHEET_WEEKLY = 'haftalık'
WORKSHEET_FONALIZ = 'Fonanaliz'
WORKSHEET_TREND = 'Kısa Trend'  # YENİ: 3 günlük trend analizi sayfası
TIMEZONE = pytz.timezone('Europe/Istanbul')
MAX_WORKERS = 10
TREND_GUN_SAYISI = 3

# Global TEFAS crawler
try:
    tefas_crawler_global = Crawler()
    print("✅ TEFAS Crawler başlatıldı.")
except Exception as e:
    print(f"❌ TEFAS Crawler hatası: {e}")
    tefas_crawler_global = None


# --- YARDIMCI FONKSİYONLAR ---
def google_sheets_auth():
    print("\n🔑 Google Sheets kimlik doğrulaması...")
    try:
        if not GSPREAD_CREDENTIALS_SECRET:
            print("❌ GCP_SERVICE_ACCOUNT_KEY bulunamadı.")
            sys.exit(1)
        gc = gspread.service_account_from_dict(json.loads(GSPREAD_CREDENTIALS_SECRET))
        print("✅ Google Sheets bağlantısı başarılı.")
        return gc
    except Exception as e:
        print(f"❌ Google Sheets hatası: {e}")
        traceback.print_exc()
        sys.exit(1)


def load_takasbank_fund_list():
    """Takasbank'tan güncel fon listesini yükler."""
    print("📥 Takasbank fon listesi yükleniyor...")
    try:
        df_excel = pd.read_excel(TAKASBANK_EXCEL_URL, engine='openpyxl')
        df_data = df_excel[['Fon Adı', 'Fon Kodu']].copy()
        df_data['Fon Kodu'] = df_data['Fon Kodu'].astype(str).str.strip().str.upper()
        df_data.dropna(subset=['Fon Kodu'], inplace=True)
        df_data = df_data[df_data['Fon Kodu'] != '']
        print(f"✅ {len(df_data)} fon bulundu.")
        return df_data
    except Exception as e:
        print(f"❌ Takasbank yükleme hatası: {e}")
        return pd.DataFrame()


def get_value_on_or_before(df_fund, target_date, column='price'):
    if df_fund is None or df_fund.empty or target_date is None:
        return np.nan
    df_filtered = df_fund[df_fund['date'] <= target_date]
    if not df_filtered.empty:
        return df_filtered.sort_values(by='date', ascending=False)[column].iloc[0]
    return np.nan


def get_value_on_or_after(df_fund, target_date, column='price'):
    if df_fund is None or df_fund.empty or target_date is None:
        return np.nan
    df_filtered = df_fund[df_fund['date'] >= target_date]
    if not df_filtered.empty:
        return df_filtered.sort_values(by='date', ascending=True)[column].iloc[0]
    return np.nan


def calculate_change(current_price, past_price):
    if pd.isna(current_price) or pd.isna(past_price) or past_price == 0:
        return np.nan
    try:
        return ((float(current_price) - float(past_price)) / float(past_price)) * 100
    except (ValueError, TypeError):
        return np.nan


def fetch_fund_data(args):
    """TEFAS v0.6.0 API ile fon verisi çeker (basitleştirilmiş)."""
    fon_kodu, start_date, end_date = args
    global tefas_crawler_global
    if tefas_crawler_global is None:
        return fon_kodu, None, None

    try:
        df = tefas_crawler_global.fetch(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            name=fon_kodu,
            columns=["date", "price", "title"]
        )
        if df.empty:
            return fon_kodu, None, None
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
        fon_adi = df['title'].iloc[0] if 'title' in df.columns else fon_kodu
        return fon_kodu, fon_adi, df
    except Exception as e:
        print(f"  ⚠️ {fon_kodu}: Veri çekme hatası - {e}")
        return fon_kodu, None, None


def hesapla_metrikler(df_fon_fiyat):
    """Risk/getiri metriklerini hesaplar (v0.6.0 uyumlu)."""
    if df_fon_fiyat is None or len(df_fon_fiyat) < 10:
        return None
    df_fon_fiyat['daily_return'] = df_fon_fiyat['price'].pct_change()
    df_fon_fiyat = df_fon_fiyat.dropna()
    if df_fon_fiyat.empty:
        return None

    getiri = (df_fon_fiyat['price'].iloc[-1] / df_fon_fiyat['price'].iloc[0]) - 1
    volatilite = df_fon_fiyat['daily_return'].std() * np.sqrt(252)
    ortalama_gunluk_getiri = df_fon_fiyat['daily_return'].mean()
    gunluk_std = df_fon_fiyat['daily_return'].std()
    sharpe = (ortalama_gunluk_getiri / gunluk_std) * np.sqrt(252) if gunluk_std != 0 else 0

    neg_getiriler = df_fon_fiyat[df_fon_fiyat['daily_return'] < 0]['daily_return']
    if neg_getiriler.empty or neg_getiriler.std() == 0:
        sortino = 0
    else:
        downside = neg_getiriler.std() * np.sqrt(252)
        sortino = (ortalama_gunluk_getiri * 252) / downside if downside != 0 else 0

    return {
        'Getiri (%)': round(getiri * 100, 2),
        'Standart Sapma (%)': round(volatilite * 100, 2),
        'Sharpe (Yıllık)': round(sharpe, 2),
        'Sortino (Yıllık)': round(sortino, 2),
    }


def write_to_sheet(spreadsheet, worksheet_name, df_data, headers=None):
    """DataFrame'i Google Sheets sayfasına yazar."""
    try:
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            print(f"  📋 '{worksheet_name}' sayfası oluşturuluyor...")
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows="2000", cols=30)

        worksheet.clear()
        df_clean = df_data.replace([np.inf, -np.inf], np.nan).fillna('')
        if headers:
            worksheet.update([headers] + df_clean.values.tolist())
        else:
            worksheet.update([df_clean.columns.values.tolist()] + df_clean.values.tolist())

        body = {"requests": [{"autoResizeDimensions": {
            "dimensions": {"sheetId": worksheet.id, "dimension": "COLUMNS"}
        }}]}
        spreadsheet.batch_update(body)
        print(f"  ✅ '{worksheet_name}' güncellendi.")
        return worksheet
    except Exception as e:
        print(f"  ❌ '{worksheet_name}' yazma hatası: {e}")
        return None


# --- TEKİL TARAMA ---
def run_single_scan(gc, scan_date=None):
    """Belirli bir tarih için tekil tarama yapar."""
    if scan_date is None:
        scan_date = datetime.now(TIMEZONE).date() - timedelta(days=1)

    print(f"\n{'='*50}")
    print(f"📊 AŞAMA 1: TEKİL TARAMA ({scan_date})")
    print(f"{'='*50}")

    fon_df = load_takasbank_fund_list()
    if fon_df.empty:
        return

    baslangic = scan_date - relativedelta(years=1, months=2) - timedelta(days=30)
    tasks = [(kod, baslangic, scan_date) for kod in fon_df['Fon Kodu'].unique()]
    sonuclar = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_fund_data, t): t[0] for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=" Tekil Tarama"):
            kod, ad, data = future.result()
            if data is None or data.empty:
                continue

            son_fiyat = get_value_on_or_before(data, scan_date)
            donemler = {
                'Günlük %': timedelta(days=1), 'Haftalık %': timedelta(weeks=1),
                '2 Haftalık %': timedelta(weeks=2), 'Aylık %': relativedelta(months=1),
                '3 Aylık %': relativedelta(months=3), '6 Aylık %': relativedelta(months=6),
                '1 Yıllık %': relativedelta(years=1)
            }
            satir = {'Fon Kodu': kod, 'Fon Adı': ad}
            if pd.notna(son_fiyat):
                for donem, delta in donemler.items():
                    onceki = get_value_on_or_before(data, scan_date - delta)
                    satir[donem] = calculate_change(son_fiyat, onceki)
            sonuclar.append(satir)

    if sonuclar:
        df = pd.DataFrame(sonuclar)
        df = df.sort_values('Haftalık %', ascending=False, na_position='last')
        write_to_sheet(gc.open_by_key(SHEET_ID), WORKSHEET_MANUAL, df)
        print(f"✅ Tekil tarama tamamlandı: {len(df)} fon")


# --- HAFTALIK TARAMA + TREND ANALİZİ (GÜNCELLENMİŞ) ---
def run_weekly_scan(gc, num_weeks=2):
    """
    Haftalık getiri analizi + 3 günlük trend analizi.
    
    Değişiklikler (v3.0):
    - Her hafta için ayrı bitiş fiyatı kullanılır (düzeltildi)
    - Son 3 iş günü vs önceki 3 iş günü trend analizi eklendi
    - Trend sınıflandırması: HIZLANAN, YUKSELEN, DONUS, DUSUS, DUSEN
    """
    print(f"\n{'='*50}")
    print(f"📊 AŞAMA 2: HAFTALIK TARAMA ({num_weeks} hafta) + TREND ANALİZİ")
    print(f"{'='*50}")

    today = datetime.now(TIMEZONE).date()
    fon_df = load_takasbank_fund_list()
    if fon_df.empty:
        return [], pd.DataFrame()

    baslangic = today - timedelta(days=(num_weeks * 7) + 30)
    tasks = [(kod, baslangic, today) for kod in fon_df['Fon Kodu'].unique()]
    sonuclar = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_fund_data, t): t[0] for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=" Haftalık+Trend"):
            kod, ad, data = future.result()
            if data is None or data.empty or len(data) < 10:
                continue

            # --- HAFTALIK GETİRİ (DÜZELTİLDİ) ---
            # Hafta_2 (en yeni): bugün / 1 hafta önce
            fiyat_now = get_value_on_or_before(data, today)
            fiyat_1w = get_value_on_or_after(data, today - timedelta(weeks=1))
            fiyat_2w = get_value_on_or_after(data, today - timedelta(weeks=2))
            
            hafta2_getiri = calculate_change(fiyat_now, fiyat_1w)
            hafta1_getiri = calculate_change(fiyat_1w, fiyat_2w)
            toplam_getiri = (hafta1_getiri + hafta2_getiri) if pd.notna(hafta1_getiri) and pd.notna(hafta2_getiri) else np.nan

            # --- 3 GÜNLÜK TREND ANALİZİ (YENİ) ---
            son_3g_ort = np.nan
            onceki_3g_ort = np.nan
            trend_skoru = np.nan
            trend_yonu = "VERI_YOK"

            if len(data) >= TREND_GUN_SAYISI * 2 + 1:
                df_trend = data.copy()
                df_trend['daily_return'] = df_trend['price'].pct_change() * 100
                df_trend = df_trend.dropna().tail(TREND_GUN_SAYISI * 2)

                if len(df_trend) >= TREND_GUN_SAYISI * 2:
                    son_3g = df_trend.tail(TREND_GUN_SAYISI)['daily_return']
                    onceki_3g = df_trend.head(TREND_GUN_SAYISI)['daily_return']
                    son_3g_ort = son_3g.mean()
                    onceki_3g_ort = onceki_3g.mean()

                    if pd.notna(onceki_3g_ort) and onceki_3g_ort != 0:
                        trend_skoru = son_3g_ort / onceki_3g_ort
                    elif son_3g_ort > 0:
                        trend_skoru = son_3g_ort

                    if son_3g_ort > 0 and onceki_3g_ort > 0:
                        trend_yonu = "HIZLANAN" if son_3g_ort > onceki_3g_ort else "YUKSELEN"
                    elif son_3g_ort > 0 and onceki_3g_ort <= 0:
                        trend_yonu = "DONUS"
                    elif son_3g_ort <= 0 and onceki_3g_ort > 0:
                        trend_yonu = "DUSUS"
                    else:
                        trend_yonu = "DUSEN"

            sonuc = {
                'Fon Kodu': kod,
                'Fon Adı': ad,
                'Hafta_1_Getiri': round(hafta1_getiri, 2) if pd.notna(hafta1_getiri) else '',
                'Hafta_2_Getiri': round(hafta2_getiri, 2) if pd.notna(hafta2_getiri) else '',
                'Toplam_Getiri': round(toplam_getiri, 2) if pd.notna(toplam_getiri) else '',
                'Son_3G_Ort_%': round(son_3g_ort, 4) if pd.notna(son_3g_ort) else '',
                'Onceki_3G_Ort_%': round(onceki_3g_ort, 4) if pd.notna(onceki_3g_ort) else '',
                'Trend_Skoru': round(trend_skoru, 4) if pd.notna(trend_skoru) else '',
                'Trend_Yonu': trend_yonu,
            }
            sonuclar.append(sonuc)

    if not sonuclar:
        print("❌ Veri bulunamadı.")
        return [], pd.DataFrame()

    df_sonuc = pd.DataFrame(sonuclar)

    # --- GOOGLE SHEETS: HAFTALIK SAYFASI ---
    haftalik_kolonlar = ['Fon Kodu', 'Fon Adı', 'Hafta_1_Getiri', 'Hafta_2_Getiri', 'Toplam_Getiri']
    df_haftalik = df_sonuc[haftalik_kolonlar].sort_values('Toplam_Getiri', ascending=False, na_position='last')
    write_to_sheet(gc.open_by_key(SHEET_ID), WORKSHEET_WEEKLY, df_haftalik)

    # --- GOOGLE SHEETS: KISA TREND SAYFASI (YENİ) ---
    trend_kolonlar = ['Fon Kodu', 'Fon Adı', 'Trend_Yonu', 'Trend_Skoru',
                       'Son_3G_Ort_%', 'Onceki_3G_Ort_%',
                       'Hafta_1_Getiri', 'Hafta_2_Getiri', 'Toplam_Getiri']
    df_trend = df_sonuc[trend_kolonlar].copy()
    
    # Trend skoruna göre sırala (HIZLANAN/YUKSELEN/DONUS önde)
    trend_sirasi = {'HIZLANAN': 0, 'YUKSELEN': 1, 'DONUS': 2, 'DUSUS': 3, 'DUSEN': 4, 'VERI_YOK': 5}
    df_trend['_sira'] = df_trend['Trend_Yonu'].map(trend_sirasi).fillna(5)
    df_trend = df_trend.sort_values(['_sira', 'Trend_Skoru'], ascending=[True, False]).drop(columns=['_sira'])

    trend_baslik = ['Fon Kodu', 'Fon Adı', 'Trend Yönü', 'Trend Skoru',
                     'Son 3G Ort %', 'Önceki 3G Ort %',
                     'Hafta 1 Getiri %', 'Hafta 2 Getiri %', 'Toplam Getiri %']
    write_to_sheet(gc.open_by_key(SHEET_ID), WORKSHEET_TREND, df_trend, headers=trend_baslik)

    print(f"\n📈 Trend Dağılımı:")
    for yon in ['HIZLANAN', 'YUKSELEN', 'DONUS', 'DUSUS', 'DUSEN', 'VERI_YOK']:
        sayi = len(df_trend[df_trend['Trend_Yonu'] == yon])
        if sayi > 0:
            print(f"  {yon}: {sayi} fon")

    # Filtrele: Toplam getiri >= %2 VE trend yükselişte
    filtre_df = df_sonuc[
        (pd.to_numeric(df_sonuc['Toplam_Getiri'], errors='coerce').fillna(0) >= 2) &
        (df_sonuc['Trend_Yonu'].isin(['HIZLANAN', 'YUKSELEN', 'DONUS']))
    ]
    fonaliz_fonlar = filtre_df['Fon Kodu'].tolist()
    print(f"\n🎯 Fonaliz için seçilen: {len(fonaliz_fonlar)} fon")
    return fonaliz_fonlar, df_sonuc


# --- FONALİZ RİSK ANALİZİ ---
def run_fonaliz(gc, fon_listesi):
    """Filtrelenmiş fonlar için risk/getiri metriklerini hesaplar."""
    if not fon_listesi:
        print("ℹ️ Analiz için fon bulunamadı.")
        return

    print(f"\n{'='*50}")
    print(f"📊 AŞAMA 3: FONALİZ RİSK ANALİZİ ({len(fon_listesi)} fon)")
    print(f"{'='*50}")

    today = datetime.now(TIMEZONE).date()
    baslangic = today - relativedelta(months=3) - timedelta(days=30)
    tasks = [(kod, baslangic, today) for kod in fon_listesi]
    sonuclar = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_fund_data, t): t[0] for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=" Fonaliz Analizi"):
            kod, ad, data = future.result()
            if data is None or data.empty:
                continue
            metrik = hesapla_metrikler(data)
            if metrik:
                sonuclar.append({'Fon Kodu': kod, 'Fon Adı': ad, **metrik})

    if sonuclar:
        df = pd.DataFrame(sonuclar)
        df = df.sort_values(['Sortino (Yıllık)', 'Sharpe (Yıllık)'], ascending=[False, False])
        write_to_sheet(gc.open_by_key(SHEET_ID), WORKSHEET_FONALIZ, df)
        print(f"✅ Fonaliz tamamlandı: {len(df)} fon analiz edildi.")
    else:
        print("ℹ️ Fonaliz için veri bulunamadı.")


# --- ANA ÇALIŞTIRMA ---
if __name__ == '__main__':
    print("=" * 50)
    print("🏦 OtoFon v3.0 - Entegre Fon Tarama ve Analiz")
    print("=" * 50)

    gc = google_sheets_auth()

    scan_type = sys.argv[1].lower() if len(sys.argv) > 1 else 'weekly'

    if scan_type == 'single':
        tarih = None
        if len(sys.argv) > 2:
            try:
                tarih = datetime.strptime(sys.argv[2], '%Y-%m-%d').date()
            except ValueError:
                print("❌ Tarih formatı YYYY-MM-DD olmalı")
                sys.exit(1)
        run_single_scan(gc, tarih)

    elif scan_type == 'weekly':
        hafta = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        
        # AŞAMA 2: Haftalık tarama + trend analizi
        fonaliz_fonlar, _ = run_weekly_scan(gc, hafta)
        
        # AŞAMA 3: Fonaliz risk analizi
        run_fonaliz(gc, fonaliz_fonlar)

        print(f"\n{'='*50}")
        print(f"✅ Tüm işlemler tamamlandı!")
        print(f"{'='*50}")
        print(f"📊 Google Sheets'te güncellenen sayfalar:")
        print(f"  1. '{WORKSHEET_WEEKLY}' - Haftalık getiriler")
        print(f"  2. '{WORKSHEET_TREND}' - Kısa Trend (3 günlük analiz) ⭐ YENİ")
        print(f"  3. '{WORKSHEET_FONALIZ}' - Risk/Getiri metrikleri")
        print(f"📈 {len(fonaliz_fonlar)} fon yükseliş trendinde ve filtreleri geçti.")

    else:
        print(f"❌ Geçersiz parametre: {scan_type}. 'weekly' veya 'single' kullanın.")
