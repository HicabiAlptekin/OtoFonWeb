# -*- coding: utf-8 -*-
# ENTEGRE FON TARAMA ARACI (OtoFon + Fonaliz - GitHub Actions için düzenlendi)

# --- 1. ADIM: Kütüphaneleri Import Etme ---
import pandas as pd
import numpy as np
import time
import gspread
import pytz
import os
import json
import sys
from datetime import datetime, timedelta, date, timezone
from dateutil.relativedelta import relativedelta
from tefas import Crawler
from tqdm import tqdm
import concurrent.futures
import traceback
import warnings

warnings.filterwarnings('ignore') # Bazı kütüphanelerin uyarılarını göz ardı et

# --- Sabitler ve Yapılandırma ---
GSPREAD_CREDENTIALS_SECRET = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
SHEET_ID = '1hSD4towyxKk9QHZFAcRlXy9NlLa_AyVrB9Jsy86ok14' # OtoFon Google Sheet ID'si
WORKSHEET_NAME_MANUAL = 'veriler' # Tekil tarama için
WORKSHEET_NAME_WEEKLY = 'haftalık' # Haftalık tarama için
WORKSHEET_NAME_FONALIZ = 'Fonanaliz' # Fonaliz için
TIMEZONE = pytz.timezone('Europe/Istanbul')
MAX_WORKERS = 10 # Paralel işlemler için maksimum işçi sayısı
TEFAS_CHUNK_DAYS = 90 # TEFAS API'sinden veri çekerken tek seferde çekilecek gün sayısı
TEFAS_MAX_RETRIES = 3 # TEFAS API hatası durumunda maksimum deneme sayısı
TEFAS_RETRY_DELAY = 5 # Yeniden deneme öncesi bekleme süresi (saniye)

# TEFAS'tan çekilecek varsayılan sütunlar (Tekil ve Haftalık taramalar için)
DEFAULT_TEFAS_COLS = ["date", "price"]
# Fonaliz için çekilecek ek sütunlar
FONALIZ_TEFAS_COLS = ["date", "price", "market_cap", "number_of_investors", "title"]


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
    """
    TEFAS API'sinden veri çekerken chunking ve yeniden deneme mantığı içerir.
    Args:
        args: Bir tuple (fon_kodu, start_date_overall, end_date_overall, columns_to_fetch)
    """
    fon_kodu, start_date_overall, end_date_overall, columns_to_fetch = args
    global tefas_crawler_global
    if tefas_crawler_global is None: return fon_kodu, None, pd.DataFrame() # fon_adi, df_data
    
    all_fon_data = pd.DataFrame()
    current_start_date_chunk = start_date_overall

    while current_start_date_chunk <= end_date_overall:
        current_end_date_chunk = min(current_start_date_chunk + timedelta(days=TEFAS_CHUNK_DAYS - 1), end_date_overall)
        retries, success, chunk_data_fetched = 0, False, pd.DataFrame()

        while retries < TEFAS_MAX_RETRIES and not success:
            try:
                if current_start_date_chunk <= current_end_date_chunk:
                    chunk_data_fetched = tefas_crawler_global.fetch(
                        start=current_start_date_chunk.strftime("%Y-%m-%d"),
                        end=current_end_date_chunk.strftime("%Y-%m-%d"),
                        name=fon_kodu,
                        columns=columns_to_fetch
                    )
                if not chunk_data_fetched.empty:
                    all_fon_data = pd.concat([all_fon_data, chunk_data_fetched], ignore_index=True)
                success = True
            except Exception as e:
                #print(f"DEBUG: {fon_kodu} için {current_start_date_chunk}-{current_end_date_chunk} aralığında hata ({retries+1}/{TEFAS_MAX_RETRIES}): {e}")
                retries += 1
                time.sleep(TEFAS_RETRY_DELAY)

        current_start_date_chunk = current_end_date_chunk + timedelta(days=1)

    if not all_fon_data.empty:
        all_fon_data.drop_duplicates(subset=['date', 'price'], keep='first', inplace=True)
        if 'date' in all_fon_data.columns:
            all_fon_data['date'] = pd.to_datetime(all_fon_data['date'], errors='coerce').dt.date
            all_fon_data.dropna(subset=['date'], inplace=True)
        all_fon_data.sort_values(by='date').reset_index(drop=True, inplace=True)
        fon_adi = all_fon_data['title'].iloc[0] if 'title' in all_fon_data.columns and not all_fon_data.empty else fon_kodu
    else:
        fon_adi = fon_kodu # Eğer veri çekilemezse fon_adi olarak kodu kullan
    
    return fon_kodu, fon_adi, all_fon_data

def apply_cell_format_request(worksheet_id, row_index, num_columns, is_highlight):
    if is_highlight:
        text_format = {"foregroundColor": {"red": 1.0, "green": 0.0, "blue": 0.0}, "bold": True}
    else:
        text_format = {"foregroundColor": {"red": 0.0, "green": 0.0, "blue": 0.0}, "bold": False}

    return {
        "repeatCell": {
            "range": {
                "sheetId": worksheet_id,
                "startRowIndex": row_index,
                "endRowIndex": row_index + 1,
                "startColumnIndex": 0,
                "endColumnIndex": num_columns
            },
            "cell": {"userEnteredFormat": {"textFormat": text_format}},
            "fields": "userEnteredFormat.textFormat.foregroundColor,userEnteredFormat.textFormat.bold"
        }
    }

# --- FONALİZ BÖLÜMÜ (tarama_script.py'den entegre edildi) ---
def hesapla_metrikler(df_fon_fiyat):
    """
    Bir fonun geçmiş fiyat verilerini kullanarak risk/getiri metriklerini hesaplar.
    """
    if df_fon_fiyat is None or len(df_fon_fiyat) < 10: return None
    df_fon_fiyat['daily_return'] = df_fon_fiyat['price'].pct_change()
    df_fon_fiyat = df_fon_fiyat.dropna()
    if df_fon_fiyat.empty: return None

    # Getiri hesaplamasında sadece ilk ve son fiyatı kullanarak daha doğru bir yıllık getiri yakala
    # (Günlük getirilerin çarpımı olarak da yapılabilir ama bu daha basit)
    getiri = (df_fon_fiyat['price'].iloc[-1] / df_fon_fiyat['price'].iloc[0]) - 1

    # Yıllık volatilite
    volatilite = df_fon_fiyat['daily_return'].std() * np.sqrt(252) # 252 iş günü varsayımı
    
    # Ortalama günlük getiri (risksiz faiz oranı sıfır kabul edilerek)
    ortalama_gunluk_getiri = df_fon_fiyat['daily_return'].mean()

    # Sharpe Oranı
    sharpe_orani = (ortalama_gunluk_getiri / df_fon_fiyat['daily_return'].std()) * np.sqrt(252) if df_fon_fiyat['daily_return'].std() != 0 else 0
    
    # Sortino Oranı
    negatif_getiriler = df_fon_fiyat[df_fon_fiyat['daily_return'] < 0]['daily_return']
    if negatif_getiriler.empty or negatif_getiriler.std() == 0:
        sortino_orani = 0
    else:
        downside_deviation = negatif_getiriler.std() * np.sqrt(252)
        # Yıllıklandırılmış ortalama getiri kullanıldı
        sortino_orani = (ortalama_gunluk_getiri * 252) / downside_deviation if downside_deviation != 0 else 0
    
    # Piyasa değeri ve yatırımcı sayısı son günkü veriler
    piyasa_degeri = df_fon_fiyat['market_cap'].iloc[-1] if 'market_cap' in df_fon_fiyat.columns else np.nan
    yatirimci_sayisi = df_fon_fiyat['number_of_investors'].iloc[-1] if 'number_of_investors' in df_fon_fiyat.columns else np.nan

    return {
        'Getiri (%)': round(getiri * 100, 2),
        'Standart Sapma (Yıllık %)': round(volatilite * 100, 2),
        'Sharpe Oranı (Yıllık)': round(sharpe_orani, 2),
        'Sortino Oranı (Yıllık)': round(sortino_orani, 2),
        'Piyasa Değeri (TL)': piyasa_degeri,
        'Yatırımcı Sayısı': yatirimci_sayisi
    }

def run_fonaliz_scan_to_gsheets(fon_listesi: list, gc):
    """
    Verilen fon listesi için Fonaliz metriklerini hesaplar ve Google Sheets'e yazar.
    """
    print("\n" + "="*40)
    print("     AŞAMA 3: FONALİZ RİSK ANALİZİ BAŞLATILIYOR")
    print(f"     {len(fon_listesi)} adet filtrelenmiş fon analiz edilecek...")
    print("="*40)
    
    if not fon_listesi:
        print("ℹ️ Fonaliz için filtreden geçen fon bulunamadı. İşlem atlanıyor.")
        return

    ANALIZ_SURESI_AY = 3 # Son 3 aylık veriler analiz edilecek
    end_date = datetime.now(TIMEZONE).date()
    # Fonaliz için 3 ay + ek tampon (TEFAS_CHUNK_DAYS kadar) geriye dönük veri çekiyoruz.
    # Bu, ilk birkaç günün eksik gelmesi veya API hatalarından korunmak için.
    start_date = end_date - pd.DateOffset(months=ANALIZ_SURESI_AY) - timedelta(days=TEFAS_CHUNK_DAYS)
    
    # Fonaliz için FONALIZ_TEFAS_COLS sütunlarını istiyoruz
    tasks = [(fon_kodu, start_date, end_date, FONALIZ_TEFAS_COLS) for fon_kodu in fon_listesi]
    analiz_sonuclari = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, task): task[0] for task in tasks}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon), total=len(tasks), desc=" Fonaliz Risk Analizi")

        for future in progress_bar:
            fon_kodu, fon_adi, data = future.result()
            if data is not None and not data.empty:
                metrikler = hesapla_metrikler(data)
                if metrikler:
                    sonuc = {'Fon Kodu': fon_kodu, 'Fon Adı': fon_adi, **metrikler}
                    analiz_sonuclari.append(sonuc)
            #else:
                #print(f"DEBUG: {fon_kodu} için Fonaliz verisi bulunamadı veya yetersiz.")

    if not analiz_sonuclari:
        print("\n--- SONUÇ: Fonaliz için analiz edilecek yeterli veri bulunamadı. ---")
        return

    df_sonuc = pd.DataFrame(analiz_sonuclari)
    sutun_sirasi = ['Fon Kodu', 'Fon Adı', 'Yatırımcı Sayısı', 'Piyasa Değeri (TL)', 'Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)', 'Getiri (%)', 'Standart Sapma (Yıllık %)']
    # Sadece var olan sütunları al
    sutun_sirasi = [col for col in sutun_sirasi if col in df_sonuc.columns]
    
    df_sonuc = df_sonuc[sutun_sirasi]
    df_sonuc_sirali = df_sonuc.sort_values(by=['Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)'], ascending=[False, False])

    print(f"\n✅ Fonaliz tamamlandı. Sonuçlar Google Sheets'teki '{WORKSHEET_NAME_FONALIZ}' sayfasına yazılıyor...")
    try:
        spreadsheet = gc.open_by_key(SHEET_ID)
        try:
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME_FONALIZ)
        except gspread.exceptions.WorksheetNotFound:
            print(f"ℹ️ '{WORKSHEET_NAME_FONALIZ}' sayfası bulunamadı, yeni sayfa oluşturuluyor...")
            worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME_FONALIZ, rows="1000", cols=20)
        
        worksheet.clear()
        df_sonuc_sirali = df_sonuc_sirali.replace([np.inf, -np.inf], np.nan).fillna('')
        worksheet.update([df_sonuc_sirali.columns.values.tolist()] + df_sonuc_sirali.values.tolist())
        
        body_resize = {"requests": [{"autoResizeDimensions": {"dimensions": {"sheetId": worksheet.id, "dimension": "COLUMNS"}}}]}
        spreadsheet.batch_update(body_resize)
        print(f"✅ Google Sheets '{WORKSHEET_NAME_FONALIZ}' sayfası güncellendi ve sütunlar yeniden boyutlandırıldı.")
    except Exception as e:
        print(f"❌ Google Sheets'e yazma hatası (Fonaliz): {e}")
        traceback.print_exc()


# --- HAFTALIK TARAMA FONKSİYONU ---
def run_weekly_scan_to_gsheets(num_weeks: int, gc):
    start_time_main = time.time()
    today = datetime.now(TIMEZONE).date()
    all_fon_data_df = load_takasbank_fund_list()

    if all_fon_data_df.empty:
        print("❌ Taranacak fon listesi alınamadı. İşlem durduruldu.")
        return pd.DataFrame() # Boş DataFrame döndür

    print(f"\n" + "="*40)
    print(f"     AŞAMA 2: HAFTALIK TARAMA BAŞLATILIYOR | {num_weeks} Hafta Geriye Dönük")
    print("="*40)

    total_fon_count = len(all_fon_data_df)
    # Haftalık tarama için gerekli olan en eski tarihi belirle.
    # num_weeks * 7 + 21 (ek tampon) gün kadar geriye gidiyoruz.
    genel_veri_cekme_baslangic_tarihi = today - timedelta(days=(num_weeks * 7) + 21 + TEFAS_CHUNK_DAYS) 
    
    # Haftalık tarama için DEFAULT_TEFAS_COLS sütunlarını istiyoruz
    fon_args_list = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, today, DEFAULT_TEFAS_COLS)
                      for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]

    weekly_results_dict = {}
    first_fund_calculated_columns = []
    first_fund_processed = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # fetch_data_for_fund_parallel şimdi fon_kodu, fon_adi, df_data döndürüyor
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in fon_args_list}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon),
                              total=total_fon_count,
                              desc=" Haftalık Fonları Tarıyor")

        for future in progress_bar:
            fon_kodu_completed, fon_adi_fetched, fund_history = future.result()
            try:
                if fund_history is None or fund_history.empty:
                    #print(f"DEBUG: {fon_kodu_completed} için haftalık tarama verisi bulunamadı veya yetersiz.")
                    continue

                current_fon_data = {'Fon Kodu': fon_kodu_completed, 'Fon Adı': fon_adi_fetched if fon_adi_fetched else fon_kodu_completed}
                calculated_cols_current_fund, weekly_changes_list = [], []
                first_week_end_price, last_week_start_price = np.nan, np.nan
                current_week_end_date_cal = today

                for i in range(num_weeks):
                    current_week_start_date_cal = current_week_end_date_cal - timedelta(days=7)
                    price_end = get_price_on_or_before(fund_history, current_week_end_date_cal)
                    price_start = get_price_on_or_before(fund_history, current_week_start_date_cal)

                    if i == 0: first_week_end_price = price_end
                    if i == num_weeks - 1: last_week_start_price = price_start

                    # Sütun adı formatını güncelleyelim: DD.MM-DD.MM.YY
                    col_name = f"{current_week_end_date_cal.day:02d}.{current_week_start_date_cal.month:02d}-{current_week_end_date_cal.day:02d}.{current_week_end_date_cal.month:02d}.{current_week_end_date_cal.year % 100:02d}"
                    weekly_change = calculate_change(price_end, price_start)
                    current_fon_data[col_name] = weekly_change
                    weekly_changes_list.append(weekly_change)
                    calculated_cols_current_fund.append(col_name)
                    current_week_end_date_cal = current_week_start_date_cal

                if not first_fund_processed and calculated_cols_current_fund:
                    first_fund_calculated_columns = calculated_cols_current_fund
                    first_fund_processed = True

                current_fon_data['Değerlendirme'] = calculate_change(first_week_end_price, last_week_start_price)
                is_desired_trend = False
                valid_changes = [chg for chg in weekly_changes_list if not pd.isna(chg)]

                # Yeterli veri varsa ve num_weeks >= 2 ise trend kontrolü yap
                if len(valid_changes) == num_weeks and num_weeks >= 2:
                    # Azalan getiri trendi kontrolü (Hafta_1 > Hafta_2 > ...)
                    if all(valid_changes[j] > valid_changes[j+1] for j in range(num_weeks - 1)):
                        is_desired_trend = True
                
                # --- DEBUG SÜTUNLARI ---
                current_fon_data['is_desired_trend'] = bool(is_desired_trend)
                current_fon_data['_DEBUG_WeeklyChanges_RAW'] = "'" + str([f"{x:.2f}" if not pd.isna(x) else "NaN" for x in weekly_changes_list])
                current_fon_data['_DEBUG_IsDesiredTrend'] = bool(is_desired_trend)
                weekly_results_dict[fon_kodu_completed] = current_fon_data
            except Exception as exc:
                print(f"❌ Hata (Haftalık - {fon_kodu_completed}): {exc}")
                traceback.print_exc()

    results_df = pd.DataFrame(list(weekly_results_dict.values()))

    if not first_fund_calculated_columns and not results_df.empty:
        # Eğer ilk fon için hesaplanan sütunlar alınamadıysa, DataFrame'den çek
        temp_row_cols = [col for col in results_df.columns if col not in ['Fon Kodu', 'Fon Adı', 'Değerlendirme', 'is_desired_trend', '_DEBUG_WeeklyChanges_RAW', '_DEBUG_IsDesiredTrend']]
        first_fund_calculated_columns = temp_row_cols if temp_row_cols else []

    base_cols = ['Fon Kodu', 'Fon Adı']
    debug_cols = ['_DEBUG_WeeklyChanges_RAW', '_DEBUG_IsDesiredTrend']
    final_view_columns = base_cols + first_fund_calculated_columns + ['Değerlendirme'] + debug_cols
    all_df_columns = final_view_columns + ['is_desired_trend'] # is_desired_trend Google Sheets'e yazılmıyor ama formatlama için lazım
    existing_cols_for_df = [col for col in all_df_columns if col in results_df.columns]

    if not results_df.empty:
        results_df = results_df[existing_cols_for_df]
        results_df.sort_values(by='Değerlendirme', ascending=False, na_position='last', inplace=True)
    else:
        results_df = pd.DataFrame(columns=existing_cols_for_df)

    # float64 ve inf/-inf değerlerini temizle
    for col in results_df.columns:
        if results_df[col].dtype == 'float64':
            results_df[col] = results_df[col].replace([np.inf, -np.inf], np.nan).astype(object).where(pd.notna(results_df[col]), None)
        if col in ['is_desired_trend', '_DEBUG_IsDesiredTrend']:
            results_df[col] = results_df[col].astype(bool)

    print(f"\n\n✅ Haftalık tarama tamamlandı. {len(results_df)} fon için sonuçlar hesaplandı.")
    print(f" Sonuçlar Google Sheets'teki '{WORKSHEET_NAME_WEEKLY}' sayfasına yazılıyor...")

    try:
        spreadsheet = gc.open_by_key(SHEET_ID)
        try:
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME_WEEKLY)
        except gspread.exceptions.WorksheetNotFound:
            print(f"ℹ️ '{WORKSHEET_NAME_WEEKLY}' sayfası bulunamadı, yeni sayfa oluşturuluyor...")
            worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME_WEEKLY, rows="1000", cols=50)
        
        worksheet.clear()
        
        # Google Sheets'e yazılacak DataFrame'i oluştur (is_desired_trend hariç)
        df_to_gsheets = results_df[[col for col in final_view_columns if col in results_df.columns]]

        if not df_to_gsheets.empty:
            worksheet.update(values=[df_to_gsheets.columns.values.tolist()] + df_to_gsheets.values.tolist(), value_input_option='USER_ENTERED')
            
            # is_desired_trend değeri True olan satırları vurgula
            format_requests = [apply_cell_format_request(worksheet.id, idx + 1, len(df_to_gsheets.columns), True)
                               for idx, row in results_df.reset_index(drop=True).iterrows() if row.get('is_desired_trend', False)]
            
            if format_requests:
                spreadsheet.batch_update({"requests": format_requests})
                print(f"✅ {len(format_requests)} satır, istenen trende uyduğu için işaretlendi.")
            
            body_resize = {"requests": [{"autoResizeDimensions": {"dimensions": {"sheetId": worksheet.id, "dimension": "COLUMNS"}}}]}
            spreadsheet.batch_update(body_resize)
            print(f"✅ Google Sheets '{WORKSHEET_NAME_WEEKLY}' sayfası güncellendi ve sütunlar yeniden boyutlandırıldı.")
        else:
            print("ℹ️ Haftalık tarama sonucunda Google Sheets'e yazılacak veri bulunamadı.")
    except Exception as e:
        print(f"❌ Google Sheets'e yazma hatası (Haftalık): {e}")
        traceback.print_exc()
    
    print(f"--- Haftalık Tarama Bitti. Toplam Süre: {time.time() - start_time_main:.2f} saniye ---")
    
    # Haftalık tarama sonucunda filtreleyerek Fonaliz için geri dönen fonları döndür
    # Sadece toplam getirisi >= 2 olan fonları Fonaliz için döndürüyoruz
    results_df['Toplam_Getiri'] = results_df['Değerlendirme'].fillna(0) # 'Değerlendirme' artık toplam getiri
    filtrelenmis_df_fonaliz = results_df[results_df['Toplam_Getiri'] >= 2].copy()

    if filtrelenmis_df_fonaliz.empty:
        print("ℹ️ Fonaliz için filtreyi geçen fon bulunamadı. Boş liste döndürülüyor.")
        return []
    
    return filtrelenmis_df_fonaliz['Fon Kodu'].tolist()


# --- TEKİL TARİH TARAMA FONKSİYONU ---
def run_single_date_scan_to_gsheets(scan_date: date, gc):
    start_time_main = time.time()
    all_fon_data_df = load_takasbank_fund_list()

    if all_fon_data_df.empty:
        print("❌ Taranacak fon listesi alınamadı. İşlem durduruldu.")
        return pd.DataFrame()

    print(f"\n" + "="*40)
    print(f"     AŞAMA 1: TEKİL TARAMA BAŞLATILIYOR | Bitiş Tarihi: {scan_date.strftime('%d.%m.%Y')}")
    print("="*40)

    total_fon_count = len(all_fon_data_df)
    # Tekil tarama için 1 yıl 2 ay + ek tampon (TEFAS_CHUNK_DAYS kadar) geriye dönük veri çekiyoruz.
    genel_veri_cekme_baslangic_tarihi = scan_date - relativedelta(years=1, months=2) - timedelta(days=TEFAS_CHUNK_DAYS)
    
    # Tekil tarama için DEFAULT_TEFAS_COLS sütunlarını istiyoruz
    fon_args_list = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, scan_date, DEFAULT_TEFAS_COLS)
                    for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]

    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in fon_args_list}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon),
                          total=total_fon_count,
                          desc=" Tekil Fonları Tarıyor")

        for future in progress_bar:
            fon_kodu_completed, fon_adi_fetched, fund_history = future.result()
            try:
                if fund_history is None or fund_history.empty:
                    #print(f"DEBUG: {fon_kodu_completed} için tekil tarama verisi bulunamadı veya yetersiz.")
                    continue

                fiyat_son = get_price_on_or_before(fund_history, scan_date)
                
                degisimler = {}
                periods = {
                    'Günlük %': timedelta(days=1), 'Haftalık %': timedelta(weeks=1),
                    '2 Haftalık %': timedelta(weeks=2), 'Aylık %': relativedelta(months=1),
                    '3 Aylık %': relativedelta(months=3), '6 Aylık %': relativedelta(months=6),
                    '1 Yıllık %': relativedelta(years=1)
                }
                
                if not pd.isna(fiyat_son):
                    for name, period_delta in periods.items():
                        target_date = scan_date - period_
