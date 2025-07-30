# -*- coding: utf-8 -*-
<<<<<<< HEAD
# OTOMATİK FON TARAMA ARACI (GitHub Actions için düzenlendi)

# --- 1. ADIM: Kütüphaneleri Import Etme ---
=======
# GÜNCELLENMİŞ FON TARAMA ARACI (GitHub Actions Uyumlu)

# --- Kütüphaneleri Import Etme ---
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
import pandas as pd
import numpy as np
import time
import gspread
import pytz
<<<<<<< HEAD
import os
import json
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from tefas import Crawler
from tqdm import tqdm
import concurrent.futures
import traceback

# --- Sabitler ve Yapılandırma ---
# GitHub Actions'da bu isimde bir secret oluşturulmalıdır.
# Secret içeriği, Google Cloud'dan indirilen JSON anahtarının metin hali olmalıdır.
GSPREAD_CREDENTIALS_SECRET = os.environ.get('GSPREAD_CREDENTIALS')
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
F_COLS = ["date", "price"]
SHEET_ID = '1hSD4towyxKk9QHZFAcRlXy9NlLa_AyVrB9Jsy86ok14' # Google Sheet ID'niz
=======
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from tefas import Crawler
# from google.colab import auth # Colab'a özel olduğu için kaldırıldı
from tqdm import tqdm
import concurrent.futures
import traceback
import os # Ortam değişkenlerini okumak için eklendi
import json # JSON anahtarını işlemek için eklendi
import sys # Script'i hata ile sonlandırmak için eklendi

# --- Sabitler ---
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
F_COLS = ["date", "price"]
SHEET_ID = '1hSD4towyxKk9QHZFAcRlXy9NlLa_AyVrB9Jsy86ok14'
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
WORKSHEET_NAME_MANUAL = 'veriler'
WORKSHEET_NAME_WEEKLY = 'haftalık'
TIMEZONE = pytz.timezone('Europe/Istanbul')

<<<<<<< HEAD
# --- Yardımcı Fonksiyonlar ---
def google_sheets_auth():
    """
    Google Sheets için kimlik doğrulaması yapar.
    GitHub Actions ortamında, environment variable olarak saklanan
    servis hesabı bilgilerini kullanarak kimlik doğrular.
    """
    print("\nGoogle Sheets için kimlik doğrulaması yapılıyor...")
    try:
        if not GSPREAD_CREDENTIALS_SECRET:
            print("❌ Hata: GSPREAD_CREDENTIALS secret bulunamadı.")
            print("Lütfen GitHub deponuzun Ayarlar -> Secrets and variables -> Actions bölümünden bu secret'ı oluşturun.")
            return None

        # Secret içeriğini bir JSON nesnesine çevir
        creds_json = json.loads(GSPREAD_CREDENTIALS_SECRET)
        
        # gspread'i servis hesabı ile yetkilendir
        gc = gspread.service_account_from_dict(creds_json)
=======
# --- Google Sheets Kimlik Doğrulama Fonksiyonu (GitHub Actions Uyumlu) ---
def google_sheets_auth_github():
    print("\n🔄 Google Hizmet Hesabı ile kimlik doğrulaması yapılıyor...")
    try:
        # GitHub Secrets'tan gelen JSON anahtarını al
        gcp_service_account_key_json = os.getenv('GCP_SERVICE_ACCOUNT_KEY')

        if not gcp_service_account_key_json:
            print("❌ Hata: GCP_SERVICE_ACCOUNT_KEY ortam değişkeni ayarlanmamış.")
            sys.exit(1) # Kritik hata, script'i sonlandır

        credentials = json.loads(gcp_service_account_key_json)
        gc = gspread.service_account_from_dict(credentials)
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
        print("✅ Kimlik doğrulama başarılı.")
        return gc
    except Exception as e:
        print(f"❌ Kimlik doğrulama sırasında hata oluştu: {e}")
        traceback.print_exc()
<<<<<<< HEAD
        return None

=======
        sys.exit(1) # Kritik hata, script'i sonlandır

# --- TEFAS Crawler Başlatma ---
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
try:
    tefas_crawler_global = Crawler()
    print("TEFAS Crawler başarıyla başlatıldı.")
except Exception as e:
    print(f"TEFAS Crawler başlatılırken hata: {e}")
<<<<<<< HEAD
    tefas_crawler_global = None

def load_takasbank_fund_list():
    print(f"Takasbank'tan güncel fon listesi yükleniyor...")
=======
    traceback.print_exc()
    tefas_crawler_global = None # Hata durumunda None olarak bırak

# --- Yardımcı Fonksiyonlar (Değişiklik Yok) ---
def load_takasbank_fund_list():
    print(f"🔄 Takasbank'tan güncel fon listesi yükleniyor...")
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
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
<<<<<<< HEAD
        return pd.DataFrame()

def get_price_on_or_before(df_fund_history, target_date: date):
    if df_fund_history is None or df_fund_history.empty or target_date is None: return np.nan
    df_filtered = df_fund_history[df_fund_history['date'] <= target_date].copy()
    if not df_filtered.empty: return df_filtered['price'].iloc[0]
=======
        traceback.print_exc()
        return pd.DataFrame()

def get_first_price_looking_back(df_fund_history, end_date: date, max_lookback_days: int = 6):
    if df_fund_history is None or df_fund_history.empty: return np.nan
    start_lookback_date = end_date - timedelta(days=max_lookback_days)
    end_lookback_date = end_date - timedelta(days=1)
    relevant_history = df_fund_history[(df_fund_history['date'] >= start_lookback_date) & (df_fund_history['date'] <= end_lookback_date)]
    if not relevant_history.empty: return relevant_history['price'].iloc[0]
    return np.nan

def get_price_on_or_before(df_fund_history, target_date: date):
    if df_fund_history is None or df_fund_history.empty or target_date is None: return np.nan
    df_filtered = df_fund_history[df_fund_history['date'] <= target_date].copy()
    if not df_filtered.empty:
        # En güncel tarihi almak için sondan ilk elemanı seç
        return df_filtered.sort_values(by='date', ascending=False)['price'].iloc[0]
    return np.nan

def get_price_at_date_or_next_available(df_fund_history, target_date: date, max_lookforward_days: int = 5):
    if df_fund_history is None or df_fund_history.empty or target_date is None: return np.nan
    future_limit_date = target_date + timedelta(days=max_lookforward_days)
    first_available = df_fund_history[(df_fund_history['date'] >= target_date) & (df_fund_history['date'] <= future_limit_date)].sort_values(by='date', ascending=True)
    if not first_available.empty: return first_available['price'].iloc[0]
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
    return np.nan

def calculate_change(current_price, past_price):
    if pd.isna(current_price) or pd.isna(past_price) or past_price is None or current_price is None: return np.nan
    try:
        current_price_float, past_price_float = float(current_price), float(past_price)
        if past_price_float == 0: return np.nan
        return ((current_price_float - past_price_float) / past_price_float) * 100
    except (ValueError, TypeError): return np.nan

def fetch_data_for_fund_parallel(args):
    fon_kodu, start_date_overall, end_date_overall, chunk_days, max_retries, retry_delay = args
    global tefas_crawler_global
    if tefas_crawler_global is None: return fon_kodu, pd.DataFrame()

    all_fon_data = pd.DataFrame()
    current_start_date_chunk = start_date_overall

    while current_start_date_chunk <= end_date_overall:
        current_end_date_chunk = min(current_start_date_chunk + timedelta(days=chunk_days - 1), end_date_overall)
        retries, success, chunk_data_fetched = 0, False, pd.DataFrame()

        while retries < max_retries and not success:
            try:
                if current_start_date_chunk <= current_end_date_chunk:
                    chunk_data_fetched = tefas_crawler_global.fetch(
                        start=current_start_date_chunk.strftime("%Y-%m-%d"),
                        end=current_end_date_chunk.strftime("%Y-%m-%d"),
                        name=fon_kodu,
                        columns=F_COLS
                    )
                if not chunk_data_fetched.empty:
                    all_fon_data = pd.concat([all_fon_data, chunk_data_fetched], ignore_index=True)
                success = True
<<<<<<< HEAD
            except Exception:
=======
            except Exception as e:
                print(f"Hata fon çekme ({fon_kodu}, {current_start_date_chunk}-{current_end_date_chunk}): {e}")
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
                retries += 1
                time.sleep(retry_delay)

        current_start_date_chunk = current_end_date_chunk + timedelta(days=1)

    if not all_fon_data.empty:
        all_fon_data.drop_duplicates(subset=['date', 'price'], keep='first', inplace=True)
        if 'date' in all_fon_data.columns:
            all_fon_data['date'] = pd.to_datetime(all_fon_data['date'], errors='coerce').dt.date
            all_fon_data.dropna(subset=['date'], inplace=True)
        all_fon_data.sort_values(by='date', ascending=False, inplace=True)

    return fon_kodu, all_fon_data

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

# --- HAFTALIK TARAMA FONKSİYONU ---
def run_weekly_scan_to_gsheets(num_weeks: int, gc):
    start_time_main = time.time()
    today = datetime.now(TIMEZONE).date()
    all_fon_data_df = load_takasbank_fund_list()

    if all_fon_data_df.empty:
        print("❌ Taranacak fon listesi alınamadı. İşlem durduruldu.")
        return

    print(f"\n--- HAFTALIK TARAMA BAŞLATILIYOR | {num_weeks} Hafta Geriye Dönük ---")

    total_fon_count = len(all_fon_data_df)
    genel_veri_cekme_baslangic_tarihi = today - timedelta(days=(num_weeks * 7) + 21)
    fon_args_list = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, today, 30, 3, 5)
<<<<<<< HEAD
                    for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]
=======
                      for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e

    MAX_WORKERS = 10
    weekly_results_dict = {}
    first_fund_calculated_columns = []
    first_fund_processed = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in fon_args_list}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon),
<<<<<<< HEAD
                          total=total_fon_count,
                          desc=" Fonlar Taranıyor (Haftalık)")
=======
                              total=total_fon_count,
                              desc="🔎 Fonlar Taranıyor (Haftalık)")
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e

        for future in progress_bar:
            fon_kodu_completed = future_to_fon[future]
            try:
                _, fund_history = future.result()
                fon_adi = all_fon_data_df.loc[all_fon_data_df['Fon Kodu'] == fon_kodu_completed, 'Fon Adı'].iloc[0]
<<<<<<< HEAD
                
=======
                if fon_kodu_completed not in all_fon_data_df['Fon Kodu'].values:
                    fon_adi = "Bilinmiyor"

>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
                current_fon_data = {'Fon Kodu': fon_kodu_completed, 'Fon Adı': fon_adi}
                calculated_cols_current_fund, weekly_changes_list = [], []
                first_week_end_price, last_week_start_price = np.nan, np.nan
                current_week_end_date_cal = today

                for i in range(num_weeks):
                    current_week_start_date_cal = current_week_end_date_cal - timedelta(days=7)
                    price_end = get_price_on_or_before(fund_history, current_week_end_date_cal)
                    price_start = get_price_on_or_before(fund_history, current_week_start_date_cal)

<<<<<<< HEAD
                    if i == 0: first_week_end_price = price_end
                    if i == num_weeks - 1: last_week_start_price = price_start

                    col_name = f"{current_week_end_date_cal.day:02d}.{current_week_end_date_cal.month:02d}-{current_week_start_date_cal.day:02d}.{current_week_start_date_cal.month:02d}"
=======
                    if i == 0:
                        first_week_end_price = price_end
                    if i == num_weeks - 1:
                        last_week_start_price = price_start

                    col_name = f"{current_week_end_date_cal.day:02d}-{current_week_start_date_cal.day:02d}/{current_week_end_date_cal.year % 100:02d}"
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
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

                if len(valid_changes) == num_weeks and num_weeks >= 2:
                    if all(valid_changes[j] > valid_changes[j+1] for j in range(num_weeks - 1)):
                        is_desired_trend = True
<<<<<<< HEAD
                
                current_fon_data['is_desired_trend'] = bool(is_desired_trend)
                weekly_results_dict[fon_kodu_completed] = current_fon_data
            except Exception as exc:
                print(f"Hata (Haftalık - {fon_kodu_completed}): {exc}")
=======

                current_fon_data['is_desired_trend'] = bool(is_desired_trend)
                current_fon_data['_DEBUG_WeeklyChanges_RAW'] = "'" + str([f"{x:.2f}" if not pd.isna(x) else "NaN" for x in weekly_changes_list])
                current_fon_data['_DEBUG_IsDesiredTrend'] = bool(is_desired_trend)
                weekly_results_dict[fon_kodu_completed] = current_fon_data
            except Exception as exc:
                print(f"Hata (Haftalık - {fon_kodu_completed}): {exc}")
                traceback.print_exc()
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e

    results_df = pd.DataFrame(list(weekly_results_dict.values()))

    if not first_fund_calculated_columns and not results_df.empty:
<<<<<<< HEAD
        temp_row_cols = [col for col in results_df.columns if col not in ['Fon Kodu', 'Fon Adı', 'Değerlendirme', 'is_desired_trend']]
        first_fund_calculated_columns = temp_row_cols if temp_row_cols else []

    base_cols = ['Fon Kodu', 'Fon Adı']
    final_view_columns = base_cols + first_fund_calculated_columns + ['Değerlendirme']
=======
        temp_row_cols = [col for col in results_df.columns
                         if col not in ['Fon Kodu', 'Fon Adı', 'Değerlendirme',
                                         'is_desired_trend', '_DEBUG_WeeklyChanges_RAW',
                                         '_DEBUG_IsDesiredTrend']]
        first_fund_calculated_columns = temp_row_cols if temp_row_cols else []

    base_cols = ['Fon Kodu', 'Fon Adı']
    debug_cols = ['_DEBUG_WeeklyChanges_RAW', '_DEBUG_IsDesiredTrend']
    final_view_columns = base_cols + first_fund_calculated_columns + ['Değerlendirme'] + debug_cols
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
    all_df_columns = final_view_columns + ['is_desired_trend']
    existing_cols_for_df = [col for col in all_df_columns if col in results_df.columns]

    if not results_df.empty:
        results_df = results_df[existing_cols_for_df]
        results_df.sort_values(by='Değerlendirme', ascending=False, na_position='last', inplace=True)
    else:
        results_df = pd.DataFrame(columns=existing_cols_for_df)

    for col in results_df.columns:
        if results_df[col].dtype == 'float64':
            results_df[col] = results_df[col].replace([np.inf, -np.inf], np.nan).astype(object).where(pd.notna(results_df[col]), None)
<<<<<<< HEAD
        if col == 'is_desired_trend':
            results_df[col] = results_df[col].astype(bool)

    print(f"\n✅ Haftalık tarama tamamlandı. {len(results_df)} fon için sonuçlar hesaplandı.")
    print(f"Sonuçlar Google Sheets'teki '{WORKSHEET_NAME_WEEKLY}' sayfasına yazılıyor...")

    try:
        spreadsheet = gc.open_by_key(SHEET_ID)
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME_WEEKLY)
        worksheet.clear()
        
        df_to_gsheets = results_df[[col for col in final_view_columns if col in results_df.columns]]

        if not df_to_gsheets.empty:
            worksheet.update(values=[df_to_gsheets.columns.values.tolist()] + df_to_gsheets.values.tolist(), value_input_option='USER_ENTERED')
            
            format_requests = [apply_cell_format_request(worksheet.id, idx + 1, len(df_to_gsheets.columns), True)
                               for idx, row in results_df.reset_index(drop=True).iterrows() if row.get('is_desired_trend', False)]
            
            if format_requests:
                spreadsheet.batch_update({"requests": format_requests})
                print(f"✅ {len(format_requests)} satır, istenen trende uyduğu için işaretlendi.")
            
            body_resize = {"requests": [{"autoResizeDimensions": {"dimensions": {"sheetId": worksheet.id, "dimension": "COLUMNS"}}}]}
            spreadsheet.batch_update(body_resize)
            print("✅ Google Sheets güncellendi ve sütunlar yeniden boyutlandırıldı.")
    except Exception as e:
        print(f"❌ Google Sheets'e yazma hatası (Haftalık): {e}")
    
    print(f"--- Haftalık Tarama Bitti. Toplam Süre: {time.time() - start_time_main:.2f} saniye ---")


# --- TEKİL TARİH TARAMA FONKSİYONU ---
def run_single_date_scan_to_gsheets(scan_date: date, gc):
=======
        elif results_df[col].dtype == 'object' and col not in ['is_desired_trend', '_DEBUG_IsDesiredTrend', '_DEBUG_WeeklyChanges_RAW']:
            results_df[col] = results_df[col].apply(lambda x: None if (isinstance(x, str) and (x.lower() in ['nan', 'nat'])) or pd.isna(x) else x)
        if col in ['is_desired_trend', '_DEBUG_IsDesiredTrend']:
            results_df[col] = results_df[col].astype(bool)

    print(f"\n\n✅ Haftalık tarama tamamlandı. {len(results_df)} fon için sonuçlar hesaplandı.")
    print(f"🔄 Sonuçlar Google Sheets'teki '{WORKSHEET_NAME_WEEKLY}' sayfasına yazılıyor...")

    try:
        spreadsheet = gc.open_by_key(SHEET_ID)
        try:
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME_WEEKLY)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME_WEEKLY, rows="1000", cols=max(100, len(final_view_columns) + 5))
        worksheet.clear()

        df_to_gsheets = results_df[[col for col in final_view_columns if col in results_df.columns]]

        if not df_to_gsheets.empty:
            worksheet.update(values=[df_to_gsheets.columns.values.tolist()] + df_to_gsheets.values.tolist(),
                             value_input_option='USER_ENTERED')

            format_requests = []
            for idx, row in results_df.reset_index(drop=True).iterrows():
                if row.get('is_desired_trend', False):
                    format_requests.append(apply_cell_format_request(worksheet.id, idx + 1, len(df_to_gsheets.columns), True))

            if format_requests:
                spreadsheet.batch_update({"requests": format_requests})
                print(f"✅ {len(format_requests)} satır, istenen trende uyduğu için işaretlendi.")
            else:
                print("ℹ️ İstenen trende (H1>H2>...) uyan hiçbir fon bulunamadı.")

            body_resize = {"requests": [{"autoResizeDimensions": {"dimensions": {"sheetId": worksheet.id,
                                                                              "dimension": "COLUMNS",
                                                                              "startIndex": 0,
                                                                              "endIndex": len(df_to_gsheets.columns)}}}]}
            spreadsheet.batch_update(body_resize)
        else:
            print("ℹ️ Google Sheets'e yazılacak veri bulunmuyor.")

        end_time_main = time.time()
        print("\n" + "="*50 +
              f"\n🎉 HAFTALIK TARAMA BAŞARIYLA TAMAMLANDI! ({datetime.now(TIMEZONE).strftime('%d.%m.%Y %H:%M:%S')})\n" +
              f"⏱️ Toplam süre: {((end_time_main - start_time_main) / 60):.2f} dakika\n" +
              "="*50)
    except Exception as e:
        print(f"❌ Google Sheets'e yazma/formatlama sırasında hata: {e}")
        traceback.print_exc()
        sys.exit(1) # Hata durumunda script'i sonlandır

# --- TEKİL TARAMA FONKSİYONU ---
def run_scan_to_gsheets(scan_date: date, gc):
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
    start_time_main = time.time()
    all_fon_data_df = load_takasbank_fund_list()

    if all_fon_data_df.empty:
<<<<<<< HEAD
        print("❌ Taranacak fon listesi alınamadı. İşlem durduruldu.")
        return

    print(f"\n--- TEKİL TARAMA BAŞLATILIYOR | Bitiş Tarihi: {scan_date.strftime('%d.%m.%Y')} ---")

    total_fon_count = len(all_fon_data_df)
    genel_veri_cekme_baslangic_tarihi = scan_date - relativedelta(years=1, months=1, days=15)
    fon_args_list = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, scan_date, 30, 3, 5)
                    for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]

    MAX_WORKERS = 10
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in fon_args_list}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon),
                          total=total_fon_count,
                          desc=" Fonlar Taranıyor (Tekil)")
=======
        print("❌ Taranacak fon listesi alınamadı.")
        return

    print(f"\n--- TEKİL TARAMA BAŞLATILIYOR | Referans Tarih: {scan_date.strftime('%d.%m.%Y')} ---")

    all_results = []
    genel_veri_cekme_baslangic_tarihi = scan_date - relativedelta(years=1, months=1, days=15)
    fon_args_list = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, scan_date, 30, 3, 5)
                      for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in fon_args_list}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon),
                              total=len(fon_args_list),
                              desc="🔎 Fonlar Taranıyor (Tekil)")
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e

        for future in progress_bar:
            fon_kodu_completed = future_to_fon[future]
            try:
                _, fund_history = future.result()
<<<<<<< HEAD
                if fund_history.empty: continue

                fiyat_son = get_price_on_or_before(fund_history, scan_date)
                if pd.isna(fiyat_son): continue

                degisimler = {}
                periods = {
                    'Günlük %': timedelta(days=1), 'Haftalık %': timedelta(weeks=1),
                    '2 Haftalık %': timedelta(weeks=2), 'Aylık %': relativedelta(months=1),
                    '3 Aylık %': relativedelta(months=3), '6 Aylık %': relativedelta(months=6),
                    '1 Yıllık %': relativedelta(years=1), 'YB %': relativedelta(years=scan_date.year, months=1, days=1)
                }
                for name, period_delta in periods.items():
                    target_date = scan_date - period_delta
                    if name == 'YB %': target_date = date(scan_date.year - 1, 12, 31)
                    
                    fiyat_onceki = get_price_on_or_before(fund_history, target_date)
                    degisimler[name] = calculate_change(fiyat_son, fiyat_onceki)
                
                fon_adi = all_fon_data_df.loc[all_fon_data_df['Fon Kodu'] == fon_kodu_completed, 'Fon Adı'].iloc[0]
                result_row = {'Fon Kodu': fon_kodu_completed, 'Fon Adı': fon_adi,
                              'Bitiş Tarihi': scan_date.strftime("%d.%m.%Y"), 'Fiyat': fiyat_son}
=======
                fiyat_son, degisimler = np.nan, {p: np.nan for p in ['Günlük %', 'Haftalık %', '2 Haftalık %',
                                                                      'Aylık %', '3 Aylık %', '6 Aylık %',
                                                                      '1 Yıllık %', 'YB %']}

                if fund_history is not None and not fund_history.empty:
                    fiyat_son = get_price_on_or_before(fund_history, scan_date)

                    if not pd.isna(fiyat_son):
                        fiyat_onceki_gun = get_first_price_looking_back(fund_history, scan_date, max_lookback_days=6)
                        degisimler['Günlük %'] = calculate_change(fiyat_son, fiyat_onceki_gun)

                        periods_other = {
                            'Haftalık %': timedelta(weeks=1),
                            '2 Haftalık %': timedelta(weeks=2),
                            'Aylık %': relativedelta(months=1),
                            '3 Aylık %': relativedelta(months=3),
                            '6 Aylık %': relativedelta(months=6),
                            '1 Yıllık %': relativedelta(years=1)
                        }

                        for name, period_delta in periods_other.items():
                            past_target_date = scan_date - period_delta
                            past_price = get_price_at_date_or_next_available(fund_history, past_target_date, max_lookforward_days=5)
                            degisimler[name] = calculate_change(fiyat_son, past_price)

                        target_yb_start_date = date(scan_date.year, 1, 1)
                        fiyat_yb_once = get_price_at_date_or_next_available(fund_history, target_yb_start_date, max_lookforward_days=5)
                        degisimler['YB %'] = calculate_change(fiyat_son, fiyat_yb_once)

                fon_adi = all_fon_data_df.loc[all_fon_data_df['Fon Kodu'] == fon_kodu_completed, 'Fon Adı'].iloc[0]
                if fon_kodu_completed not in all_fon_data_df['Fon Kodu'].values:
                    fon_adi = "Bilinmiyor"

                result_row = {
                    'Fon Kodu': fon_kodu_completed,
                    'Fon Adı': fon_adi,
                    'Bitiş Tarihi': scan_date.strftime("%d.%m.%Y"),
                    'Fiyat': fiyat_son
                }
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
                result_row.update(degisimler)
                all_results.append(result_row)
            except Exception as exc:
                print(f"Hata (Tekil - {fon_kodu_completed}): {exc}")
<<<<<<< HEAD

    print(f"\n✅ Tekil tarama tamamlandı. {len(all_results)} fon için sonuç hesaplandı.")
    print(f"Sonuçlar Google Sheets'teki '{WORKSHEET_NAME_MANUAL}' sayfasına yazılıyor...")

    results_df_tekil = pd.DataFrame(all_results)
    column_order = ['Fon Kodu', 'Fon Adı', 'Bitiş Tarihi', 'Fiyat', 'Günlük %', 'Haftalık %',
                   '2 Haftalık %', 'Aylık %', '3 Aylık %', '6 Aylık %', '1 Yıllık %', 'YB %']
=======
                traceback.print_exc()


    print(f"\n\n✅ Tekil tarama tamamlandı. {len(all_results)} fon için sonuç hesaplandı.")
    print(f"🔄 Sonuçlar Google Sheets'teki '{WORKSHEET_NAME_MANUAL}' sayfasına yazılıyor...")

    results_df_tekil = pd.DataFrame(all_results)
    column_order = ['Fon Kodu', 'Fon Adı', 'Bitiş Tarihi', 'Fiyat',
                    'Günlük %', 'Haftalık %', '2 Haftalık %', 'Aylık %',
                    '3 Aylık %', '6 Aylık %', '1 Yıllık %', 'YB %']
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
    existing_cols_tekil = [col for col in column_order if col in results_df_tekil.columns]

    if not results_df_tekil.empty:
        results_df_tekil = results_df_tekil[existing_cols_tekil].sort_values(by='YB %', ascending=False, na_position='last')
    else:
        results_df_tekil = pd.DataFrame(columns=existing_cols_tekil)

    for col in results_df_tekil.columns:
        if results_df_tekil[col].dtype == 'float64':
            results_df_tekil[col] = results_df_tekil[col].replace([np.inf, -np.inf], np.nan).astype(object).where(pd.notna(results_df_tekil[col]), None)
<<<<<<< HEAD

    try:
        spreadsheet = gc.open_by_key(SHEET_ID)
        worksheet_tekil = spreadsheet.worksheet(WORKSHEET_NAME_MANUAL)
=======
        elif results_df_tekil[col].dtype == 'object':
            results_df_tekil[col] = results_df_tekil[col].apply(lambda x: None if (isinstance(x, str) and (x.lower() in ['nan', 'nat'])) or pd.isna(x) else x)

    try:
        spreadsheet = gc.open_by_key(SHEET_ID)
        try:
            worksheet_tekil = spreadsheet.worksheet(WORKSHEET_NAME_MANUAL)
        except gspread.exceptions.WorksheetNotFound:
            worksheet_tekil = spreadsheet.add_worksheet(title=WORKSHEET_NAME_MANUAL, rows="1000", cols=max(100, len(existing_cols_tekil) + 5))
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
        worksheet_tekil.clear()

        if not results_df_tekil.empty:
            data_to_upload_tekil = [results_df_tekil.columns.values.tolist()] + results_df_tekil.values.tolist()
            worksheet_tekil.update(values=data_to_upload_tekil, range_name='A1')
<<<<<<< HEAD
            
            body_resize_tekil = {"requests": [{"autoResizeDimensions": {"dimensions": {"sheetId": worksheet_tekil.id, "dimension": "COLUMNS"}}}]}
            spreadsheet.batch_update(body_resize_tekil)
            print("✅ Google Sheets güncellendi ve sütunlar yeniden boyutlandırıldı.")
        
        return results_df_tekil # EKLENDİ: Sonuçları döndür

    except Exception as e:
        print(f"❌ Google Sheets'e yazma hatası (Tekil): {e}")
        return pd.DataFrame() # EKLENDİ: Hata durumunda boş DataFrame döndür

    print(f"--- Tekil Tarama Bitti. Toplam Süre: {time.time() - start_time_main:.2f} saniye ---")


# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    print("Otomatik Tarama Script'i Başlatıldı.")
    
    # 1. Kimlik Doğrulama
    gc_auth = google_sheets_auth()

    if gc_auth:
        # 2. Otomatik Değerleri Ayarla
        # Tekil tarama için bugünün tarihini kullan
        scan_date_input = datetime.now(TIMEZONE).date()
        
        # Haftalık tarama için 2 hafta geriye dönük çalıştır
        num_weeks_input = 2

        # 3. Taramaları Çalıştır
        print("\n" + "="*40)
        print("     ÖNCE TEKİL TARAMA ÇALIŞTIRILIYOR")
        print("="*40)
        results_df = run_single_date_scan_to_gsheets(scan_date_input, gc_auth)

        # --- YENİDEN DENEME KONTROLÜ ---
        if results_df is not None and not results_df.empty:
            empty_price_count = results_df['Fiyat'].isnull().sum()
            print(f"\nTarama sonrası kontrol: {empty_price_count} adet fonun fiyat verisi boş.")

            if empty_price_count >= 5:
                print(f"⚠️ 5 veya daha fazla fonun fiyat verisi boş. 20 dakika sonra yeniden denenecek...")
                time.sleep(1200) # 20 dakika bekle
                print("\n=== TEKİL TARAMA YENİDEN BAŞLIYOR (Otomatik Yeniden Deneme) ===")
                run_single_date_scan_to_gsheets(scan_date_input, gc_auth)
            else:
                print("✅ Fiyat verisi boş olan fon sayısı eşiğin altında. Yeniden denemeye gerek yok.")
        else:
            print("⚠️ İlk tarama sonucu boş veya hatalı. Yeniden deneme mekanizması atlanıyor.")
        # --- YENİDEN DENEME KONTROLÜ SONU ---

        print("\n" + "="*40)
        print("     SONRA HAFTALIK TARAMA ÇALIŞTIRILIYOR")
        print("="*40)
        run_weekly_scan_to_gsheets(num_weeks_input, gc_auth)
        
        print("\nTüm görevler tamamlandı.")
    else:
        print("\nKimlik doğrulama başarısız olduğu için işlemler durduruldu.")
=======

            body_resize_tekil = {
                "requests": [{
                    "autoResizeDimensions": {
                        "dimensions": {
                            "sheetId": worksheet_tekil.id,
                            "dimension": "COLUMNS",
                            "startIndex": 0,
                            "endIndex": len(existing_cols_tekil)
                        }
                    }
                }]
            }
            spreadsheet.batch_update(body_resize_tekil)
        else:
            print("ℹ️ Google Sheets'e yazılacak veri bulunmuyor (Tekil Tarama).")

        end_time_main_tekil = time.time()
        print("\n" + "="*50 +
              f"\n🎉 TEKİL TARAMA BAŞARIYLA TAMAMLANDI! ({datetime.now(TIMEZONE).strftime('%d.%m.%Y %H:%M:%S')})\n" +
              f"⏱️ Toplam süre: {((end_time_main_tekil - start_time_main) / 60):.2f} dakika\n" +
              "="*50)
    except Exception as e:
        print(f"❌ Google Sheets'e yazma sırasında hata (Tekil): {e}")
        traceback.print_exc()
        sys.exit(1) # Hata durumunda script'i sonlandır

# --- Ana Çalışma Bloğu (GitHub Actions için) ---
if __name__ == "__main__":
    print("\n--- GitHub Actions Otomatik Tarama Başlıyor ---")
    gc_auth = google_sheets_auth_github()
    if not gc_auth:
        print("❌ Google Sheets yetkilendirmesi başarısız olduğu için işlem iptal edildi.")
        sys.exit(1) # Kimlik doğrulama başarısız olursa çıkış yap

    # Otomatik tarama için bugünün tarihini al
    today_in_istanbul = datetime.now(TIMEZONE).date()
    print(f"Bugünün tarihi (İstanbul Saati): {today_in_istanbul.strftime('%d.%m.%Y')}")

    # Manuel Tekil Tarama Seçimi için (Girdi olmadan)
    # Otomatik tetikleyici için burayı çalıştıracağız, tarihi bugünün tarihi olacak
    # Siz "manuel tarama seçimi" dediniz ama aslında otomasyon içinde
    # `interactive_menu`'den 1 veya 4'ü seçmek yerine doğrudan fonksiyonu çağıracağız.
    print("\n=== TEKİL TARAMA BAŞLIYOR (Otomatik Tarih Seçimi ile) ===")
    run_scan_to_gsheets(today_in_istanbul, gc_auth)

    # Haftalık Tarama Seçimi için (2 hafta sabit)
    print("\n=== HAFTALIK TARAMA BAŞLIYOR (2 Hafta Sabit ile) ===")
    run_weekly_scan_to_gsheets(2, gc_auth)

    print("\n--- Tüm Otomatik Tarama İşlemleri Tamamlandı ---")
>>>>>>> 2bc599cd80568dc92249860efe9a963ffa9ca40e
