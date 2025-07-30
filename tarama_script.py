# -*- coding: utf-8 -*-
# OTOMATİK FON TARAMA ARACI (GitHub Actions için düzenlendi)

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

# --- Sabitler ve Yapılandırma ---
GSPREAD_CREDENTIALS_SECRET = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
F_COLS = ["date", "price"]
SHEET_ID = '1hSD4towyxKk9QHZFAcRlXy9NlLa_AyVrB9Jsy86ok14'
WORKSHEET_NAME_MANUAL = 'veriler'
WORKSHEET_NAME_WEEKLY = 'haftalık'
TIMEZONE = pytz.timezone('Europe/Istanbul')

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
            except Exception:
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
    fon_args_list = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, today, 90, 3, 5)
                      for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]

    MAX_WORKERS = 10
    weekly_results_dict = {}
    first_fund_calculated_columns = []
    first_fund_processed = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in fon_args_list}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon),
                              total=total_fon_count,
                              desc=" Fonlar Taranıyor (Haftalık)")

        for future in progress_bar:
            fon_kodu_completed = future_to_fon[future]
            try:
                _, fund_history = future.result()
                fon_adi = all_fon_data_df.loc[all_fon_data_df['Fon Kodu'] == fon_kodu_completed, 'Fon Adı'].iloc[0]
                
                current_fon_data = {'Fon Kodu': fon_kodu_completed, 'Fon Adı': fon_adi}
                calculated_cols_current_fund, weekly_changes_list = [], []
                first_week_end_price, last_week_start_price = np.nan, np.nan
                current_week_end_date_cal = today

                for i in range(num_weeks):
                    current_week_start_date_cal = current_week_end_date_cal - timedelta(days=7)
                    price_end = get_price_on_or_before(fund_history, current_week_end_date_cal)
                    price_start = get_price_on_or_before(fund_history, current_week_start_date_cal)

                    if i == 0: first_week_end_price = price_end
                    if i == num_weeks - 1: last_week_start_price = price_start

                    col_name = f"{current_week_end_date_cal.day:02d}-{current_week_start_date_cal.day:02d}/{current_week_end_date_cal.year % 100:02d}"
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
                
                # --- DEBUG SÜTUNLARI GERİ EKLENDİ ---
                current_fon_data['is_desired_trend'] = bool(is_desired_trend)
                current_fon_data['_DEBUG_WeeklyChanges_RAW'] = "'" + str([f"{x:.2f}" if not pd.isna(x) else "NaN" for x in weekly_changes_list])
                current_fon_data['_DEBUG_IsDesiredTrend'] = bool(is_desired_trend)
                weekly_results_dict[fon_kodu_completed] = current_fon_data
            except Exception as exc:
                print(f"Hata (Haftalık - {fon_kodu_completed}): {exc}")

    results_df = pd.DataFrame(list(weekly_results_dict.values()))

    if not first_fund_calculated_columns and not results_df.empty:
        temp_row_cols = [col for col in results_df.columns if col not in ['Fon Kodu', 'Fon Adı', 'Değerlendirme', 'is_desired_trend', '_DEBUG_WeeklyChanges_RAW', '_DEBUG_IsDesiredTrend']]
        first_fund_calculated_columns = temp_row_cols if temp_row_cols else []

    base_cols = ['Fon Kodu', 'Fon Adı']
    debug_cols = ['_DEBUG_WeeklyChanges_RAW', '_DEBUG_IsDesiredTrend']
    final_view_columns = base_cols + first_fund_calculated_columns + ['Değerlendirme'] + debug_cols
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
        traceback.print_exc()
    
    print(f"--- Haftalık Tarama Bitti. Toplam Süre: {time.time() - start_time_main:.2f} saniye ---")


# --- TEKİL TARİH TARAMA FONKSİYONU ---
def run_single_date_scan_to_gsheets(scan_date: date, gc):
    start_time_main = time.time()
    all_fon_data_df = load_takasbank_fund_list()

    if all_fon_data_df.empty:
        print("❌ Taranacak fon listesi alınamadı. İşlem durduruldu.")
        return pd.DataFrame()

    print(f"\n--- TEKİL TARAMA BAŞLATILIYOR | Bitiş Tarihi: {scan_date.strftime('%d.%m.%Y')} ---")

    total_fon_count = len(all_fon_data_df)
    genel_veri_cekme_baslangic_tarihi = scan_date - relativedelta(years=1, months=2)
    fon_args_list = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, scan_date, 90, 3, 5)
                    for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]

    MAX_WORKERS = 10
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in fon_args_list}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon),
                          total=total_fon_count,
                          desc=" Fonlar Taranıyor (Tekil)")

        for future in progress_bar:
            fon_kodu_completed = future_to_fon[future]
            try:
                _, fund_history = future.result()
                if fund_history.empty:
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
                        target_date = scan_date - period_delta
                        fiyat_onceki = get_price_on_or_before(fund_history, target_date)
                        degisimler[name] = calculate_change(fiyat_son, fiyat_onceki)
                    
                    try:
                        if scan_date and scan_date.year > 1:
                            yb_target_date = date(scan_date.year - 1, 12, 31)
                            fiyat_onceki_yb = get_price_on_or_before(fund_history, yb_target_date)
                            degisimler['YB %'] = calculate_change(fiyat_son, fiyat_onceki_yb)
                        else:
                            degisimler['YB %'] = np.nan
                    except Exception:
                        degisimler['YB %'] = np.nan
                
                fon_adi = all_fon_data_df.loc[all_fon_data_df['Fon Kodu'] == fon_kodu_completed, 'Fon Adı'].iloc[0]
                result_row = {'Fon Kodu': fon_kodu_completed, 'Fon Adı': fon_adi,
                              'Bitiş Tarihi': scan_date.strftime("%d.%m.%Y"), 'Fiyat': fiyat_son}
                result_row.update(degisimler)
                all_results.append(result_row)
            except Exception as exc:
                print(f"Hata (Ana Döngü - {fon_kodu_completed}): {exc}")
                traceback.print_exc()

    print(f"\n✅ Tekil tarama tamamlandı. {len(all_results)} fon için sonuç hesaplandı.")
    
    results_df_tekil = pd.DataFrame(all_results)
    column_order = ['Fon Kodu', 'Fon Adı', 'Bitiş Tarihi', 'Fiyat', 'Günlük %', 'Haftalık %',
                   '2 Haftalık %', 'Aylık %', '3 Aylık %', '6 Aylık %', '1 Yıllık %', 'YB %']
    
    existing_cols_tekil = [col for col in column_order if col in results_df_tekil.columns]
    
    if not results_df_tekil.empty:
        results_df_tekil = results_df_tekil[existing_cols_tekil]

    print(f"Sonuçlar Google Sheets'teki '{WORKSHEET_NAME_MANUAL}' sayfasına yazılıyor...")
    try:
        spreadsheet = gc.open_by_key(SHEET_ID)
        try:
            worksheet_tekil = spreadsheet.worksheet(WORKSHEET_NAME_MANUAL)
        except gspread.exceptions.WorksheetNotFound:
            print(f"ℹ️ '{WORKSHEET_NAME_MANUAL}' sayfası bulunamadı, yeni sayfa oluşturuluyor...")
            worksheet_tekil = spreadsheet.add_worksheet(title=WORKSHEET_NAME_MANUAL, rows="1000", cols=50)
        
        worksheet_tekil.clear()

        if not results_df_tekil.empty:
            if 'YB %' in results_df_tekil.columns:
                results_df_tekil.sort_values(by='YB %', ascending=False, na_position='last', inplace=True)
            
            for col in results_df_tekil.columns:
                if results_df_tekil[col].dtype == 'float64':
                    results_df_tekil[col] = results_df_tekil[col].replace([np.inf, -np.inf], np.nan).astype(object).where(pd.notna(results_df_tekil[col]), None)

            data_to_upload_tekil = [results_df_tekil.columns.values.tolist()] + results_df_tekil.values.tolist()
            worksheet_tekil.update(values=data_to_upload_tekil, range_name='A1')
            
            body_resize_tekil = {"requests": [{"autoResizeDimensions": {"dimensions": {"sheetId": worksheet_tekil.id, "dimension": "COLUMNS"}}}]}
            spreadsheet.batch_update(body_resize_tekil)
            print("✅ Google Sheets güncellendi ve sütunlar yeniden boyutlandırıldı.")
        else:
            print("ℹ️ Google Sheets'e yazılacak veri bulunamadı.")
        
        return results_df_tekil

    except Exception as e:
        print(f"❌ Google Sheets'e yazma hatası (Tekil): {e}")
        traceback.print_exc()
        return pd.DataFrame()

# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    print("Otomatik Tarama Script'i Başlatıldı.")
    
    gc_auth = google_sheets_auth()
    if not gc_auth:
        sys.exit(1)

    scan_date_input = datetime.now(TIMEZONE).date()
    num_weeks_input = 2 # Haftalık tarama için 2 hafta geriye dönük

    # --- TEKİL TARAMA ---
    print("\n" + "="*40)
    print("     ÖNCE TEKİL TARAMA ÇALIŞTIRILIYOR")
    print("="*40)
    results_df = run_single_date_scan_to_gsheets(scan_date_input, gc_auth)

    if results_df is not None and not results_df.empty:
        empty_price_count = results_df['Fiyat'].isnull().sum()
        print(f"\nTarama sonrası kontrol: {empty_price_count} adet fonun fiyat verisi boş.")

        if empty_price_count >= 5:
            print(f"⚠️ 5 veya daha fazla fonun fiyat verisi boş. 20 dakika sonra yeniden denenecek...")
            time.sleep(1200)
            print("\n=== TEKİL TARAMA YENİDEN BAŞLIYOR (Otomatik Yeniden Deneme) ===")
            run_single_date_scan_to_gsheets(scan_date_input, gc_auth)
        else:
            print("✅ Fiyat verisi boş olan fon sayısı eşiğin altında. Yeniden denemeye gerek yok.")
    else:
        print("⚠️ Tarama sonucu boş veya hatalı. Yeniden deneme mekanizması atlanıyor.")
    
    # --- HAFTALIK TARAMA ---
    print("\n" + "="*40)
    print("     SONRA HAFTALIK TARAMA ÇALIŞTIRILIYOR")
    print("="*40)
    run_weekly_scan_to_gsheets(num_weeks_input, gc_auth)

    print("\n--- Tüm işlemler tamamlandı ---")