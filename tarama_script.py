# -*- coding: utf-8 -*-
# GÜNCELLENMİŞ FON TARAMA ARACI (GitHub Actions & CSV Uyumlu)

# --- Kütüphaneleri Import Etme ---
import pandas as pd
import numpy as np
import time
import gspread
import pytz
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from tefas import Crawler
from tqdm import tqdm
import concurrent.futures
import traceback
import os
import json
import sys

# --- Sabitler ---
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
F_COLS = ["date", "price"]
SHEET_ID = '1hSD4towyxKk9QHZFAcRlXy9NlLa_AyVrB9Jsy86ok14'
WORKSHEET_NAME_MANUAL = 'veriler'
WORKSHEET_NAME_WEEKLY = 'haftalık'
TIMEZONE = pytz.timezone('Europe/Istanbul')
# --- YENİ: OLUŞTURULACAK CSV DOSYASININ ADI ---
OUTPUT_CSV_FILENAME = "Fon_Verileri.csv"


# --- Google Sheets Kimlik Doğrulama Fonksiyonu (GitHub Actions Uyumlu) ---
def google_sheets_auth_github():
    print("\n Google Hizmet Hesabı ile kimlik doğrulaması yapılıyor...")
    try:
        gcp_service_account_key_json = os.getenv('GCP_SERVICE_ACCOUNT_KEY')
        if not gcp_service_account_key_json:
            print("❌ Hata: GCP_SERVICE_ACCOUNT_KEY ortam değişkeni ayarlanmamış.")
            sys.exit(1)
        credentials = json.loads(gcp_service_account_key_json)
        gc = gspread.service_account_from_dict(credentials)
        print("✅ Kimlik doğrulama başarılı.")
        return gc
    except Exception as e:
        print(f"❌ Kimlik doğrulama sırasında hata oluştu: {e}")
        traceback.print_exc()
        sys.exit(1)

# --- TEFAS Crawler Başlatma ---
try:
    tefas_crawler_global = Crawler()
    print("TEFAS Crawler başarıyla başlatıldı.")
except Exception as e:
    print(f"TEFAS Crawler başlatılırken hata: {e}")
    traceback.print_exc()
    tefas_crawler_global = None

# --- Yardımcı Fonksiyonlar (Değişiklik Yok) ---
def load_takasbank_fund_list():
    print(f" Takasbank'tan güncel fon listesi yükleniyor...")
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
        traceback.print_exc()
        return pd.DataFrame()

def get_price_on_or_before(df_fund_history, target_date: date):
    if df_fund_history is None or df_fund_history.empty or target_date is None: return np.nan
    df_filtered = df_fund_history[df_fund_history['date'] <= target_date].copy()
    if not df_filtered.empty:
        return df_filtered.sort_values(by='date', ascending=False)['price'].iloc[0]
    return np.nan

def get_price_at_date_or_next_available(df_fund_history, target_date: date, max_lookforward_days: int = 5):
    if df_fund_history is None or df_fund_history.empty or target_date is None: return np.nan
    future_limit_date = target_date + timedelta(days=max_lookforward_days)
    first_available = df_fund_history[(df_fund_history['date'] >= target_date) & (df_fund_history['date'] <= future_limit_date)].sort_values(by='date', ascending=True)
    if not first_available.empty: return first_available['price'].iloc[0]
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
            except Exception as e:
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

# --- TEKİL TARAMA FONKSİYONU ---
def run_scan_to_gsheets(scan_date: date, gc):
    start_time_main = time.time()
    all_fon_data_df = load_takasbank_fund_list()
    if all_fon_data_df.empty:
        print("❌ Taranacak fon listesi alınamadı. İşlem durduruldu.")
        return

    print(f"\n--- TEKİL TARAMA BAŞLATILIYOR | Tarih: {scan_date.strftime('%d.%m.%Y')} ---")
    genel_veri_cekme_baslangic_tarihi = scan_date - relativedelta(years=1, days=15)
    fon_args_list = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, scan_date, 90, 3, 5)
                      for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]

    MAX_WORKERS = 10
    results_list_tekil = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in fon_args_list}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon), total=len(fon_args_list), desc=" Fonlar Taranıyor (Tekil)")
        for future in progress_bar:
            fon_kodu_completed = future_to_fon[future]
            try:
                _, fund_history = future.result()
                fiyat_son = get_price_on_or_before(fund_history, scan_date)
                degisimler = {}
                if not pd.isna(fiyat_son):
                    periods_days = {'Günlük %': 1, 'Haftalık %': 7, '2 Haftalık %': 14}
                    for name, days in periods_days.items():
                        past_price = get_price_on_or_before(fund_history, scan_date - timedelta(days=days))
                        degisimler[name] = calculate_change(fiyat_son, past_price)
                    periods_other = {'Aylık %': relativedelta(months=1), '3 Aylık %': relativedelta(months=3),
                                     '6 Aylık %': relativedelta(months=6), '1 Yıllık %': relativedelta(years=1)}
                    for name, period_delta in periods_other.items():
                        past_target_date = scan_date - period_delta
                        past_price = get_price_at_date_or_next_available(fund_history, past_target_date, max_lookforward_days=5)
                        degisimler[name] = calculate_change(fiyat_son, past_price)
                    target_yb_start_date = date(scan_date.year, 1, 1)
                    fiyat_yb_once = get_price_at_date_or_next_available(fund_history, target_yb_start_date, max_lookforward_days=5)
                    degisimler['YB %'] = calculate_change(fiyat_son, fiyat_yb_once)
                fon_adi = all_fon_data_df.loc[all_fon_data_df['Fon Kodu'] == fon_kodu_completed, 'Fon Adı'].iloc[0]
                result_row = {'Fon Kodu': fon_kodu_completed, 'Fon Adı': fon_adi, 'Son Fiyat': fiyat_son, **degisimler}
                results_list_tekil.append(result_row)
            except Exception as exc:
                print(f"Hata (Tekil - {fon_kodu_completed}): {exc}")

    results_df_tekil = pd.DataFrame(results_list_tekil)
    column_order = ['Fon Kodu', 'Fon Adı', 'Son Fiyat', 'Günlük %', 'Haftalık %', '2 Haftalık %', 'Aylık %',
                    '3 Aylık %', '6 Aylık %', '1 Yıllık %', 'YB %']
    existing_cols_tekil = [col for col in column_order if col in results_df_tekil.columns]
    if not results_df_tekil.empty:
        results_df_tekil = results_df_tekil[existing_cols_tekil].sort_values(by='YB %', ascending=False, na_position='last')
    else:
        results_df_tekil = pd.DataFrame(columns=existing_cols_tekil)

    # --- YENİ EKLENEN ADIM: Verileri CSV dosyasına kaydet ---
    try:
        print(f"\n Sonuçlar '{OUTPUT_CSV_FILENAME}' dosyasına kaydediliyor...")
        # encoding='utf-8-sig' Türkçe karakterlerin Excel'de doğru görünmesini sağlar
        results_df_tekil.to_csv(OUTPUT_CSV_FILENAME, index=False, encoding='utf-8-sig')
        print(f"✅ Veriler başarıyla '{OUTPUT_CSV_FILENAME}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"❌ CSV dosyasına yazma hatası: {e}")
    # --- CSV KAYDETME ADIMI SONU ---

    try:
        print("\n Google Sheets'e veriler yazılıyor (Tekil Tarama)...")
        spreadsheet = gc.open_by_key(SHEET_ID)
        worksheet_tekil = spreadsheet.worksheet(WORKSHEET_NAME_MANUAL)
        worksheet_tekil.clear()
        df_to_gsheets_tekil = results_df_tekil.copy()
        for col in df_to_gsheets_tekil.columns:
            if df_to_gsheets_tekil[col].dtype == 'float64':
                df_to_gsheets_tekil[col] = df_to_gsheets_tekil[col].replace([np.inf, -np.inf], np.nan).astype(object).where(pd.notna(df_to_gsheets_tekil[col]), None)
        if not df_to_gsheets_tekil.empty:
            worksheet_tekil.update(values=[df_to_gsheets_tekil.columns.values.tolist()] + df_to_gsheets_tekil.values.tolist(), value_input_option='USER_ENTERED')
            body_resize_tekil = {"requests": [{"autoResizeDimensions": {"dimensions": {"sheetId": worksheet_tekil.id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": len(existing_cols_tekil)}}}]}
            spreadsheet.batch_update(body_resize_tekil)
        else:
            print("ℹ️ Google Sheets'e yazılacak veri bulunmuyor (Tekil Tarama).")
    except Exception as e:
        print(f"❌ Google Sheets'e yazma hatası (Tekil): {e}")

# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    print("--- Otomatik Tarama Scripti Başlatıldı ---")
    gc_auth = google_sheets_auth_github()
    today_in_istanbul = datetime.now(TIMEZONE).date()
    print("\n=== TEKİL TARAMA BAŞLIYOR (Otomatik Tarih Seçimi ile) ===")
    run_scan_to_gsheets(today_in_istanbul, gc_auth)
    # Haftalık tarama şimdilik devre dışı bırakıldı, istenirse açılabilir.
    # print("\n=== HAFTALIK TARAMA BAŞLIYOR (2 Hafta Sabit ile) ===")
    # run_weekly_scan_to_gsheets(2, gc_auth)
    print("\n--- Tüm Otomatik Tarama İşlemleri Tamamlandı ---")
