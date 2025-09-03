# -*- coding: utf-8 -*-
# ENTEGRE FON TARAMA ARACI (OtoFon + Fonaliz - GitHub Actions için düzenlendi)
# Bu script, haftalık fon getirilerini tarar, belirli bir filtreden geçirir
# ve sonucu 'filtrelenmis_fonlar.txt' dosyasına yazar.

import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime, timedelta, date
from tefas import Crawler
import concurrent.futures
import warnings
import os

warnings.filterwarnings('ignore')

# --- Sabitler ve Yapılandırma ---
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
MAX_WORKERS = 10

# --- Yardımcı Fonksiyonlar ---
def load_takasbank_fund_list():
    print("Takasbank'tan güncel fon listesi yükleniyor...")
    try:
        df_excel = pd.read_excel(TAKASBANK_EXCEL_URL, engine='openpyxl')
        df_data = df_excel[['Fon Adı', 'Fon Kodu']].copy()
        df_data['Fon Kodu'] = df_data['Fon Kodu'].astype(str).str.strip().str.upper()
        df_data.dropna(subset=['Fon Kodu'], inplace=True)
        df_data = df_data[df_data['Fon Kodu'] != '']
        print(f"{len(df_data)} adet fon bilgisi okundu.")
        return df_data
    except Exception as e:
        print(f"Takasbank Excel yükleme hatası: {e}")
        return pd.DataFrame()

def get_price_on_or_before(df_fund_history, target_date: date):
    if df_fund_history is None or df_fund_history.empty or target_date is None: return np.nan
    df_filtered = df_fund_history[df_fund_history['date'] <= target_date].copy()
    if not df_filtered.empty: return df_filtered.sort_values(by='date', ascending=False)['price'].iloc[0]
    return np.nan

def calculate_change(current_price, past_price):
    if pd.isna(current_price) or pd.isna(past_price): return np.nan
    try:
        current_price_float, past_price_float = float(current_price), float(past_price)
        if past_price_float == 0: return np.nan
        return ((current_price_float - past_price_float) / past_price_float) * 100
    except (ValueError, TypeError): return np.nan

def fetch_data_for_fund_parallel(args):
    fon_kodu, start_date, end_date = args
    try:
        crawler = Crawler()
        df = crawler.fetch(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            name=fon_kodu,
            columns=["date", "price", "title"]
        )
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            return fon_kodu, df.sort_values(by='date').reset_index(drop=True)
    except Exception:
        return fon_kodu, None
    return fon_kodu, None

def run_weekly_scan(num_weeks: int):
    start_time_main = time.time()
    today = date.today()
    all_fon_data_df = load_takasbank_fund_list()

    if all_fon_data_df.empty:
        print("Taranacak fon listesi alınamadı. İşlem durduruldu.")
        return pd.DataFrame()

    print(f"\n{num_weeks} Haftalık Tarama Başlatılıyor...")
    
    genel_veri_cekme_baslangic_tarihi = today - timedelta(days=(num_weeks * 7) + 21)
    tasks = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, today) for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]
    
    weekly_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in tasks}
        
        for future in concurrent.futures.as_completed(future_to_fon):
            fon_kodu, fund_history = future.result()
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
                result = {'Fon Kodu': fon_kodu}
                for i, change in enumerate(weekly_changes):
                    result[f'Hafta_{i+1}_Getiri'] = change
                weekly_results.append(result)

    results_df = pd.DataFrame(weekly_results)
    print(f"Haftalık tarama tamamlandı. Toplam Süre: {time.time() - start_time_main:.2f} saniye")
    return results_df

# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    print("--- Tarama Script'i Başlatıldı ---")
    
    try:
        num_weeks_arg = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1].lower() == 'weekly' else 2
    except (ValueError, IndexError):
        num_weeks_arg = 2 # Varsayılan 2 hafta

    haftalik_sonuclar_df = run_weekly_scan(num_weeks=num_weeks_arg)

    if not haftalik_sonuclar_df.empty:
        print("\nFiltreleme uygulanıyor: Son 2 haftanın toplam getirisi >= %2")
        
        haftalik_sonuclar_df['Toplam_Getiri'] = haftalik_sonuclar_df['Hafta_1_Getiri'].fillna(0) + haftalik_sonuclar_df['Hafta_2_Getiri'].fillna(0)
        filtrelenmis_df = haftalik_sonuclar_df[haftalik_sonuclar_df['Toplam_Getiri'] >= 2].copy()
        
        if not filtrelenmis_df.empty:
            filtrelenmis_fon_listesi = filtrelenmis_df['Fon Kodu'].tolist()
            print(f"{len(filtrelenmis_fon_listesi)} fon Fonaliz için seçildi.")
            
            try:
                with open('filtrelenmis_fonlar.txt', 'w', encoding='utf-8') as f:
                    for fon_kodu in filtrelenmis_fon_listesi:
                        f.write(f"{fon_kodu}\n")
                print("'filtrelenmis_fonlar.txt' dosyasına yazıldı.")
            except Exception as e:
                print(f"Hata: Filtrelenmiş fon listesi dosyaya yazılırken bir sorun oluştu: {e}")
        else:
            print("Filtreyi geçen fon bulunamadı.")
    else:
        print("Haftalık tarama sonucu boş.")

    print("\n--- Tarama Script'i Tamamlandı ---")
