# -*- coding: utf-8 -*-
# Fonaliz - Dinamik Fon Analiz Aracı (GitHub Actions için düzenlendi)
# Bu script, 'filtrelenmis_fonlar.txt' dosyasındaki fonları okur,
# her biri için detaylı bir analiz yapar ve sonucu bir Excel dosyasına yazar.

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
FON_LISTESI_DOSYASI = 'filtrelenmis_fonlar.txt'
ANALIZ_SURESI_GUN = 90  # Analiz için geriye dönük gün sayısı
MAX_WORKERS = 10
RISKSIZ_FAIZ_ORANI = 0.45  # Yıllık risksiz faiz oranı (örneğin mevduat faizi)

# --- Ana Analiz Fonksiyonları ---
def get_valid_previous_workday(target_date: date) -> date:
    """Verilen tarihten önceki geçerli iş gününü bulur."""
    offset = 1
    while True:
        prev_day = target_date - timedelta(days=offset)
        if prev_day.weekday() < 5:  # Pazartesi-Cuma
            return prev_day
        offset += 1

def fetch_fund_data(fon_kodu, start_date, end_date):
    """Belirtilen fon için tarih aralığındaki verileri çeker."""
    try:
        crawler = Crawler()
        df = crawler.fetch(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            name=fon_kodu,
            columns=["date", "price", "market_cap", "number_of_investors"]
        )
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.date
            return fon_kodu, df.sort_values(by='date').reset_index(drop=True)
    except Exception as e:
        print(f"Hata: {fon_kodu} için veri çekilemedi - {e}")
        return fon_kodu, None
    return fon_kodu, None

def calculate_metrics(df_fund):
    """Fon verilerinden performans metriklerini hesaplar."""
    if df_fund is None or len(df_fund) < 2:
        return None

    df = df_fund.copy()
    df['daily_return'] = df['price'].pct_change()
    
    total_return = (df['price'].iloc[-1] / df['price'].iloc[0]) - 1
    annualized_std = df['daily_return'].std() * np.sqrt(252)
    
    # Negatif getirileri kullanarak Sortino için down_side_std hesapla
    down_side_return = df['daily_return'][df['daily_return'] < 0]
    down_side_std = down_side_return.std() * np.sqrt(252)
    
    # Sharpe Oranı
    annualized_return = (1 + df['daily_return'].mean())**252 - 1
    sharpe_ratio = (annualized_return - RISKSIZ_FAIZ_ORANI) / annualized_std if annualized_std != 0 else np.nan
    
    # Sortino Oranı
    sortino_ratio = (annualized_return - RISKSIZ_FAIZ_ORANI) / down_side_std if down_side_std != 0 else np.nan

    return {
        'Sortino Oranı (Yıllık)': sortino_ratio,
        'Sharpe Oranı (Yıllık)': sharpe_ratio,
        'Dönemsel Getiri (%)': total_return * 100,
        'Standart Sapma (Yıllık %)': annualized_std * 100,
        'Piyasa Değeri (TL)': df['market_cap'].iloc[-1],
        'Yatırımcı Sayısı': df['number_of_investors'].iloc[-1]
    }

def analyze_fund_parallel(fon_kodu, start_date, end_date):
    """Paralel işlem için bir fonu analiz eden sarmalayıcı fonksiyon."""
    _, df_fund = fetch_fund_data(fon_kodu, start_date, end_date)
    metrics = calculate_metrics(df_fund)
    if metrics:
        metrics['Fon Kodu'] = fon_kodu
        return metrics
    return None

# --- Dosya İşlemleri ---
def save_to_excel(df_results):
    """Sonuçları formatlanmış bir Excel dosyasına kaydeder."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"Hisse_Senedi_Fon_Analizi_{today_str}.xlsx"
    
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, sheet_name='Analiz Sonuçları', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Analiz Sonuçları']
        
        # Sütun genişliklerini ayarla
        for i, col in enumerate(df_results.columns):
            column_len = max(df_results[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_len)
            
        # Sayı formatlarını tanımla
        percent_format = workbook.add_format({'num_format': '#,##0.00"%"'})
        ratio_format = workbook.add_format({'num_format': '#,##0.00'})
        integer_format = workbook.add_format({'num_format': '#,##0'})
        
        # Formatları uygula
        worksheet.set_column('C:C', None, ratio_format) # Sharpe
        worksheet.set_column('B:B', None, ratio_format) # Sortino
        worksheet.set_column('D:E', None, percent_format) # Getiri, Std Sapma
        worksheet.set_column('F:G', None, integer_format) # Piyasa Değeri, Yatırımcı Sayısı

    print(f"\nAnaliz sonuçları başarıyla '{filename}' dosyasına kaydedildi.")

# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    print("--- Fonaliz Dinamik Analiz Script'i Başlatıldı ---")

    # Adım 1: Filtrelenmiş fon listesinin varlığını ve içeriğini kontrol et
    if not os.path.exists(FON_LISTESI_DOSYASI) or os.path.getsize(FON_LISTESI_DOSYASI) == 0:
        print(f"'{FON_LISTESI_DOSYASI}' dosyası bulunamadı veya boş.")
        print("Tarama script'i çalıştırılmamış veya filtreden geçen fon olmamış olabilir.")
        print("Analiz işlemi durduruldu.")
        sys.exit(0) # Başarılı bir şekilde çık, çünkü bu bir hata değil.

    with open(FON_LISTESI_DOSYASI, 'r', encoding='utf-8') as f:
        fon_kodlari = [line.strip() for line in f if line.strip()]
    
    print(f"{len(fon_kodlari)} adet fon analiz için yüklendi.")

    # Adım 2: Tarih aralığını belirle
    end_date = date.today()
    if end_date.weekday() >= 5: # Eğer bugün hafta sonu ise, son iş gününü al
        end_date = get_valid_previous_workday(end_date)
    start_date = end_date - timedelta(days=ANALIZ_SURESI_GUN)
    
    print(f"Analiz Tarih Aralığı: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")

    # Adım 3: Fonları paralel olarak analiz et
    start_time = time.time()
    all_metrics = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fund = {executor.submit(analyze_fund_parallel, fon, start_date, end_date): fon for fon in fon_kodlari}
        
        for future in concurrent.futures.as_completed(future_to_fund):
            result = future.result()
            if result:
                all_metrics.append(result)

    print(f"Veri çekme ve analiz {time.time() - start_time:.2f} saniyede tamamlandı.")

    # Adım 4: Sonuçları işle ve Excel'e kaydet
    if not all_metrics:
        print("Analiz edilecek geçerli veri bulunamadı.")
    else:
        results_df = pd.DataFrame(all_metrics)
        # Sütun sırasını düzenle
        ordered_columns = ['Fon Kodu', 'Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)', 
                           'Dönemsel Getiri (%)', 'Standart Sapma (Yıllık %)', 
                           'Piyasa Değeri (TL)', 'Yatırımcı Sayısı']
        results_df = results_df[ordered_columns]
        
        # Sortino oranına göre sırala
        results_df = results_df.sort_values(by='Sortino Oranı (Yıllık)', ascending=False)
        
        save_to_excel(results_df)

    print("\n--- Analiz Script'i Tamamlandı ---")