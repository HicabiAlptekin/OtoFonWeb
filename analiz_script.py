# -*- coding: utf-8 -*-
# Fonaliz - Dinamik Fon Analiz Aracı
# Bu script, 'filtrelenmis_fonlar.txt' dosyasında listelenen fonlar için
# detaylı risk ve getiri analizi yapar.

import pandas as pd
import numpy as np
from datetime import datetime, date
import time
import warnings
import concurrent.futures
import sys
import os
from tefas import Crawler
from dateutil.relativedelta import relativedelta

# Uyarıları kapat
warnings.filterwarnings('ignore')

# --- AYARLAR ---
ANALIZ_SURESI_AY = 3
MAX_WORKERS = 10
INPUT_FILE = "filtrelenmis_fonlar.txt"

# --- Yardımcı Fonksiyonlar ---
def load_filtered_fund_list():
    """
    'filtrelenmis_fonlar.txt' dosyasından analiz edilecek fon kodlarının
    listesini okur.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"HATA: Analiz için fon listesini içeren '{INPUT_FILE}' dosyası bulunamadı.")
        print("Lütfen önce tarama script'ini çalıştırarak bu dosyayı oluşturun.")
        sys.exit(1)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        fon_kodlari = [line.strip() for line in f if line.strip()]
        
    if not fon_kodlari:
        print(f"UYARI: '{INPUT_FILE}' dosyası boş. Analiz edilecek fon bulunamadı.")
        sys.exit(0)
        
    print(f"'{INPUT_FILE}' dosyasından {len(fon_kodlari)} adet fon kodu okundu.")
    return fon_kodlari

def fetch_data_for_fund_parallel(args):
    """
    Verilen bir fon kodu için TEFAS'tan paralel olarak veri çeker.
    """
    fon_kodu, start_date, end_date = args
    try:
        crawler = Crawler()
        df = crawler.fetch(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            name=fon_kodu,
            columns=["date", "price", "market_cap", "number_of_investors", "title"]
        )
        if df.empty:
            return fon_kodu, None, None
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        fon_adi = df['title'].iloc[0] if not df.empty and 'title' in df.columns else fon_kodu
        return fon_kodu, fon_adi, df.sort_values(by='date').reset_index(drop=True)
    except Exception as e:
        print(f"HATA ({fon_kodu}): Veri çekilirken sorun oluştu - {e}")
        return fon_kodu, None, None

def hesapla_metrikler(df_fon_fiyat):
    """
    Bir fonun geçmiş fiyat verilerini kullanarak risk/getiri metriklerini hesaplar.
    """
    if df_fon_fiyat is None or len(df_fon_fiyat) < 10: return None
    df_fon_fiyat['daily_return'] = df_fon_fiyat['price'].pct_change()
    df_fon_fiyat = df_fon_fiyat.dropna()
    if df_fon_fiyat.empty: return None

    getiri = (df_fon_fiyat['price'].iloc[-1] / df_fon_fiyat['price'].iloc[0]) - 1
    volatilite = df_fon_fiyat['daily_return'].std() * np.sqrt(252)
    ortalama_gunluk_getiri = df_fon_fiyat['daily_return'].mean()
    sharpe_orani = (ortalama_gunluk_getiri / df_fon_fiyat['daily_return'].std()) * np.sqrt(252) if df_fon_fiyat['daily_return'].std() != 0 else 0
    
    negatif_getiriler = df_fon_fiyat[df_fon_fiyat['daily_return'] < 0]['daily_return']
    downside_deviation = negatif_getiriler.std() * np.sqrt(252) if not negatif_getiriler.empty else 0
    sortino_orani = (ortalama_gunluk_getiri * 252) / downside_deviation if downside_deviation != 0 else 0
        
    return {
        'Getiri (%)': round(getiri * 100, 2),
        'Standart Sapma (Yıllık %)': round(volatilite * 100, 2),
        'Sharpe Oranı (Yıllık)': round(sharpe_orani, 2),
        'Sortino Oranı (Yıllık)': round(sortino_orani, 2),
        'Piyasa Değeri (TL)': df_fon_fiyat['market_cap'].iloc[-1],
        'Yatırımcı Sayısı': df_fon_fiyat['number_of_investors'].iloc[-1]
    }

def main():
    """
    Ana fonksiyon: fon listesini okur, verileri çeker, analiz eder ve sonucu Excel'e yazar.
    """
    print("--- Fonaliz Dinamik Analiz Script'i Başlatıldı ---")
    start_time = time.time()
    
    fon_listesi = load_filtered_fund_list()
    
    end_date = date.today()
    start_date = end_date - relativedelta(months=ANALIZ_SURESI_AY)
    
    tasks = [(fon_kodu, start_date, end_date) for fon_kodu in fon_listesi]
    analiz_sonuclari = []

    print(f"\n{len(fon_listesi)} adet fon için {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')} tarih aralığında analiz başlatılıyor...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_fon):
            fon_kodu, fon_adi, data = future.result()
            if data is not None:
                metrikler = hesapla_metrikler(data)
                if metrikler:
                    sonuc = {'Fon Kodu': fon_kodu, 'Fon Adı': fon_adi, **metrikler}
                    analiz_sonuclari.append(sonuc)

    if not analiz_sonuclari:
        print("\n--- SONUÇ: Analiz edilecek yeterli veri bulunamadı. ---")
        return

    df_sonuc = pd.DataFrame(analiz_sonuclari)
    sutun_sirasi = ['Fon Kodu', 'Fon Adı', 'Yatırımcı Sayısı', 'Piyasa Değeri (TL)', 'Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)', 'Getiri (%)', 'Standart Sapma (Yıllık %)']
    df_sonuc = df_sonuc[sutun_sirasi]
    df_sonuc_sirali = df_sonuc.sort_values(by=['Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)'], ascending=[False, False])

    excel_dosya_adi = f"Fonaliz_Sonuclari_{end_date.strftime('%Y-%m-%d')}.xlsx"
    print(f"\nAnaliz tamamlandı. Sonuçlar '{excel_dosya_adi}' dosyasına yazılıyor...")
    
    try:
        with pd.ExcelWriter(excel_dosya_adi, engine='xlsxwriter') as writer:
            df_sonuc_sirali.to_excel(writer, sheet_name='Analiz Sonuclari', index=False)
            worksheet = writer.sheets['Analiz Sonuclari']
            for i, col in enumerate(df_sonuc_sirali.columns):
                column_len = max(df_sonuc_sirali[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, column_len)
        print(f"'{excel_dosya_adi}' dosyası başarıyla oluşturuldu.")
    except Exception as e:
        print(f"HATA: Excel dosyası oluşturulurken bir sorun oluştu: {e}")

    end_time = time.time()
    print(f"\n--- Tüm işlemler {end_time - start_time:.2f} saniyede tamamlandı ---")

if __name__ == "__main__":
    main()
