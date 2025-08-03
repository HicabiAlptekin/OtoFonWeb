# Fonaliz - Otomatik Fon Analiz Arac
# Gerekli kütüphaneler: pandas, numpy, tefas-crawler, xlsxwriter

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tefas import Crawler
import time
import warnings
import concurrent.futures

# Uyarıları bastır
warnings.filterwarnings('ignore')

# --- AYARLAR ---
# Analiz için geriye dönük kaç ay bakılacağı
ANALIZ_SURESI_AY = 3
# Bir fonun "Hisse Senedi Yoğun" sayılması için gereken minimum hisse senedi oranı
HISSE_YOGUN_ORANI = 80.0
# Paralel veri çekme işleminde aynı anda çalışacak maksimum thread sayısı
MAX_WORKERS = 10

# --- YARDIMCI FONKSİYONLAR ---
def get_last_business_day():
    """Hafta sonu veya tatil günlerini atlayarak en son geçerli iş gününü bulur."""
    today = datetime.now()
    for i in range(7):
        day_to_check = today - timedelta(days=i)
        # Pazartesi=0, Cuma=4. Hafta içine denk geliyorsa o günü kullan.
        if day_to_check.weekday() < 5:
            return day_to_check
    return today - timedelta(days=1) # Fallback

# --- ANA FONKSİYONLAR ---
def get_dynamic_hisse_fon_listesi(crawler, tarih_gg_aa_yyyy, min_hisse_orani):
    """
    Belirli bir tarihteki tüm fonların varlık dağılımını 'get_breakdown' ile çeker
    ve Hisse Senedi Yoğun Fonları filtreleyerek döndürür.
    """
    print(f"\n{tarih_gg_aa_yyyy} tarihi için tüm fonların varlık dağılımları 'get_breakdown' ile çekiliyor...")
    try:
        df_breakdown = crawler.get_breakdown(date=tarih_gg_aa_yyyy)
        if df_breakdown.empty:
            print(f"UYARI: {tarih_gg_aa_yyyy} için varlık dağılım verisi çekilemedi.")
            return None
        
        print(f"BİLGİ: {len(df_breakdown)} adet fon için varlık dağılımı çekildi. Filtreleniyor...")
        
        if 'HS' in df_breakdown.columns:
            df_breakdown['HS'] = pd.to_numeric(df_breakdown['HS'], errors='coerce').fillna(0)
            df_hisse_fonlari = df_breakdown[df_breakdown['HS'] >= min_hisse_orani].copy()
            
            if df_hisse_fonlari.empty:
                print("UYARI: Belirtilen oranda Hisse Senedi Yoğun Fon bulunamadı.")
                return None
            
            print(f"BİLGİ: Varlık dağılımına göre {len(df_hisse_fonlari)} adet Hisse Senedi Yoğun Fon bulundu.")
            return df_hisse_fonlari[['code', 'title']]
        else:
            print("HATA: 'get_breakdown' verisinde 'HS' (Hisse Senedi) sütunu bulunamadı.")
            return None
    except Exception as e:
        print(f"KRİTİK HATA: Dinamik fon listesi oluşturulurken hata oluştu: {e}")
        return None

def get_fon_verileri_parallel(args):
    """Paralel işlem için tasarlanmış veri çekme fonksiyonu."""
    fon_kodu, start_date, end_date = args
    print(f"'{fon_kodu}' için veri çekiliyor...")
    try:
        # Her thread kendi crawler nesnesini oluşturur
        crawler = Crawler()
        df = crawler.fetch(start=start_date, end=end_date, name=fon_kodu, 
                           columns=["date", "price", "market_cap", "number_of_investors"])
        if df.empty:
            return fon_kodu, None
        df['date'] = pd.to_datetime(df['date'])
        return fon_kodu, df.sort_values(by='date').reset_index(drop=True)
    except Exception:
        return fon_kodu, None

def hesapla_metrikler(df_fon_fiyat):
    """Verilen fiyat geçmişi üzerinden performans metriklerini hesaplar."""
    if df_fon_fiyat is None or len(df_fon_fiyat) < 10:
        return None
    df_fon_fiyat['daily_return'] = df_fon_fiyat['price'].pct_change()
    df_fon_fiyat = df_fon_fiyat.dropna()
    if df_fon_fiyat.empty:
        return None
    
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

# --- ANA İŞ AKIŞI ---
def main():
    """Ana program akışı."""
    print(f"--- DİNAMİK FON ANALİZİ BAŞLATILDI ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    
    # Adım 1: Tarih aralığını dinamik olarak belirle
    end_date = get_last_business_day()
    start_date = end_date - pd.DateOffset(months=ANALIZ_SURESI_AY)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Analiz Tarih Aralığı: {start_date_str} -> {end_date_str}")
    
    # Adım 2: Hisse Senedi Yoğun Fonları dinamik olarak bul
    crawler = Crawler()
    end_date_for_breakdown = end_date.strftime('%d-%m-%Y')
    df_hisse_fonlari = get_dynamic_hisse_fon_listesi(crawler, end_date_for_breakdown, HISSE_YOGUN_ORANI)

    if df_hisse_fonlari is None or df_hisse_fonlari.empty:
        print("\n--- SONUÇ: Hisse Senedi Yoğun Fon bulunamadığı için analiz başlatılamadı. ---")
        return

    # Adım 3: Paralel olarak tüm fonların verilerini çek
    tasks = [(row['code'], start_date_str, end_date_str) for index, row in df_hisse_fonlari.iterrows()]
    fon_verileri = {}
    
    print(f"\n{len(tasks)} adet fon için veriler paralel olarak çekiliyor (Max Workers: {MAX_WORKERS})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(get_fon_verileri_parallel, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(future_to_fon):
            fon_kodu, data = future.result()
            if data is not None:
                fon_verileri[fon_kodu] = data
            else:
                print(f"UYARI: '{fon_kodu}' için veri alınamadı.")

    # Adım 4: Metrikleri hesapla
    analiz_sonuclari = []
    print("\nVeriler çekildi, metrikler hesaplanıyor...")
    for index, row in df_hisse_fonlari.iterrows():
        fon_kodu = row['code']
        fon_adi = row['title']
        
        if fon_kodu in fon_verileri:
            metrikler = hesapla_metrikler(fon_verileri[fon_kodu])
            if metrikler:
                sonuc = {'Fon Kodu': fon_kodu, 'Fon Adı': fon_adi, **metrikler}
                analiz_sonuclari.append(sonuc)
            else:
                print(f"UYARI: '{fon_kodu}' için metrikler hesaplanamadı (yetersiz veri).")

    # Adım 5: Sonuçları işle ve Excel'e yaz
    if analiz_sonuclari:
        df_sonuc = pd.DataFrame(analiz_sonuclari)
        sutun_sirasi = [
            'Fon Kodu', 'Fon Adı', 'Yatırımcı Sayısı', 'Piyasa Değeri (TL)',
            'Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)', 'Getiri (%)', 'Standart Sapma (Yıllık %)'
        ]
        df_sonuc = df_sonuc[sutun_sirasi]
        df_sonuc_sirali = df_sonuc.sort_values(by=['Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)'], ascending=[False, False])
        
        print("\n--- FON ANALİZ SONUÇLARI (Stratejiye Göre Sıralanmış) ---")
        print(df_sonuc_sirali.to_string())

        excel_dosya_adi = f"Hisse_Senedi_Fon_Analizi_{end_date_str}.xlsx"
        with pd.ExcelWriter(excel_dosya_adi, engine='xlsxwriter') as writer:
            df_sonuc_sirali.to_excel(writer, sheet_name='Fon Analizi', index=False)
            worksheet = writer.sheets['Fon Analizi']
            for i, col in enumerate(df_sonuc_sirali.columns):
                column_len = max(df_sonuc_sirali[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, column_len)

        print(f"\nAnaliz sonuçları '{excel_dosya_adi}' dosyasına kaydedildi ve hücreler otomatik boyutlandırıldı.")
    else:
        print("\n--- SONUÇ: Analiz edilecek yeterli veri bulunamadı. ---")

if __name__ == '__main__':
    main()
