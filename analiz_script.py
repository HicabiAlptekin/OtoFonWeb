# Fonaliz - Otomatik Fon Analiz Arac
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tefas import Crawler
import time
import warnings
import concurrent.futures
import sys

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# --- AYARLAR ---
ANALIZ_SURESI_AY = 3
MAX_WORKERS = 10
MANUAL_FON_KODLARI = [
    "DKR", "ESP", "AOJ", "SBH", "PPH", "KOT", "FD1", "PP1", "KHC", "PBN", 
    "AEV", "YAS", "OTM", "DTL", "BID", "BDY", "GNH", "HKH", "KHA", "HFI", 
    "KPA", "BV1", "GNS", "DXP", "YHB", "BIH", "OHK", "MTK", "HJB", "KPH", 
    "BFT", "AES", "TZD", "FCK", "GOH", "IHT", "POS", "RDF", "KUA", "FRC", 
    "GKF", "FDG", "MD2", "TI3", "MD1", "ICH", "MTH", "NNF", "HMS", "ICF", 
    "RPD", "RTP", "GO3", "FBC", "GO4", "JUP", "IAE", "BUL", "IFN", "TPV", 
    "BTZ", "THD", "TGA", "HNC", "YLY", "IHK", "BVM", "GO1", "HGM", "ZJL", 
    "TKF", "TI2", "HMC", "BIO", "YFV", "GPF", "ACD", "RIK", "HMG", "HVK", 
    "PGS", "KHT", "HKG", "MGB", "PGD", "KLH", "RTH", "YPV", "EKF", "KTN", 
    "UNT", "MPK", "IV8", "RKS", "MPF", "IAT", "DBK", "OPD", "RKH", "NJY", 
    "DBZ", "YCK", "PPM", "KSV", "KLU", "AC5", "RBV", "NSH", "MUT", "VMV", 
    "DID", "DDA", "TPF", "BHI", "OTK", "HDK", "KIA", "DPK", "HIM", "SHE", 
    "MCU", "IML", "ICS", "KIH", "DKL", "HML", "MAD", "YZK", "CKF", "NKA", 
    "TMM", "IDH", "RD1", "KMF", "OJK", "NJF", "PAF", "MKG", "HBF", "NAU", 
    "OGD", "YNK", "GOL", "PKF", "KZU", "TTA", "RPG", "TCA", "DBA", "AFO", 
    "YKT", "GGK", "ONE"
]

def get_last_business_day():
    today = datetime.now()
    for i in range(7):
        day_to_check = today - timedelta(days=i)
        if day_to_check.weekday() < 5:
            return day_to_check
    return today - timedelta(days=1)

def get_fon_verileri_parallel(args):
    fon_kodu, start_date, end_date = args
    print(f"'{fon_kodu}' için veri çekiliyor...")
    try:
        crawler = Crawler()
        df = crawler.fetch(start=start_date, end=end_date, name=fon_kodu, 
                           columns=["date", "price", "market_cap", "number_of_investors", "title"])
        if df.empty:
            return fon_kodu, None, None
        df['date'] = pd.to_datetime(df['date'])
        fon_adi = df['title'].iloc[0] if not df.empty else fon_kodu
        return fon_kodu, fon_adi, df.sort_values(by='date').reset_index(drop=True)
    except Exception as e:
        print(f"HATA: '{fon_kodu}' verisi çekilirken bir sorun oluştu: {e}")
        return fon_kodu, None, None

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

def analyze_daily_correlations(df_fon_data, fon_kodu):
    if df_fon_data is None or df_fon_data.empty:
        return None, 0 # Sentiment puanı da döndür

    df = df_fon_data.copy()
    df['investor_change'] = df['number_of_investors'].diff()
    df['market_cap_pct_change'] = df['market_cap'].pct_change()
    df['price_pct_change'] = df['price'].pct_change()

    # Eşik Değerler (yüzde olarak, örn: 0.005 = %0.5)
    market_cap_change_threshold = 0.005
    price_change_threshold = 0.005

    # Ağırlıklar (toplamı 1.0 olmalı)
    weights = {
        'same_day_price': 0.4,
        'same_day_market_cap': 0.3,
        'next_day_price': 0.2,
        'next_day_market_cap': 0.1
    }

    # Shift for 1-day lag analysis using pre-calculated percentage changes
    df['market_cap_pct_change_next_day'] = df['market_cap_pct_change'].shift(-1)
    df['price_pct_change_next_day'] = df['price_pct_change'].shift(-1)

    # Scenario 1: Investor count increases, market cap increases on the same day (eşik değeri ile)
    same_day_positive_correlation = df[(df['investor_change'] > 0) & (df['market_cap_pct_change'] >= market_cap_change_threshold)].shape[0]
    same_day_positive_price_correlation = df[(df['investor_change'] > 0) & (df['price_pct_change'] >= price_change_threshold)].shape[0]

    # Scenario 2: Investor count increases, market cap increases on the next day (eşik değeri ile)
    next_day_positive_correlation = df[(df['investor_change'] > 0) & (df['market_cap_pct_change_next_day'] >= market_cap_change_threshold)].shape[0]
    next_day_positive_price_correlation = df[(df['investor_change'] > 0) & (df['price_pct_change_next_day'] >= price_change_threshold)].shape[0]

    total_investor_increases = df[df['investor_change'] > 0].shape[0]

    print(f"--- {fon_kodu} için Günlük Korelasyon Analizi ---")
    print(f"Toplam yatırımcı artışı olan gün sayısı: {total_investor_increases}")
    print(f"Yatırımcı artışı ve aynı gün piyasa değeri artışı: {same_day_positive_correlation} gün")
    print(f"Yatırımcı artışı ve aynı gün fiyat artışı: {same_day_positive_price_correlation} gün")
    print(f"Yatırımcı artışı ve ertesi gün piyasa değeri artışı: {next_day_positive_correlation} gün")
    print(f"Yatırımcı artışı ve ertesi gün fiyat artışı: {next_day_positive_price_correlation} gün")
    print("--------------------------------------------------")

    # Sentiment Puanı Hesaplama (Ağırlıklı)
    sentiment_score = 0
    if total_investor_increases > 0:
        weighted_score = (
            (same_day_positive_price_correlation * weights['same_day_price']) +
            (same_day_positive_correlation * weights['same_day_market_cap']) +
            (next_day_positive_price_correlation * weights['next_day_price']) +
            (next_day_positive_correlation * weights['next_day_market_cap'])
        )
        # Maksimum olası ağırlıklı puan (her yatırımcı artışı için tüm senaryoların ağırlıklarının toplamı)
        max_possible_weighted_score = total_investor_increases * sum(weights.values())
        
        if max_possible_weighted_score > 0:
            sentiment_score = (weighted_score / max_possible_weighted_score) * 100
        else:
            sentiment_score = 0

    print(f"--- {fon_kodu} için Günlük Korelasyon Analizi ---")
    print(f"Toplam yatırımcı artışı olan gün sayısı: {total_investor_increases}")
    print(f"Yatırımcı artışı ve aynı gün piyasa değeri artışı: {same_day_positive_correlation} gün")
    print(f"Yatırımcı artışı ve aynı gün fiyat artışı: {same_day_positive_price_correlation} gün")
    print(f"Yatırımcı artışı ve ertesi gün piyasa değeri artışı: {next_day_positive_correlation} gün")
    print(f"Yatırımcı artışı ve ertesi gün fiyat artışı: {next_day_positive_price_correlation} gün")
    print(f"HESAPLANAN SENTIMENT PUANI: {round(sentiment_score, 2)}")
    print("--------------------------------------------------")

    return {
        'total_investor_increases': total_investor_increases,
        'same_day_market_cap_increase': same_day_positive_correlation,
        'same_day_price_increase': same_day_positive_price_correlation,
        'next_day_market_cap_increase': next_day_positive_correlation,
        'next_day_price_increase': next_day_positive_price_correlation
    }, round(sentiment_score, 2)

def calistir_analiz():
    """Fon analizini çalıştırır ve sonuçları bir DataFrame olarak döndürür."""
    print(f"--- MANUEL FON LİSTESİ İLE ANALİZ BAŞLATILDI ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    end_date = get_last_business_day()
    start_date = end_date - pd.DateOffset(months=ANALIZ_SURESI_AY)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    print(f"Analiz Tarih Aralığı: {start_date_str} -> {end_date_str}")

    tasks = [(fon_kodu, start_date_str, end_date_str) for fon_kodu in MANUAL_FON_KODLARI]
    analiz_sonuclari = []
    
    print(f"\n{len(tasks)} adet fon için veriler paralel olarak çekiliyor...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(get_fon_verileri_parallel, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(future_to_fon):
            fon_kodu, fon_adi, data = future.result()
            if data is not None:
                correlation_results, sentiment_score = analyze_daily_correlations(data, fon_kodu)
                print(f"DEBUG: Fon '{fon_kodu}' için Sentiment Puanı: {sentiment_score}")

                metrikler = hesapla_metrikler(data)
                if metrikler:
                    sonuc = {'Fon Kodu': fon_kodu, 'Fon Adı': fon_adi, **metrikler, 'Sentiment Puanı': sentiment_score}
                    analiz_sonuclari.append(sonuc)
                else:
                    print(f"UYARI: '{fon_kodu}' için metrikler hesaplanamadı.")
            else:
                print(f"UYARI: '{fon_kodu}' için veri alınamadı.")

    if not analiz_sonuclari:
        print("\n--- SONUÇ: Analiz edilecek yeterli veri bulunamadı. ---")
        return None, None

    df_sonuc = pd.DataFrame(analiz_sonuclari)
    
    print("\n--- DEBUG: DataFrame oluşturuldu, ilk 5 satır ---")
    print(df_sonuc.head().to_string())
    print("--------------------------------------------------\n")

    # 'Sentiment Puanı' sütununun varlığını kontrol et
    if 'Sentiment Puanı' not in df_sonuc.columns:
        print("HATA: 'Sentiment Puanı' sütunu DataFrame'de bulunamadı!")
        df_sonuc['Sentiment Puanı'] = 0 # Hata durumunda sütunu varsayılan değerle ekle
    
    sutun_sirasi = [
        'Fon Kodu', 'Fon Adı', 'Yatırımcı Sayısı', 'Piyasa Değeri (TL)',
        'Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)', 'Getiri (%)', 'Standart Sapma (Yıllık %)',
        'Sentiment Puanı'
    ]
    
    df_sonuc = df_sonuc[sutun_sirasi]
    df_sonuc_sirali = df_sonuc.sort_values(by=['Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)'], ascending=[False, False])
    
    print("\n--- FON ANALİZ SONUÇLARI ---")
    print(df_sonuc_sirali.to_string())
    
    return df_sonuc_sirali, end_date_str

def main():
    """Script doğrudan çalıştırıldığında analiz yapar ve dosyayı kaydeder."""
    df_sonuc_sirali, end_date_str = calistir_analiz()
    
    if df_sonuc_sirali is not None:
        # Kullanıcının İndirilenler klasörünün yolunu bul
        from pathlib import Path
        downloads_path = Path.home() / "Downloads"
        # Klasörün mevcut olduğundan emin ol
        downloads_path.mkdir(parents=True, exist_ok=True)
        excel_dosya_adi = f"Hisse_Senedi_Fon_Analizi_{end_date_str}.xlsx"
        full_excel_path = downloads_path / excel_dosya_adi

        with pd.ExcelWriter(full_excel_path, engine='xlsxwriter') as writer:
            df_sonuc_sirali.to_excel(writer, sheet_name='Fon Analizi', index=False)
            worksheet = writer.sheets['Fon Analizi']
            for i, col in enumerate(df_sonuc_sirali.columns):
                column_len = max(df_sonuc_sirali[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, column_len)
        print(f"\nAnaliz sonuçları '{full_excel_path}' dosyasına kaydedildi.")


if __name__ == '__main__':
    main()