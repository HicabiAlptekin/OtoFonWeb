# Fonaliz - Otomatik Fon Analiz Araci
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tefas import Crawler
import time
import warnings
import concurrent.futures
import sys
import random

# Deprecation uyarılarını kapat
warnings.filterwarnings('ignore')
# Konsol çıktısının Türkçe karakterleri doğru göstermesini sağlar
sys.stdout.reconfigure(encoding='utf-8')

# --- AYARLAR ---
ANALIZ_SURESI_AY = 3
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
    """Son iş gününü hesaplar."""
    today = datetime.now()
    for i in range(7):
        day_to_check = today - timedelta(days=i)
        if day_to_check.weekday() < 5:
            return day_to_check
    return today - timedelta(days=1)

def get_fon_verileri_parallel(args):
    """Belirtilen fon için TEFAS'tan verileri çeker ve olası hataları tekrar dener."""
    fon_kodu, start_date, end_date, bekleme_suresi = args
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Sunucu kısıtlamalarını aşmak için rastgele bekleme süresi
            if bekleme_suresi > 0:
                time.sleep(random.uniform(bekleme_suresi, bekleme_suresi + 1.5))
            
            crawler = Crawler()
            df = crawler.fetch(start=start_date, end=end_date, name=fon_kodu,
                                columns=["date", "price", "market_cap", "number_of_investors", "title"])
            
            if df.empty:
                return fon_kodu, None, "Veri bulunamadı"
            
            df['date'] = pd.to_datetime(df['date'])
            fon_adi = df['title'].iloc[0] if not df.empty else fon_kodu
            return fon_kodu, fon_adi, df.sort_values(by='date').reset_index(drop=True)
        except Exception as e:
            error_message = str(e)
            print(f"⚠️ Uyarı: '{fon_kodu}' için hata oluştu: {error_message}")
            if "rate limiting" in error_message or "robot check" in error_message or "timeout" in error_message:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10 + random.uniform(0, 5)
                    print(f"🔄 Tekrar deneme... {wait_time:.2f} saniye bekleniyor... (Deneme {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    return fon_kodu, None, f"Veri çekme başarısız. Son deneme de rate-limiting/timeout hatası verdi. Hata: {error_message}"
            else:
                return fon_kodu, None, f"Beklenmedik hata: {error_message}"
    
    return fon_kodu, None, "Tüm denemeler başarısız oldu."


def hesapla_metrikler(df_fon_fiyat):
    """Fon için finansal metrikleri (Sharpe, Sortino vb.) hesaplar."""
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

def analyze_daily_correlations(df_fon_data):
    """Yatırımcı sayısı değişimi ile fonun performansı arasındaki korelasyonu inceler."""
    if df_fon_data is None or df_fon_data.empty:
        return None, 0
    df = df_fon_data.copy()
    df['investor_change'] = df['number_of_investors'].diff()
    df['market_cap_pct_change'] = df['market_cap'].pct_change()
    df['price_pct_change'] = df['price'].pct_change()
    market_cap_change_threshold = 0.005
    price_change_threshold = 0.005
    weights = {'same_day_price': 0.4, 'same_day_market_cap': 0.3, 'next_day_price': 0.2, 'next_day_market_cap': 0.1}
    df['market_cap_pct_change_next_day'] = df['market_cap_pct_change'].shift(-1)
    df['price_pct_change_next_day'] = df['price_pct_change'].shift(-1)
    same_day_positive_correlation = df[(df['investor_change'] > 0) & (df['market_cap_pct_change'] >= market_cap_change_threshold)].shape[0]
    same_day_positive_price_correlation = df[(df['investor_change'] > 0) & (df['price_pct_change'] >= price_change_threshold)].shape[0]
    next_day_positive_correlation = df[(df['investor_change'] > 0) & (df['market_cap_pct_change_next_day'] >= market_cap_change_threshold)].shape[0]
    next_day_positive_price_correlation = df[(df['investor_change'] > 0) & (df['price_pct_change_next_day'] >= price_change_threshold)].shape[0]
    total_investor_increases = df[df['investor_change'] > 0].shape[0]
    sentiment_score = 0
    if total_investor_increases > 0:
        weighted_score = ((same_day_positive_price_correlation * weights['same_day_price']) + (same_day_positive_correlation * weights['same_day_market_cap']) + (next_day_positive_price_correlation * weights['next_day_price']) + (next_day_positive_correlation * weights['next_day_market_cap']))
        max_possible_weighted_score = total_investor_increases * sum(weights.values())
        if max_possible_weighted_score > 0:
            sentiment_score = (weighted_score / max_possible_weighted_score) * 100
    return {'total_investor_increases': total_investor_increases, 'same_day_market_cap_increase': same_day_positive_correlation, 'same_day_price_increase': same_day_price_correlation, 'next_day_market_cap_increase': next_day_positive_correlation, 'next_day_price_increase': next_day_positive_price_correlation}, round(sentiment_score, 2)

def tarama_asamasini_calistir(fon_kodlari, max_workers, bekleme_suresi, start_date_str, end_date_str, asama_adi):
    """Belirtilen fon listesini, verilen parametrelerle tarar ve detaylı çıktı verir."""
    analiz_sonuclari = []
    basarisiz_fonlar = {}
    tasks = [(fon_kodu, start_date_str, end_date_str, bekleme_suresi) for fon_kodu in fon_kodlari]

    print(f"\n--- {asama_adi} ({len(tasks)} adet fon) başlatılıyor ---")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_fon = {executor.submit(get_fon_verileri_parallel, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_fon):
            fon_kodu, fon_adi, data_or_error = future.result()
            if isinstance(data_or_error, pd.DataFrame):
                data = data_or_error
                # Veri çekme başarılıysa analiz yap
                _, sentiment_score = analyze_daily_correlations(data)
                metrikler = hesapla_metrikler(data)
                
                if metrikler:
                    sonuc = {'Fon Kodu': fon_kodu, 'Fon Adı': fon_adi, **metrikler, 'Sentiment Puanı': sentiment_score}
                    analiz_sonuclari.append(sonuc)
                    print(f"✅ Başarılı: '{fon_kodu}' için veri çekildi ve analiz edildi.")
                else:
                    basarisiz_fonlar[fon_kodu] = "Metrik hesaplanamadı (yetersiz veri)."
                    print(f"⚠️ Uyarı: '{fon_kodu}' için veri çekildi ancak metrik hesaplamaya yetmedi.")
            else:
                basarisiz_fonlar[fon_kodu] = data_or_error
                print(f"❌ Hata: '{fon_kodu}' verisi çekilemedi. Neden: {data_or_error}")
                
    print(f"\n--- {asama_adi} tamamlandı ---")
    print(f"Toplam {len(tasks)} fondan {len(analiz_sonuclari)} tanesi başarıyla işlendi.")
    print(f"{len(basarisiz_fonlar)} fon başarısız oldu veya atlandı.")

    return analiz_sonuclari, basarisiz_fonlar

def main():
    """Üç aşamalı tarama mantığını yönetir ve sonuçları dosyalar."""
    end_date = get_last_business_day()
    start_date = end_date - pd.DateOffset(months=ANALIZ_SURESI_AY)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"--- ANALİZ BAŞLATILDI ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    print(f"Analiz Tarih Aralığı: {start_date_str} -> {end_date_str}")
    print(f"Toplam analize alınacak fon sayısı: {len(MANUAL_FON_KODLARI)}")

    # Aşamaların Parametreleri
    # Kullanıcının isteği üzerine 1. Aşamaya 0.05 saniyelik bir bekleme süresi eklendi.
    asama_parametreleri = [
        {'asama_adi': '1. AŞAMA: Hızlı Tarama', 'max_workers': 10, 'bekleme_suresi': 0.05},
        {'asama_adi': '2. AŞAMA: Orta Hızda Tarama', 'max_workers': 5, 'bekleme_suresi': 1.5},
        {'asama_adi': '3. AŞAMA: Yavaş Tarama', 'max_workers': 2, 'bekleme_suresi': 3.0},
    ]

    tum_sonuclar = []
    basarisiz_fonlar_listesi = MANUAL_FON_KODLARI

    for asama in asama_parametreleri:
        if not basarisiz_fonlar_listesi:
            print(f"\n{asama['asama_adi']} atlanıyor. Önceki aşamada tüm fonlar başarıyla çekildi.")
            break
        
        yeni_sonuclar, yeni_basarisiz_fonlar = tarama_asamasini_calistir(
            fon_kodlari=basarisiz_fonlar_listesi,
            max_workers=asama['max_workers'],
            bekleme_suresi=asama['bekleme_suresi'],
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            asama_adi=asama['asama_adi']
        )
        
        tum_sonuclar.extend(yeni_sonuclar)
        basarisiz_fonlar_listesi = list(yeni_basarisiz_fonlar.keys())

    # Raporlama
    if not tum_sonuclar:
        print("\n--- SONUÇ: Analiz edilecek yeterli veri bulunamadı. ---")
    else:
        df_sonuc = pd.DataFrame(tum_sonuclar)
        sutun_sirasi = ['Fon Kodu', 'Fon Adı', 'Yatırımcı Sayısı', 'Piyasa Değeri (TL)', 'Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)', 'Getiri (%)', 'Standart Sapma (Yıllık %)', 'Sentiment Puanı']
        df_sonuc = df_sonuc[sutun_sirasi]
        df_sonuc_sirali = df_sonuc.sort_values(by=['Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)'], ascending=[False, False])

        print("\n--- BİRLEŞTİRİLMİŞ FON ANALİZ SONUÇLARI ---")
        print(df_sonuc_sirali.to_string())

        excel_dosya_adi = f"Hisse_Senedi_Fon_Analizi_{end_date_str}.xlsx"
        try:
            with pd.ExcelWriter(excel_dosya_adi, engine='xlsxwriter') as writer:
                df_sonuc_sirali.to_excel(writer, sheet_name='Fon Analizi', index=False)
                worksheet = writer.sheets['Fon Analizi']
                for i, col in enumerate(df_sonuc_sirali.columns):
                    column_len = max(df_sonuc_sirali[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, column_len)
            print(f"\n✅ Analiz sonuçları '{excel_dosya_adi}' dosyasına başarıyla kaydedildi.")
        except Exception as e:
            print(f"❌ Hata: Excel dosyası yazılırken bir sorun oluştu: {e}")
            
    # Özet Raporu
    toplam_fon_sayisi = len(MANUAL_FON_KODLARI)
    basarili_fon_sayisi = len(tum_sonuclar)
    nihai_basarisiz_fon_sayisi = len(basarisiz_fonlar_listesi)
    
    nihai_basarisiz_detaylari = ""
    if basarisiz_fonlar_listesi:
        # Son aşamada başarısız olanların detaylarını al
        _, nihai_basarisiz_fonlar_dict = tarama_asamasini_calistir(basarisiz_fonlar_listesi, 1, 0, start_date_str, end_date_str, 'Hata Raporlama Aşaması')
        nihai_basarisiz_detaylari = "\n".join([f"- {kod}: {sebep}" for kod, sebep in nihai_basarisiz_fonlar_dict.items()])

    ozet_metni = f"""
--- Analiz Özet Raporu ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---
Toplam Analize Alınan Fon Sayısı: {toplam_fon_sayisi}
Başarıyla Analiz Edilen Fon Sayısı: {basarili_fon_sayisi}
Nihai Başarısız Fon Sayısı: {nihai_basarisiz_fon_sayisi}

Başarısız Olan Fonlar ve Nedenleri:
{nihai_basarisiz_detaylari if nihai_basarisiz_detaylari else 'Tüm fonlar başarıyla işlendi.'}

Olası Genel Hata Nedenleri:
- TEFAS web sitesi, kısa sürede yapılan çok sayıda isteği engellemek için 'Rate Limiting' (erişim kısıtlaması) uygulamaktadır.
- Veri çekme işlemi sırasında internet bağlantısı kaynaklı zaman aşımları ('Timeout') yaşanmış olabilir.
- Listede yer alan bazı fon kodları güncel olmayabilir veya TEFAS platformunda bulunamayabilir.
"""
    print(ozet_metni)
    ozet_dosya_adi = "analiz_ozeti.txt"
    with open(ozet_dosya_adi, "w", encoding="utf-8") as f:
        f.write(ozet_metni)
    print(f"✅ Özet rapor '{ozet_dosya_adi}' dosyasına da kaydedildi.")

if __name__ == '__main__':
    main()
