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
MANUAL_FON_KODLARI = ['AAV', 'ACC', 'ACD', 'ADP', 'AED', 'AEV', 'AHI', 'AK3', 'AKU', 'ALC', 'AN1', 'AOJ', 'ARM', 'APJ', 'ASJ', 'AYA', 'BDS', 'BDY', 'BFT', 'BHA', 'BHL', 'BID', 'BIG', 'BIH', 'BIO', 'BNH', 'BRT', 'BST', 'BTE', 'BUL', 'BTZ', 'BUV', 'BUY', 'BVM', 'CIN', 'CKL', 'DAH', 'DBP', 'DBA', 'DEF', 'DDA', 'DFC', 'DHJ', 'DHM', 'DKH', 'DKR', 'DLD', 'DPT', 'DTL', 'DTZ', 'DTM', 'DXP', 'DYN', 'DZE', 'ENJ', 'ELZ', 'ESG', 'ESP', 'EID', 'FBC', 'FD1', 'FCK', 'FID', 'FDG', 'FIB', 'FPH', 'FRC', 'FSG', 'FUA', 'FYD', 'FZJ', 'GAE', 'GAF', 'GBH', 'GGK', 'GHS', 'GIE', 'GIH', 'GJB', 'GKF', 'GKG', 'GKV', 'GL1', 'GMA', 'GMR', 'GNH', 'GNS', 'GO1', 'GO2', 'GO3', 'GO4', 'GO9', 'GOH', 'GOL', 'GPU', 'GRT', 'GSP', 'GTA', 'GTM', 'GTH', 'GTY', 'GVA', 'GVI', 'GZJ', 'GZP', 'GZR', 'HBN', 'HBU', 'HGM', 'HFI', 'HGV', 'HIH', 'HJB', 'HKG', 'HKM', 'HKR', 'HKH', 'HMS', 'HPD', 'HNC', 'HRZ', 'HSA', 'HTJ', 'HVU', 'HVZ', 'IAE', 'HVS', 'ICC', 'ICF', 'ICH', 'ICV', 'ICZ', 'IDD', 'IDI', 'IDY', 'IFN', 'IHA', 'IHK', 'IHP', 'IHZ', 'IHT', 'IIH', 'IJA', 'IJB', 'IMB', 'IML', 'IPJ', 'IUV', 'IYB', 'JUP', 'KHA', 'KHB', 'KHC', 'KHT', 'KIA', 'KLH', 'KOT', 'KPA', 'KPH', 'KRF', 'KVT', 'KYA', 'KZL', 'KZU', 'LID', 'LLA', 'MAC', 'MAD', 'MD1', 'MD2', 'MHF', 'MGB', 'MKA', 'MMH', 'MTG', 'MTH', 'MTK', 'MTX', 'NAU', 'NJF', 'NHY', 'NNF', 'NRC', 'NPH', 'OBP', 'ODD', 'ODV', 'OIL', 'OHK', 'OJK', 'OHB', 'OKD', 'OMG', 'ONE', 'OPF', 'OPH', 'OPI', 'ORC', 'OTM', 'PAO', 'PBI', 'PBN', 'PEA', 'PGS', 'PHE', 'PHI', 'PHK', 'PKF', 'POS', 'PMP', 'PP1', 'PPB', 'PPH', 'PTO', 'PYR', 'RBN', 'RDF', 'RHI', 'RHS', 'RIK', 'RKH', 'RPD', 'RPI', 'RTH', 'RTP', 'SAS', 'SBH', 'SKO', 'SNY', 'SRL', 'SSS', 'ST1', 'SUB', 'SVB', 'TAR', 'TCD', 'TE3', 'TGA', 'THT', 'THF', 'THD', 'TI2', 'TI3', 'TIE', 'TJI', 'TJF', 'TKF', 'TLH', 'TLY', 'TPP', 'TPV', 'TRO', 'TUA', 'TTL', 'TZD', 'TYH', 'UPH', 'VAY', 'VCY', 'YAC', 'YAS', 'YAN', 'YBR', 'YDI', 'YDP', 'YEF', 'YHI', 'YHB', 'YHS', 'YKT', 'YLE', 'YLC', 'YLY', 'YPC', 'YPV', 'YPR', 'YSU', 'YUB', 'ZBD', 'ZBO', 'ZCD', 'ZDZ', 'ZFB', 'ZJB', 'ZHH', 'ZJI', 'ZLH', 'ZJV']

def get_last_business_day():
    print("LOG: Son iş günü hesaplanıyor...")
    today = datetime.now()
    for i in range(7):
        day_to_check = today - timedelta(days=i)
        if day_to_check.weekday() < 5:
            print(f"LOG: Son iş günü bulundu: {day_to_check.strftime('%Y-%m-%d')}")
            return day_to_check
    # Fallback
    fallback_day = today - timedelta(days=1)
    print(f"LOG: Son iş günü bulunamadı, fallback kullanılıyor: {fallback_day.strftime('%Y-%m-%d')}")
    return fallback_day

def get_fon_verileri_parallel(args):
    fon_kodu, start_date, end_date, bekleme_suresi_araligi = args
    max_retries = 3
    print(f"LOG: '{fon_kodu}' için veri çekme işlemi başlatıldı. Tarih aralığı: {start_date} -> {end_date}")
    for attempt in range(max_retries):
        try:
            if bekleme_suresi_araligi:
                sleep_time = random.uniform(bekleme_suresi_araligi[0], bekleme_suresi_araligi[1])
                print(f"LOG: '{fon_kodu}' için {sleep_time:.2f} saniye bekleniyor...")
                time.sleep(sleep_time)

            crawler = Crawler()
            df = crawler.fetch(
                start=start_date,
                end=end_date,
                name=fon_kodu,
                columns=["date", "price", "market_cap", "number_of_investors", "title"]
            )

            if df.empty:
                print(f"UYARI: '{fon_kodu}' için TEFAS'tan veri bulunamadı.")
                return fon_kodu, None, "Veri bulunamadı"

            df['date'] = pd.to_datetime(df['date'])
            fon_adi = df['title'].iloc[0] if not df.empty else fon_kodu
            print(f"LOG: '{fon_kodu}' ({fon_adi}) için {len(df)} satır veri başarıyla çekildi.")
            return fon_kodu, fon_adi, df.sort_values(by='date').reset_index(drop=True)
        except Exception as e:
            error_message = str(e)
            print(f"HATA: '{fon_kodu}' verisi çekilirken bir sorun oluştu: {error_message}")
            if "rate limiting" in error_message.lower() or "robot check" in error_message.lower() or "timeout" in error_message.lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10 + random.uniform(0, 5)
                    print(f"LOG: Tekrar deneme... {wait_time:.2f} saniye bekleniyor... (Deneme {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    return fon_kodu, None, f"Veri çekme başarısız. Son deneme de rate-limiting/timeout hatası verdi. Hata: {error_message}"
            else:
                return fon_kodu, None, f"Beklenmedik hata: {error_message}"
    return fon_kodu, None, "Tüm denemeler başarısız oldu."

def hesapla_metrikler(df_fon_fiyat):
    if df_fon_fiyat is None or len(df_fon_fiyat) < 10:
        print(f"LOG: Metrik hesaplama için yetersiz veri (veri yok veya satır sayısı < 10).")
        return None
    df_fon_fiyat['daily_return'] = df_fon_fiyat['price'].pct_change()
    df_fon_fiyat = df_fon_fiyat.dropna()
    if df_fon_fiyat.empty:
        print("LOG: Günlük getiriler hesaplandıktan sonra veri kalmadı.")
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

def analyze_daily_correlations(df_fon_data, fon_kodu):
    if df_fon_data is None or df_fon_data.empty:
        print(f"LOG: '{fon_kodu}' için korelasyon analizi yapılamadı: Veri yok veya boş.")
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
        weighted_score = (
            (same_day_positive_price_correlation * weights['same_day_price']) +
            (same_day_positive_correlation * weights['same_day_market_cap']) +
            (next_day_positive_price_correlation * weights['next_day_price']) +
            (next_day_positive_correlation * weights['next_day_market_cap'])
        )
        max_possible_weighted_score = total_investor_increases * sum(weights.values())
        if max_possible_weighted_score > 0:
            sentiment_score = (weighted_score / max_possible_weighted_score) * 100
    
    print(f"--- {fon_kodu} için Sentiment Analizi ---")
    print(f"Toplam yatırımcı artışı olan gün sayısı: {total_investor_increases}")
    print(f"HESAPLANAN SENTIMENT PUANI: {round(sentiment_score, 2):.2f}")
    print("--------------------------------------------------")
    
    return {
        'total_investor_increases': total_investor_increases,
        'same_day_market_cap_increase': same_day_positive_correlation,
        'same_day_price_increase': same_day_positive_price_correlation,
        'next_day_market_cap_increase': next_day_positive_correlation,
        'next_day_price_increase': next_day_positive_price_correlation
    }, round(sentiment_score, 2)

def tarama_asamasini_calistir(fon_kodlari, max_workers, bekleme_suresi_araligi, start_date_str, end_date_str, asama_adi):
    print(f"\n--- {asama_adi} ({len(fon_kodlari)} adet fon) başlatılıyor ---")
    analiz_sonuclari = []
    basarisiz_fonlar = {}
    tasks = [(fon_kodu, start_date_str, end_date_str, bekleme_suresi_araligi) for fon_kodu in fon_kodlari]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_fon = {executor.submit(get_fon_verileri_parallel, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_fon):
            fon_kodu, fon_adi, data_or_error = future.result()
            if isinstance(data_or_error, pd.DataFrame):
                data = data_or_error
                print(f"LOG: '{fon_kodu}' için metrikler ve sentiment analizi hesaplanıyor.")
                _, sentiment_score = analyze_daily_correlations(data, fon_kodu)
                metrikler = hesapla_metrikler(data)
                
                if metrikler:
                    print(f"LOG: '{fon_kodu}' için metrikler başarıyla hesaplandı.")
                    sonuc = {'Fon Kodu': fon_kodu, 'Fon Adı': fon_adi, **metrikler, 'Sentiment Puanı': sentiment_score}
                    analiz_sonuclari.append(sonuc)
                else:
                    print(f"UYARI: '{fon_kodu}' için metrik hesaplanamadı (yetersiz veri).")
                    basarisiz_fonlar[fon_kodu] = "Metrik hesaplanamadı (yetersiz veri)."
            else:
                print(f"UYARI: '{fon_kodu}' için veri alınamadı. Hata: {data_or_error}")
                basarisiz_fonlar[fon_kodu] = data_or_error

    print(f"\n--- {asama_adi} tamamlandı ---")
    print(f"Toplam {len(tasks)} fondan {len(analiz_sonuclari)} tanesi başarıyla işlendi.")
    print(f"{len(basarisiz_fonlar)} fon başarısız oldu veya atlandı.")
    return analiz_sonuclari, basarisiz_fonlar

def main():
    print(f"--- LOG: Script Başlatıldı --- ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    end_date = get_last_business_day()
    start_date = end_date - pd.DateOffset(months=ANALIZ_SURESI_AY)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"--- ANALİZ BAŞLATILDI ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    print(f"Analiz Tarih Aralığı: {start_date_str} -> {end_date_str}")
    print(f"Toplam analize alınacak fon sayısı: {len(MANUAL_FON_KODLARI)}")

    asama_parametreleri = [
        {'asama_adi': '1. AŞAMA: Hızlı Tarama', 'max_workers': 8, 'bekleme_suresi_araligi': (1.0, 2.0)},
        {'asama_adi': '2. AŞAMA: Orta Hızda Tarama', 'max_workers': 4, 'bekleme_suresi_araligi': (3.0, 5.0)},
        {'asama_adi': '3. AŞAMA: Yavaş Tarama', 'max_workers': 1, 'bekleme_suresi_araligi': (5.0, 7.0)},
    ]

    tum_sonuclar = []
    basarisiz_fonlar_listesi = MANUAL_FON_KODLARI
    son_basarisiz_dict = {}

    for asama in asama_parametreleri:
        if not basarisiz_fonlar_listesi:
            print(f"\nLOG: {asama['asama_adi']} atlanıyor. Önceki aşamada tüm fonlar başarıyla çekildi.")
            break
        
        yeni_sonuclar, son_basarisiz_dict = tarama_asamasini_calistir(
            fon_kodlari=basarisiz_fonlar_listesi,
            max_workers=asama['max_workers'],
            bekleme_suresi_araligi=asama['bekleme_suresi_araligi'],
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            asama_adi=asama['asama_adi']
        )
        
        if asama['asama_adi'] == '1. AŞAMA: Hızlı Tarama' and not yeni_sonuclar:
            print("\nHATA: İlk aşamada hiçbir fonun verisi çekilemedi. İşlem durduruluyor.")
            return
            
        tum_sonuclar.extend(yeni_sonuclar)
        basarisiz_fonlar_listesi = list(son_basarisiz_dict.keys())

    print("\n--- LOG: Raporlama Aşaması Başladı ---")
    if not tum_sonuclar:
        print("\n--- SONUÇ: Analiz edilecek yeterli veri bulunamadı. Excel dosyası oluşturulmayacak. ---")
    else:
        print(f"LOG: {len(tum_sonuclar)} adet fon için sonuç DataFrame'i oluşturuluyor.")
        df_sonuc = pd.DataFrame(tum_sonuclar)
        sutun_sirasi = [
            'Fon Kodu', 'Fon Adı', 'Yatırımcı Sayısı', 'Piyasa Değeri (TL)',
            'Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)', 'Getiri (%)',
            'Standart Sapma (Yıllık %)', 'Sentiment Puanı'
        ]
        df_sonuc = df_sonuc[sutun_sirasi]
        df_sonuc_sirali = df_sonuc.sort_values(by=['Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)'], ascending=[False, False])

        print("\n--- BİRLEŞTİRİLMİŞ FON ANALİZ SONUÇLARI ---")
        print(df_sonuc_sirali.to_string())

        excel_dosya_adi = f"Hisse_Senedi_Fon_Analizi_{end_date_str}.xlsx"
        print(f"LOG: Sonuçlar Excel dosyasına yazılıyor: '{excel_dosya_adi}'")
        try:
            with pd.ExcelWriter(excel_dosya_adi, engine='xlsxwriter') as writer:
                df_sonuc_sirali.to_excel(writer, sheet_name='Fon Analizi', index=False)
                worksheet = writer.sheets['Fon Analizi']
                for i, col in enumerate(df_sonuc_sirali.columns):
                    column_len = max(df_sonuc_sirali[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, column_len)
            print(f"LOG: Analiz sonuçları '{excel_dosya_adi}' dosyasına başarıyla kaydedildi.")
        except Exception as e:
            print(f"HATA: Excel dosyası '{excel_dosya_adi}' oluşturulurken bir hata oluştu: {e}")

    print("\n--- LOG: Özet Raporu Oluşturuluyor ---")
    toplam_fon_sayisi = len(MANUAL_FON_KODLARI)
    basarili_fon_sayisi = len(tum_sonuclar)
    nihai_basarisiz_fon_sayisi = len(son_basarisiz_dict)
    
    basarisiz_detaylari = "\n".join([f"- {kod}: {sebep}" for kod, sebep in son_basarisiz_dict.items()])

    ozet_metni = f"""
--- Analiz Özet Raporu ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---
Toplam Analize Alınan Fon Sayısı: {toplam_fon_sayisi}
Başarıyla Analiz Edilen Fon Sayısı: {basarili_fon_sayisi}
Nihai Başarısız Fon Sayısı: {nihai_basarisiz_fon_sayisi}

Başarısız Olan Fonlar ve Nedenleri:
{basarisiz_detaylari if basarisiz_detaylari else 'Tüm fonlar başarıyla işlendi.'}

Olası Genel Hata Nedenleri:
- TEFAS web sitesi, kısa sürede yapılan çok sayıda isteği engellemek için 'Rate Limiting' (erişim kısıtlaması) uygulamaktadır.
- Veri çekme işlemi sırasında internet bağlantısı kaynaklı zaman aşımları ('Timeout') yaşanmış olabilir.
- Listede yer alan bazı fon kodları güncel olmayabilir veya TEFAS platformunda bulunamayabilir.
"""
    print(ozet_metni)
    ozet_dosya_adi = "analiz_ozeti.txt"
    try:
        with open(ozet_dosya_adi, "w", encoding="utf-8") as f:
            f.write(ozet_metni)
        print(f"LOG: Özet rapor '{ozet_dosya_adi}' dosyasına da kaydedildi.")
    except Exception as e:
        print(f"HATA: Özet rapor '{ozet_dosya_adi}' oluşturulurken bir hata oluştu: {e}")
    
    print(f"--- LOG: Script Tamamlandı --- ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

if __name__ == '__main__':
    main()
