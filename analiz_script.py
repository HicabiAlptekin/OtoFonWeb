# -*- coding: utf-8 -*-
# Fonaliz - Dinamik Fon Analiz Aracı (v2.1)
# Bu script, 'filtrelenmis_fonlar.txt' dosyasında listelenen fonlar için
# detaylı risk ve getiri analizi yapar.
# v2.1 - tefas-crawler v0.6.0+ API desteği + 6 günlük iş günü trend analizi

import pandas as pd
import numpy as np
from datetime import date, timedelta
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

# Resmi tatiller (2026)
RESMI_TATILLER_2026 = [
    date(2026, 1, 1),
    date(2026, 4, 23),
    date(2026, 5, 1),
    date(2026, 5, 19),
    date(2026, 7, 15),
    date(2026, 8, 30),
    date(2026, 10, 29),
]

# Global TEFAS crawler (tekil örnek)
try:
    tefas_crawler_global = Crawler()
    print("TEFAS Crawler başarıyla başlatıldı.")
except Exception as e:
    print("TEFAS Crawler başlatılırken hata: {}".format(e))
    tefas_crawler_global = None


# --- İş Günü Fonksiyonları ---
def is_business_day(check_date):
    """Belirtilen tarihin iş günü olup olmadığını kontrol eder."""
    if check_date.weekday() >= 5:  # Cumartesi/Pazar
        return False
    if check_date in RESMI_TATILLER_2026:
        return False
    return True


def get_previous_business_day(input_date):
    """Bir önceki iş gününü döndürür."""
    prev_day = input_date - timedelta(days=1)
    while not is_business_day(prev_day):
        prev_day -= timedelta(days=1)
    return prev_day


def get_business_days(count, end_date=None):
    """Belirtilen sayıda iş gününü döndürür."""
    if end_date is None:
        end_date = date.today()

    current_date = end_date
    if not is_business_day(current_date):
        current_date = get_previous_business_day(current_date)

    business_days = []
    while len(business_days) < count:
        if is_business_day(current_date):
            business_days.append(current_date)
        current_date = get_previous_business_day(current_date)

    return business_days


def get_last_6_business_days():
    """Son 6 iş gününü döndürür (önceki 3 + son 3)."""
    today = date.today()

    if not is_business_day(today):
        current_end = get_previous_business_day(today)
    else:
        current_end = today

    # Son 3 iş günü
    last_3_days = get_business_days(3, current_end)
    # Önceki 3 iş günü
    first_of_previous = get_previous_business_day(last_3_days[-1])
    previous_3_days = get_business_days(3, first_of_previous)

    return {
        'recent_3_days': last_3_days,
        'previous_3_days': previous_3_days,
        'all_6_days': previous_3_days + last_3_days
    }


# --- Yardımcı Fonksiyonlar ---
def load_filtered_fund_list():
    """
    'filtrelenmis_fonlar.txt' dosyasından analiz edilecek fon kodlarının
    listesini okur.
    """
    if not os.path.exists(INPUT_FILE):
        print("HATA: Analiz için fon listesini içeren '{}' dosyası bulunamadı.".format(INPUT_FILE))
        print("Lütfen önce tarama script'ini çalıştırarak bu dosyayı oluşturun.")
        sys.exit(1)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        fon_kodlari = [line.strip() for line in f if line.strip()]

    if not fon_kodlari:
        print("UYARI: '{}' dosyası boş. Analiz edilecek fon bulunamadı.".format(INPUT_FILE))
        sys.exit(0)

    print("'{}' dosyasından {} adet fon kodu okundu.".format(INPUT_FILE, len(fon_kodlari)))
    return fon_kodlari


def fetch_data_for_fund_parallel(args):
    """
    Verilen bir fon kodu için TEFAS v0.6.0 API'sinden veri çeker.
    """
    fon_kodu, start_date, end_date = args
    global tefas_crawler_global
    if tefas_crawler_global is None:
        return fon_kodu, None, None

    try:
        df = tefas_crawler_global.fetch(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            name=fon_kodu,
            columns=["date", "price", "title"]
        )

        if df.empty:
            return fon_kodu, None, None

        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        fon_adi = df['title'].iloc[0] if 'title' in df.columns and not df.empty else fon_kodu
        return fon_kodu, fon_adi, df.sort_values(by='date').reset_index(drop=True)

    except Exception as e:
        print("HATA ({}): Veri çekilirken sorun oluştu - {}".format(fon_kodu, e))
        return fon_kodu, None, None


def get_value_on_or_before(df, target_date, column='price'):
    """Belirtilen tarihten önceki veya o tarihteki değeri döndürür."""
    if df is None or df.empty or target_date is None:
        return None
    df_filtered = df[df['date'] <= target_date]
    if not df_filtered.empty:
        return df_filtered.sort_values('date', ascending=False)[column].iloc[0]
    return None


def hesapla_metrikler(df_fon_fiyat):
    """
    Bir fonun geçmiş fiyat verilerini kullanarak risk/getiri metriklerini hesaplar.
    Tüm dönem getirileri: haftalık, 2 haftalık, aylık, 3 aylık, 6 aylık, yılbaşı, 1 yıl
    """
    if df_fon_fiyat is None or len(df_fon_fiyat) < 10:
        return None

    df_fon_fiyat = df_fon_fiyat.sort_values('date').reset_index(drop=True)
    df_fon_fiyat['daily_return'] = df_fon_fiyat['price'].pct_change()
    df_fon_fiyat = df_fon_fiyat.dropna()
    if df_fon_fiyat.empty:
        return None

    today = df_fon_fiyat['date'].max()

    # Tüm dönem getirilerini hesapla
    metrikler = {}

    # 1 hafta (5 iş günü)
    hafta1_fiyat = get_value_on_or_before(df_fon_fiyat, today - timedelta(days=7))
    if hafta1_fiyat:
        metrikler['Haftalik_%'] = round(((df_fon_fiyat['price'].iloc[-1] / hafta1_fiyat) - 1) * 100, 2)

    # 2 hafta (10 iş günü)
    hafta2_fiyat = get_value_on_or_before(df_fon_fiyat, today - timedelta(days=14))
    if hafta2_fiyat:
        metrikler['2_Haftalik_%'] = round(((df_fon_fiyat['price'].iloc[-1] / hafta2_fiyat) - 1) * 100, 2)

    # 1 ay
    ay1_fiyat = get_value_on_or_before(df_fon_fiyat, today - relativedelta(months=1))
    if ay1_fiyat:
        metrikler['Aylik_%'] = round(((df_fon_fiyat['price'].iloc[-1] / ay1_fiyat) - 1) * 100, 2)

    # 3 ay
    ay3_fiyat = get_value_on_or_before(df_fon_fiyat, today - relativedelta(months=3))
    if ay3_fiyat:
        metrikler['3_Aylik_%'] = round(((df_fon_fiyat['price'].iloc[-1] / ay3_fiyat) - 1) * 100, 2)

    # 6 ay
    ay6_fiyat = get_value_on_or_before(df_fon_fiyat, today - relativedelta(months=6))
    if ay6_fiyat:
        metrikler['6_Aylik_%'] = round(((df_fon_fiyat['price'].iloc[-1] / ay6_fiyat) - 1) * 100, 2)

    # Yılbaşından itibaren
    yilbasi = date(today.year, 1, 1)
    yilbasi_fiyat = get_value_on_or_before(df_fon_fiyat, yilbasi)
    if yilbasi_fiyat:
        metrikler['Yilbasi_%'] = round(((df_fon_fiyat['price'].iloc[-1] / yilbasi_fiyat) - 1) * 100, 2)

    # 1 yıl
    yil1_fiyat = get_value_on_or_before(df_fon_fiyat, today - relativedelta(years=1))
    if yil1_fiyat:
        metrikler['1_Yillik_%'] = round(((df_fon_fiyat['price'].iloc[-1] / yil1_fiyat) - 1) * 100, 2)

    # Genel getiri (analiz dönemi)
    getiri = (df_fon_fiyat['price'].iloc[-1] / df_fon_fiyat['price'].iloc[0]) - 1
    metrikler['Getiri_%'] = round(getiri * 100, 2)

    # Volatilite ve Sharpe
    volatilite = df_fon_fiyat['daily_return'].std() * np.sqrt(252)
    ortalama_gunluk_getiri = df_fon_fiyat['daily_return'].mean()

    gunluk_std = df_fon_fiyat['daily_return'].std()
    sharpe_orani = (ortalama_gunluk_getiri / gunluk_std) * np.sqrt(252) if gunluk_std != 0 else 0

    # Sortino Oranı
    negatif_getiriler = df_fon_fiyat[df_fon_fiyat['daily_return'] < 0]['daily_return']
    if negatif_getiriler.empty or negatif_getiriler.std() == 0:
        sortino_orani = 0
    else:
        downside_deviation = negatif_getiriler.std() * np.sqrt(252)
        sortino_orani = (ortalama_gunluk_getiri * 252) / downside_deviation if downside_deviation != 0 else 0

    metrikler['Volatilite_%'] = round(volatilite * 100, 2)
    metrikler['Sharpe_Orani'] = round(sharpe_orani, 2)
    metrikler['Sortino_Orani'] = round(sortino_orani, 2)

    return metrikler


def hesapla_6gunluk_trend(df_fon_fiyat):
    """
    6 günlük iş günü trend analizi yapar.
    Son 3 iş günü vs Önceki 3 iş günü karşılaştırması.
    """
    if df_fon_fiyat is None or len(df_fon_fiyat) < 10:
        return None

    try:
        is_gunleri = get_last_6_business_days()
        recent_days = is_gunleri['recent_3_days']
        previous_days = is_gunleri['previous_3_days']
    except Exception:
        return None

    if len(recent_days) != 3 or len(previous_days) != 3:
        return None

    # Fiyatları al
    df_fon_fiyat = df_fon_fiyat.sort_values('date').reset_index(drop=True)

    fiyatlar = {}
    for tarih in set(recent_days + previous_days):
        df_filtered = df_fon_fiyat[df_fon_fiyat['date'] <= tarih]
        if not df_filtered.empty:
            fiyatlar[tarih] = df_filtered.sort_values('date', ascending=False)['price'].iloc[0]

    if len(fiyatlar) < 4:
        return None

    # Son 3 iş günü getirileri
    son_3_getiriler = []
    for i in range(len(recent_days) - 1):
        if recent_days[i] in fiyatlar and recent_days[i+1] in fiyatlar:
            degisim = ((fiyatlar[recent_days[i]] / fiyatlar[recent_days[i+1]]) - 1) * 100
            son_3_getiriler.append(degisim)

    onceki_3_getiriler = []
    for i in range(len(previous_days) - 1):
        if previous_days[i] in fiyatlar and previous_days[i+1] in fiyatlar:
            degisim = ((fiyatlar[previous_days[i]] / fiyatlar[previous_days[i+1]]) - 1) * 100
            onceki_3_getiriler.append(degisim)

    if not son_3_getiriler or not onceki_3_getiriler:
        return None

    son_3g_ort = np.mean(son_3_getiriler)
    onceki_3g_ort = np.mean(onceki_3_getiriler)

    # Trend yönü
    if son_3g_ort > 0 and onceki_3g_ort > 0:
        trend_yonu = "HIZLANAN" if son_3g_ort > onceki_3g_ort else "YUKSELEN"
    elif son_3g_ort > 0 and onceki_3g_ort <= 0:
        trend_yonu = "DONUS_POZITIF"
    elif son_3g_ort <= 0 and onceki_3g_ort > 0:
        trend_yonu = "DONUS_NEGATIF"
    else:
        trend_yonu = "DUSEN"

    return {
        'Son_3G_Ort_%': round(son_3g_ort, 4),
        'Onceki_3G_Ort_%': round(onceki_3g_ort, 4),
        'Trend_Yonu': trend_yonu
    }
    """
    Bir fonun geçmiş fiyat verilerini kullanarak risk/getiri metriklerini hesaplar.
    Not: v0.6.0 API'de market_cap ve number_of_investors sütunları kaldırılmıştır.
    """
    if df_fon_fiyat is None or len(df_fon_fiyat) < 10:
        return None

    df_fon_fiyat['daily_return'] = df_fon_fiyat['price'].pct_change()
    df_fon_fiyat = df_fon_fiyat.dropna()
    if df_fon_fiyat.empty:
        return None

    getiri = (df_fon_fiyat['price'].iloc[-1] / df_fon_fiyat['price'].iloc[0]) - 1
    volatilite = df_fon_fiyat['daily_return'].std() * np.sqrt(252)
    ortalama_gunluk_getiri = df_fon_fiyat['daily_return'].mean()

    # Sharpe Oranı
    gunluk_std = df_fon_fiyat['daily_return'].std()
    sharpe_orani = (ortalama_gunluk_getiri / gunluk_std) * np.sqrt(252) if gunluk_std != 0 else 0

    # Sortino Oranı
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
    }


def main():
    """
    Ana fonksiyon: fon listesini okur, verileri çeker, analiz eder ve sonucu Excel'e yazar.
    """
    print("--- Fonaliz Dinamik Analiz Script'i Başlatıldı (v2.1) ---")
    print("6 Günlük İş Günü Trend Analizi Aktif")

    # İş günlerini göster
    try:
        is_gunleri = get_last_6_business_days()
        print("\n[IS GUNU] Hesaplanan tarih aralıkları:")
        print("  Önceki 3 iş günü: {}".format(', '.join([d.strftime('%d.%m.%Y') for d in is_gunleri['previous_3_days']])))
        print("  Son 3 iş günü   : {}".format(', '.join([d.strftime('%d.%m.%Y') for d in is_gunleri['recent_3_days']])))
    except Exception as e:
        print("[HATA] İş günü hesaplama başarısız: {}".format(e))

    start_time = time.time()

    fon_listesi = load_filtered_fund_list()

    end_date = date.today()
    # Veri buffer'ı için fazladan 30 gün ekle
    start_date = end_date - relativedelta(months=ANALIZ_SURESI_AY) - timedelta(days=30)

    tasks = [(fon_kodu, start_date, end_date) for fon_kodu in fon_listesi]
    analiz_sonuclari = []

    print("\n{} adet fon için {} - {} tarih aralığında analiz başlatılıyor...".format(
        len(fon_listesi), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, task): task[0] for task in tasks}

        for future in concurrent.futures.as_completed(future_to_fon):
            fon_kodu, fon_adi, data = future.result()
            if data is not None and not data.empty:
                metrikler = hesapla_metrikler(data)
                trend_verisi = hesapla_6gunluk_trend(data)
                if metrikler:
                    sonuc = {'Fon Kodu': fon_kodu, 'Fon Adı': fon_adi, **metrikler}
                    # Trend verisini ekle
                    if trend_verisi:
                        sonuc.update(trend_verisi)
                    analiz_sonuclari.append(sonuc)

    if not analiz_sonuclari:
        print("\n--- SONUÇ: Analiz edilecek yeterli veri bulunamadı. ---")
        return

    df_sonuc = pd.DataFrame(analiz_sonuclari)
    # Tüm metrikleri içeren sıralı liste
    sutun_sirasi = [
        'Fon Kodu', 'Fon Adı',
        # Dönem getirileri
        'Haftalik_%', '2_Haftalik_%', 'Aylik_%', '3_Aylik_%', '6_Aylik_%', 'Yilbasi_%', '1_Yillik_%', 'Getiri_%',
        # Risk metrikleri
        'Volatilite_%', 'Sharpe_Orani', 'Sortino_Orani',
        # Kısa trend
        'Son_3G_Ort_%', 'Onceki_3G_Ort_%', 'Trend_Yonu'
    ]
    # Sadece mevcut sutunlari kullan
    sutun_sirasi = [col for col in sutun_sirasi if col in df_sonuc.columns]
    df_sonuc = df_sonuc[sutun_sirasi]
    df_sonuc_sirali = df_sonuc.sort_values(
        by=['Sortino_Orani', 'Sharpe_Orani'],
        ascending=[False, False]
    )

    excel_dosya_adi = "Fonaliz_Sonuclari_{}_v2.xlsx".format(end_date.strftime('%Y-%m-%d'))
    print("\nAnaliz tamamlandı. Sonuçlar '{}' dosyasına yazılıyor...".format(excel_dosya_adi))

    try:
        with pd.ExcelWriter(excel_dosya_adi, engine='xlsxwriter') as writer:
            df_sonuc_sirali.to_excel(writer, sheet_name='Analiz Sonuclari', index=False)
            worksheet = writer.sheets['Analiz Sonuclari']
            for i, col in enumerate(df_sonuc_sirali.columns):
                column_len = max(df_sonuc_sirali[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, column_len)
        print("'{}' dosyası başarıyla oluşturuldu.".format(excel_dosya_adi))
    except Exception as e:
        print("HATA: Excel dosyası oluşturulurken bir sorun oluştu: {}".format(e))

    end_time = time.time()
    print("\n--- Tüm işlemler {:.2f} saniyede tamamlandı ---".format(end_time - start_time))


if __name__ == "__main__":
    main()
