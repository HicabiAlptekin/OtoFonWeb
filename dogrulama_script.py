# -*- coding: utf-8 -*-
# OtoFon TEFAS Veri Doğrulama Scripti
# Bu script, seçili fonlar için TEFAS API'sinden veri çeker ve
# aylık, 3 aylık, 6 aylık, yılbaşından itibaren ve 1 yıllık
# değişim oranlarını hesaplayarak doğrulama yapar.

import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from tefas import Crawler
import sys
import warnings
warnings.filterwarnings('ignore')

# TEFAS Crawler başlat
print("=" * 70)
print("OTOFON - TEFAS VERİ DOĞRULAMA ARACI")
print("=" * 70)

try:
    crawler = Crawler()
    print("[OK] TEFAS Crawler başlatıldı.")
except Exception as e:
    print(f"[HATA] Crawler başlatılamadı: {e}")
    sys.exit(1)

# Doğrulama için seçilen fonlar (çeşitli türlerden)
TEST_FONLARI = [
    "AFA",  # Hisse senedi fonu
    "TGE",  # Teknoloji fonu
    "MAD",  # Katılım fonu
    "HVS",  # Hisse senedi fonu
    "IPP",  # Serbest fon
]

BUGUN = date.today()
YIL_BASLANGICI = date(BUGUN.year, 1, 1)  # Yılbaşı

# Hesaplanacak dönemler
DONEMLER = {
    '1 Aylık': BUGUN - relativedelta(months=1),
    '3 Aylık': BUGUN - relativedelta(months=3),
    '6 Aylık': BUGUN - relativedelta(months=6),
    'Yılbaşından İtibaren': YIL_BASLANGICI,
    '1 Yıllık': BUGUN - relativedelta(years=1),
}


def get_price_on_or_before(df_fund_history, target_date):
    """Belirtilen tarihteki veya önceki en yakın fiyatı döndürür."""
    if df_fund_history is None or df_fund_history.empty or target_date is None:
        return np.nan
    df_filtered = df_fund_history[df_fund_history['date'] <= target_date].copy()
    if not df_filtered.empty:
        return df_filtered.sort_values(by='date', ascending=False)['price'].iloc[0]
    return np.nan


def fon_verisi_getir(fon_kodu, baslangic_tarihi, bitis_tarihi):
    """TEFAS v0.6.0 API'den fon verisi çeker."""
    try:
        df = crawler.fetch(
            start=baslangic_tarihi.strftime("%Y-%m-%d"),
            end=bitis_tarihi.strftime("%Y-%m-%d"),
            name=fon_kodu,
            columns=["date", "price", "title"]
        )

        if df.empty:
            return None, None, None

        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        df = df.sort_values(by='date').reset_index(drop=True)

        fon_adi = df['title'].iloc[0] if 'title' in df.columns else fon_kodu
        return fon_kodu, fon_adi, df
    except Exception as e:
        print(f"  [HATA] {fon_kodu}: {e}")
        return fon_kodu, None, None


# Ana doğrulama döngüsü
print(f"\n{'=' * 70}")
print(f"Doğrulama Tarihi: {BUGUN.strftime('%d.%m.%Y')}")
print(f"Test Edilen Fon Sayısı: {len(TEST_FONLARI)}")
print(f"{'=' * 70}\n")

tum_sonuclar = []

for fon_kodu in TEST_FONLARI:
    print(f"{'-' * 50}")
    print(f"Fon: {fon_kodu}")
    print(f"{'-' * 50}")

    # Veri çekme başlangıcını 1 yıl 3 ay geri al
    veri_baslangic = BUGUN - relativedelta(years=1, months=3)
    _, fon_adi, df_fiyat = fon_verisi_getir(fon_kodu, veri_baslangic, BUGUN)

    if df_fiyat is None or df_fiyat.empty:
        print(f"  [UYARI] {fon_kodu} için veri bulunamadı.")
        continue

    print(f"  Fon Adı   : {fon_adi if fon_adi else 'Bilinmiyor'}")
    print(f"  Veri Sayısı: {len(df_fiyat)} gün")
    print(f"  Tarih     : {df_fiyat['date'].iloc[0]} - {df_fiyat['date'].iloc[-1]}")
    print(f"  Son Fiyat : {df_fiyat['price'].iloc[-1]:.6f} TL")
    print()

    # Son fiyat
    son_fiyat = get_price_on_or_before(df_fiyat, BUGUN)

    sonuc = {'Fon Kodu': fon_kodu, 'Fon Adı': fon_adi}

    print(f"  {'Dönem':<25} {'Önceki Fiyat':<15} {'Değişim %':<10}")
    print(f"  {'-'*50}")

    for donem_adi, donem_baslangic in DONEMLER.items():
        onceki_fiyat = get_price_on_or_before(df_fiyat, donem_baslangic)

        if pd.notna(son_fiyat) and pd.notna(onceki_fiyat) and onceki_fiyat != 0:
            degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
            print(f"  {donem_adi:<25} {onceki_fiyat:<15.6f} {degisim:<+10.2f}")
            sonuc[donem_adi] = round(degisim, 2)
        else:
            print(f"  {donem_adi:<25} {'---':<15} {'---':<10}")
            sonuc[donem_adi] = None

    tum_sonuclar.append(sonuc)
    print()

# Özet tablo
print(f"\n{'=' * 70}")
print("DOĞRULAMA SONUÇLARI - ÖZET TABLO")
print(f"{'=' * 70}")

if tum_sonuclar:
    df_ozet = pd.DataFrame(tum_sonuclar)
    sutunlar = ['Fon Kodu', 'Fon Adı'] + list(DONEMLER.keys())
    sutunlar = [col for col in sutunlar if col in df_ozet.columns]
    df_ozet = df_ozet[sutunlar]
    
    print(f"\n{df_ozet.to_string(index=False)}")
    
    # İstatistikler
    print(f"\n{'=' * 70}")
    print("İSTATİSTİKLER")
    print(f"{'=' * 70}")
    sayisal_sutunlar = [col for col in DONEMLER.keys() if col in df_ozet.columns]
    for col in sayisal_sutunlar:
        vals = df_ozet[col].dropna()
        if not vals.empty:
            print(f"  {col:<25} | Ort: %{vals.mean():>+7.2f} | Min: %{vals.min():>+7.2f} | Max: %{vals.max():>+7.2f}")
else:
    print("\n[UYARI] Hiçbir fon için veri alınamadı.")

print(f"\n{'=' * 70}")
print("DOĞRULAMA TAMAMLANDI")
print(f"{'=' * 70}")
