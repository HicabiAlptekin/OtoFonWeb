# -*- coding: utf-8 -*-
# ENTEGRE FON TARAMA VE ANALİZ ARACI (OtoFon + Fonaliz + Trend Analizi)
# v3.1 - tefas-crawler v0.6.0+ API + 3 Günlük Trend Analizi + Excel Çıktısı
#
# Bu script:
# 1. Takasbank'tan fon listesini alır
# 2. TEFAS API'sinden (v0.6.0+) fon verilerini çeker
# 3. Haftalık getiri analizi yapar
# 4. 3 günlük kısa trend analizi yapar
# 5. Fonaliz risk/getiri metriklerini hesaplar
# 6. Tüm sonuçları Excel dosyasına kaydeder

import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from tefas import Crawler
from tqdm import tqdm
import concurrent.futures
import traceback
import warnings

warnings.filterwarnings('ignore')

# --- SABİTLER ---
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
MAX_WORKERS = 10
TREND_GUN_SAYISI = 3
EXCEL_DOSYASI = f"OtoFon_Raporu_{date.today().strftime('%Y-%m-%d')}.xlsx"

# Global TEFAS crawler
try:
    tefas_crawler_global = Crawler()
    print("[OK] TEFAS Crawler baslatildi.")
except Exception as e:
    print(f"[HATA] TEFAS Crawler: {e}")
    tefas_crawler_global = None


# --- YARDIMCI FONKSİYONLAR ---
def load_takasbank_fund_list():
    """Takasbank'tan güncel fon listesini yükler."""
    print("[INDIR] Takasbank fon listesi yukleniyor...")
    try:
        df_excel = pd.read_excel(TAKASBANK_EXCEL_URL, engine='openpyxl')
        df_data = df_excel[['Fon Adı', 'Fon Kodu']].copy()
        df_data['Fon Kodu'] = df_data['Fon Kodu'].astype(str).str.strip().str.upper()
        df_data.dropna(subset=['Fon Kodu'], inplace=True)
        df_data = df_data[df_data['Fon Kodu'] != '']
        print(f"[OK] {len(df_data)} fon bulundu.")
        return df_data
    except Exception as e:
        print(f"[HATA] Takasbank yukleme: {e}")
        return pd.DataFrame()


def get_value_on_or_before(df_fund, target_date, column='price'):
    if df_fund is None or df_fund.empty or target_date is None:
        return np.nan
    df_filtered = df_fund[df_fund['date'] <= target_date]
    if not df_filtered.empty:
        return df_filtered.sort_values(by='date', ascending=False)[column].iloc[0]
    return np.nan


def get_value_on_or_after(df_fund, target_date, column='price'):
    if df_fund is None or df_fund.empty or target_date is None:
        return np.nan
    df_filtered = df_fund[df_fund['date'] >= target_date]
    if not df_filtered.empty:
        return df_filtered.sort_values(by='date', ascending=True)[column].iloc[0]
    return np.nan


def calculate_change(current_price, past_price):
    if pd.isna(current_price) or pd.isna(past_price) or past_price == 0:
        return np.nan
    try:
        return ((float(current_price) - float(past_price)) / float(past_price)) * 100
    except (ValueError, TypeError):
        return np.nan


def fetch_fund_data(args):
    """TEFAS v0.6.0 API ile fon verisi çeker."""
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
        df = df.sort_values(by='date').reset_index(drop=True)
        fon_adi = df['title'].iloc[0] if 'title' in df.columns else fon_kodu
        return fon_kodu, fon_adi, df
    except Exception as e:
        print(f"  [UYARI] {fon_kodu}: Veri cekme hatasi - {e}")
        return fon_kodu, None, None


def hesapla_metrikler(df_fon_fiyat):
    """Risk/getiri metriklerini hesaplar."""
    if df_fon_fiyat is None or len(df_fon_fiyat) < 10:
        return None
    df_fon_fiyat['daily_return'] = df_fon_fiyat['price'].pct_change()
    df_fon_fiyat = df_fon_fiyat.dropna()
    if df_fon_fiyat.empty:
        return None

    getiri = (df_fon_fiyat['price'].iloc[-1] / df_fon_fiyat['price'].iloc[0]) - 1
    volatilite = df_fon_fiyat['daily_return'].std() * np.sqrt(252)
    ortalama_gunluk_getiri = df_fon_fiyat['daily_return'].mean()
    gunluk_std = df_fon_fiyat['daily_return'].std()
    sharpe = (ortalama_gunluk_getiri / gunluk_std) * np.sqrt(252) if gunluk_std != 0 else 0

    neg_getiriler = df_fon_fiyat[df_fon_fiyat['daily_return'] < 0]['daily_return']
    if neg_getiriler.empty or neg_getiriler.std() == 0:
        sortino = 0
    else:
        downside = neg_getiriler.std() * np.sqrt(252)
        sortino = (ortalama_gunluk_getiri * 252) / downside if downside != 0 else 0

    return {
        'Getiri (%)': round(getiri * 100, 2),
        'Standart Sapma (Yıllık %)': round(volatilite * 100, 2),
        'Sharpe Oranı (Yıllık)': round(sharpe, 2),
        'Sortino Oranı (Yıllık)': round(sortino, 2),
    }


def write_to_excel(haftalik_df, trend_df, fonaliz_df, filename=EXCEL_DOSYASI):
    """DataFrame'leri Excel dosyasına çoklu sayfa olarak yazar."""
    try:
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book

            # --- Sayfa 1: Haftalık Getiriler ---
            if haftalik_df is not None and not haftalik_df.empty:
                haftalik_kolonlar = ['Fon Kodu', 'Fon Adı', 'Hafta_1_Getiri', 'Hafta_2_Getiri', 'Toplam_Getiri']
                df_haftalik = haftalik_df[haftalik_kolonlar].sort_values('Toplam_Getiri', ascending=False, na_position='last')
                df_haftalik.to_excel(writer, sheet_name='Haftalık', index=False)
                ws = writer.sheets['Haftalık']
                for i, col in enumerate(df_haftalik.columns):
                    max_len = max(df_haftalik[col].astype(str).map(len).max(), len(col)) + 2
                    ws.set_column(i, i, max_len)
                print(f"  [OK] 'Haftalik' sayfasi: {len(df_haftalik)} fon")

            # --- Sayfa 2: Kısa Trend Analizi ---
            if trend_df is not None and not trend_df.empty:
                trend_kolonlar = ['Fon Kodu', 'Fon Adı', 'Trend_Yonu', 'Trend_Skoru',
                                   'Son_3G_Ort_%', 'Onceki_3G_Ort_%',
                                   'Hafta_1_Getiri', 'Hafta_2_Getiri', 'Toplam_Getiri']
                df_kisa_trend = trend_df[trend_kolonlar].copy()

                trend_sirasi = {'HIZLANAN': 0, 'YUKSELEN': 1, 'DONUS': 2, 'DUSUS': 3, 'DUSEN': 4, 'VERI_YOK': 5}
                df_kisa_trend['_sira'] = df_kisa_trend['Trend_Yonu'].map(trend_sirasi).fillna(5)
                df_kisa_trend = df_kisa_trend.sort_values(['_sira', 'Trend_Skoru'], ascending=[True, False]).drop(columns=['_sira'])

                trend_baslik = ['Fon Kodu', 'Fon Adı', 'Trend Yönü', 'Trend Skoru',
                                 'Son 3G Ort %', 'Önceki 3G Ort %',
                                 'Hafta 1 Getiri %', 'Hafta 2 Getiri %', 'Toplam Getiri %']
                df_kisa_trend.to_excel(writer, sheet_name='Kısa Trend', index=False, header=trend_baslik)
                ws = writer.sheets['Kısa Trend']
                for i, col in enumerate(df_kisa_trend.columns):
                    max_len = max(df_kisa_trend[col].astype(str).map(len).max(), len(trend_baslik[i])) + 2
                    ws.set_column(i, i, max_len)
                print(f"  [OK] 'Kisa Trend' sayfasi: {len(df_kisa_trend)} fon")

            # --- Sayfa 3: Fonaliz Risk Analizi ---
            if fonaliz_df is not None and not fonaliz_df.empty:
                fonaliz_kolonlar = ['Fon Kodu', 'Fon Adı', 'Sortino Oranı (Yıllık)',
                                     'Sharpe Oranı (Yıllık)', 'Getiri (%)', 'Standart Sapma (Yıllık %)']
                df_fonaliz = fonaliz_df[fonaliz_kolonlar].sort_values(
                    ['Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)'],
                    ascending=[False, False]
                )
                df_fonaliz.to_excel(writer, sheet_name='Fonaliz', index=False)
                ws = writer.sheets['Fonaliz']
                for i, col in enumerate(df_fonaliz.columns):
                    max_len = max(df_fonaliz[col].astype(str).map(len).max(), len(col)) + 2
                    ws.set_column(i, i, max_len)
                print(f"  [OK] 'Fonaliz' sayfasi: {len(df_fonaliz)} fon")

        print(f"\n[DOSYA] Excel olusturuldu: {filename}")
        return True
    except Exception as e:
        print(f"[HATA] Excel yazma: {e}")
        traceback.print_exc()
        return False


# --- TEKİL TARAMA ---
def run_single_scan(scan_date=None):
    """Belirli bir tarih için tekil tarama yapar ve Excel'e kaydeder."""
    if scan_date is None:
        scan_date = datetime.now().date() - timedelta(days=1)

    print(f"\n{'='*50}")
    print(f"[ASAMA 1] TEKIL TARAMA ({scan_date})")
    print(f"{'='*50}")

    fon_df = load_takasbank_fund_list()
    if fon_df.empty:
        return

    baslangic = scan_date - relativedelta(years=1, months=2) - timedelta(days=30)
    tasks = [(kod, baslangic, scan_date) for kod in fon_df['Fon Kodu'].unique()]
    sonuclar = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_fund_data, t): t[0] for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=" Tekil Tarama"):
            kod, ad, data = future.result()
            if data is None or data.empty:
                continue

            son_fiyat = get_value_on_or_before(data, scan_date)
            donemler = {
                'Günlük %': timedelta(days=1), 'Haftalık %': timedelta(weeks=1),
                '2 Haftalık %': timedelta(weeks=2), 'Aylık %': relativedelta(months=1),
                '3 Aylık %': relativedelta(months=3), '6 Aylık %': relativedelta(months=6),
                '1 Yıllık %': relativedelta(years=1)
            }
            satir = {'Fon Kodu': kod, 'Fon Adı': ad}
            if pd.notna(son_fiyat):
                for donem, delta in donemler.items():
                    onceki = get_value_on_or_before(data, scan_date - delta)
                    satir[donem] = calculate_change(son_fiyat, onceki)
            sonuclar.append(satir)

    if sonuclar:
        df = pd.DataFrame(sonuclar)
        df = df.sort_values('Haftalık %', ascending=False, na_position='last')
        # Tekil taramayı da aynı Excel'e ekleyelim
        with pd.ExcelWriter(EXCEL_DOSYASI, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Tekil Tarama', index=False)
            ws = writer.sheets['Tekil Tarama']
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                ws.set_column(i, i, max_len)
        print(f"[OK] Tekil tarama tamamlandi: {len(df)} fon -> {EXCEL_DOSYASI}")


# --- HAFTALIK TARAMA + TREND ANALİZİ ---
def run_weekly_scan(num_weeks=2):
    """
    Haftalık getiri analizi + 3 günlük trend analizi.
    v3.1: Excel çıktısı (Google Sheets kaldırıldı)
    """
    print(f"\n{'='*50}")
    print(f"[ASAMA 1] HAFTALIK TARAMA ({num_weeks} hafta) + TREND ANALIZI")
    print(f"{'='*50}")

    today = date.today()
    fon_df = load_takasbank_fund_list()
    if fon_df.empty:
        return [], pd.DataFrame()

    baslangic = today - timedelta(days=(num_weeks * 7) + 30)
    tasks = [(kod, baslangic, today) for kod in fon_df['Fon Kodu'].unique()]
    sonuclar = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_fund_data, t): t[0] for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=" Haftalık+Trend"):
            kod, ad, data = future.result()
            if data is None or data.empty or len(data) < 10:
                continue

            # Haftalık getiri
            fiyat_now = get_value_on_or_before(data, today)
            fiyat_1w = get_value_on_or_after(data, today - timedelta(weeks=1))
            fiyat_2w = get_value_on_or_after(data, today - timedelta(weeks=2))

            hafta2_getiri = calculate_change(fiyat_now, fiyat_1w)
            hafta1_getiri = calculate_change(fiyat_1w, fiyat_2w)
            toplam_getiri = (hafta1_getiri + hafta2_getiri) if pd.notna(hafta1_getiri) and pd.notna(hafta2_getiri) else np.nan

            # 3 günlük trend analizi
            son_3g_ort = np.nan
            onceki_3g_ort = np.nan
            trend_skoru = np.nan
            trend_yonu = "VERI_YOK"

            if len(data) >= TREND_GUN_SAYISI * 2 + 1:
                df_trend = data.copy()
                df_trend['daily_return'] = df_trend['price'].pct_change() * 100
                df_trend = df_trend.dropna().tail(TREND_GUN_SAYISI * 2)

                if len(df_trend) >= TREND_GUN_SAYISI * 2:
                    son_3g = df_trend.tail(TREND_GUN_SAYISI)['daily_return']
                    onceki_3g = df_trend.head(TREND_GUN_SAYISI)['daily_return']
                    son_3g_ort = son_3g.mean()
                    onceki_3g_ort = onceki_3g.mean()

                    if pd.notna(onceki_3g_ort) and onceki_3g_ort != 0:
                        trend_skoru = son_3g_ort / onceki_3g_ort
                    elif son_3g_ort > 0:
                        trend_skoru = son_3g_ort

                    if son_3g_ort > 0 and onceki_3g_ort > 0:
                        trend_yonu = "HIZLANAN" if son_3g_ort > onceki_3g_ort else "YUKSELEN"
                    elif son_3g_ort > 0 and onceki_3g_ort <= 0:
                        trend_yonu = "DONUS"
                    elif son_3g_ort <= 0 and onceki_3g_ort > 0:
                        trend_yonu = "DUSUS"
                    else:
                        trend_yonu = "DUSEN"

            sonuc = {
                'Fon Kodu': kod,
                'Fon Adı': ad,
                'Hafta_1_Getiri': round(hafta1_getiri, 2) if pd.notna(hafta1_getiri) else np.nan,
                'Hafta_2_Getiri': round(hafta2_getiri, 2) if pd.notna(hafta2_getiri) else np.nan,
                'Toplam_Getiri': round(toplam_getiri, 2) if pd.notna(toplam_getiri) else np.nan,
                'Son_3G_Ort_%': round(son_3g_ort, 4) if pd.notna(son_3g_ort) else np.nan,
                'Onceki_3G_Ort_%': round(onceki_3g_ort, 4) if pd.notna(onceki_3g_ort) else np.nan,
                'Trend_Skoru': round(trend_skoru, 4) if pd.notna(trend_skoru) else np.nan,
                'Trend_Yonu': trend_yonu,
            }
            sonuclar.append(sonuc)

    if not sonuclar:
        print("[HATA] Veri bulunamadi.")
        return [], pd.DataFrame()

    df_sonuc = pd.DataFrame(sonuclar)

    # Trend dağılımı
    print(f"\n[TREND] Dagitim:")
    for yon in ['HIZLANAN', 'YUKSELEN', 'DONUS', 'DUSUS', 'DUSEN', 'VERI_YOK']:
        sayi = len(df_sonuc[df_sonuc['Trend_Yonu'] == yon])
        if sayi > 0:
            print(f"  {yon}: {sayi} fon")

    # Filtreleme
    filtre_df = df_sonuc[
        (pd.to_numeric(df_sonuc['Toplam_Getiri'], errors='coerce').fillna(0) >= 2) &
        (df_sonuc['Trend_Yonu'].isin(['HIZLANAN', 'YUKSELEN', 'DONUS']))
    ]
    fonaliz_fonlar = filtre_df['Fon Kodu'].tolist()
    print(f"\n[FILTRE] Fonaliz icin secilen: {len(fonaliz_fonlar)} fon")

    return fonaliz_fonlar, df_sonuc


# --- FONALİZ RİSK ANALİZİ ---
def run_fonaliz(fon_listesi):
    """Filtrelenmiş fonlar için risk/getiri metriklerini hesaplar."""
    if not fon_listesi:
        print("[BILGI] Analiz icin fon bulunamadi.")
        return pd.DataFrame()

    print(f"\n{'='*50}")
    print(f"[ASAMA 2] FONALIZ RISK ANALIZI ({len(fon_listesi)} fon)")
    print(f"{'='*50}")

    today = date.today()
    baslangic = today - relativedelta(months=3) - timedelta(days=30)
    tasks = [(kod, baslangic, today) for kod in fon_listesi]
    sonuclar = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_fund_data, t): t[0] for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=" Fonaliz Analizi"):
            kod, ad, data = future.result()
            if data is None or data.empty:
                continue
            metrik = hesapla_metrikler(data)
            if metrik:
                sonuclar.append({'Fon Kodu': kod, 'Fon Adı': ad, **metrik})

    if sonuclar:
        df = pd.DataFrame(sonuclar)
        df = df.sort_values(['Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)'], ascending=[False, False])
        print(f"[OK] Fonaliz tamamlandi: {len(df)} fon analiz edildi.")
        return df
    else:
        print("[BILGI] Fonaliz icin veri bulunamadi.")
        return pd.DataFrame()


# --- ANA ÇALIŞTIRMA ---
if __name__ == '__main__':
    print("=" * 50)
    print("OtoFon v3.1 - Entegre Fon Tarama ve Analiz")
    print("=" * 50)

    scan_type = sys.argv[1].lower() if len(sys.argv) > 1 else 'weekly'

    if scan_type == 'single':
        tarih = None
        if len(sys.argv) > 2:
            try:
                tarih = datetime.strptime(sys.argv[2], '%Y-%m-%d').date()
            except ValueError:
                print("[HATA] Tarih formati YYYY-MM-DD olmali")
                sys.exit(1)
        run_single_scan(tarih)

    elif scan_type == 'weekly':
        hafta = int(sys.argv[2]) if len(sys.argv) > 2 else 2

        # AŞAMA 1: Haftalık tarama + trend analizi
        fonaliz_fonlar, df_sonuc = run_weekly_scan(hafta)

        # AŞAMA 2: Fonaliz risk analizi
        df_fonaliz = run_fonaliz(fonaliz_fonlar)

        # Excel'e yaz
        print(f"\n{'='*50}")
        print("[DOSYA] Excel raporu olusturuluyor...")
        print(f"{'='*50}")

        if df_sonuc is not None and not df_sonuc.empty:
            # Trend verilerini ayır
            haftalik_df = df_sonuc.copy()
            trend_df = df_sonuc.copy()

            # Fonaliz analizi yoksa boş DF
            if df_fonaliz is None or df_fonaliz.empty:
                df_fonaliz = pd.DataFrame()

            write_to_excel(haftalik_df, trend_df, df_fonaliz)

            # Ayrıca filtrelenmiş fon listesini txt olarak kaydet
            with open('filtrelenmis_fonlar.txt', 'w', encoding='utf-8') as f:
                for kod in fonaliz_fonlar:
                    f.write(f"{kod}\n")
            print(f"[OK] filtrelenmis_fonlar.txt kaydedildi ({len(fonaliz_fonlar)} fon)")

        print(f"\n{'='*50}")
        print(f"[OK] Tum islemler tamamlandi!")
        print(f"{'='*50}")
        print(f"[RAPOR] Excel: {EXCEL_DOSYASI}")
        print(f"  - Sayfa 1: 'Haftalik' - Haftalik getiriler")
        print(f"  - Sayfa 2: 'Kisa Trend' - 3 gunluk trend analizi")
        print(f"  - Sayfa 3: 'Fonaliz' - Risk/Getiri metrikleri")
        print(f"[SONUC] {len(fonaliz_fonlar)} fon yukselis trendinde ve filtreleri gecti.")

    else:
        print(f"[HATA] Gecersiz parametre: {scan_type}. 'weekly' veya 'single' kullanin.")
