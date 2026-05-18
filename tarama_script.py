# -*- coding: utf-8 -*-
# ENTEGRE FON TARAMA ARACI (OtoFon + Fonaliz - GitHub Actions için düzenlendi)
# Bu script, haftalık fon getirilerini tarar, belirli bir filtreden geçirir
# ve sonucu 'filtrelenmis_fonlar.txt' dosyasına yazar.
# v3.0 - Haftalık getiri hesaplama düzeltmesi + 3 iş günü trend analizi

import pandas as pd
import numpy as np
import time
import sys
from datetime import timedelta, date
from tefas import Crawler
import concurrent.futures
# Global TEFAS crawler (tekil örnek - performans için)
try:
    tefas_crawler_global = Crawler()
except Exception:
    tefas_crawler_global = None

# --- Sabitler ve Yapılandırma ---
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
MAX_WORKERS = 10
TREND_GUN_SAYISI = 3  # Trend analizi için iş günü sayısı

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

def get_value_on_or_before(df_fund_history, target_date, column='price'):
    """Belirtilen tarihteki veya o tarihten önceki en yakın değeri döndürür."""
    if df_fund_history is None or df_fund_history.empty or target_date is None:
        return np.nan
    df_filtered = df_fund_history[df_fund_history['date'] <= target_date].copy()
    if not df_filtered.empty:
        return df_filtered.sort_values(by='date', ascending=False)[column].iloc[0]
    return np.nan

def get_value_on_or_after(df_fund_history, target_date, column='price'):
    """Belirtilen tarihteki veya o tarihten sonraki en yakın değeri döndürür."""
    if df_fund_history is None or df_fund_history.empty or target_date is None:
        return np.nan
    df_filtered = df_fund_history[df_fund_history['date'] >= target_date].copy()
    if not df_filtered.empty:
        return df_filtered.sort_values(by='date', ascending=True)[column].iloc[0]
    return np.nan

def calculate_change(current_price, past_price):
    if pd.isna(current_price) or pd.isna(past_price):
        return np.nan
    try:
        current_price_float, past_price_float = float(current_price), float(past_price)
        if past_price_float == 0:
            return np.nan
        return ((current_price_float - past_price_float) / past_price_float) * 100
    except (ValueError, TypeError):
        return np.nan

def fetch_data_for_fund_parallel(args):
    """TEFAS v0.6.0 API ile fon verisi çeker."""
    fon_kodu, start_date, end_date = args
    global tefas_crawler_global
    if tefas_crawler_global is None:
        return fon_kodu, None
    try:
        df = tefas_crawler_global.fetch(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            name=fon_kodu,
            columns=["date", "price"]
        )
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            df = df.dropna(subset=['date'])
            return fon_kodu, df.sort_values(by='date').reset_index(drop=True)
    except Exception:
        return fon_kodu, None
    return fon_kodu, None


def run_weekly_scan(num_weeks: int):
    """
    Gelişmiş fon tarama motoru.
    
    Değişiklikler (v3.0):
    1. Haftalık getiri hesaplama düzeltmesi: Her hafta için ayrı bitiş fiyatı kullanılır.
       Eskiden tüm haftalar için aynı (bugünkü) bitiş fiyatı kullanılıyordu, bu da
       Hafta_1'in aslında 2 haftalık toplam getiri gibi görünmesine yol açıyordu.
    2. Yeni: Son 3 iş günü trend analizi - kısa vadeli yükseliş trendi tespiti.
    """
    start_time_main = time.time()
    today = date.today()
    all_fon_data_df = load_takasbank_fund_list()

    if all_fon_data_df.empty:
        print("Taranacak fon listesi alınamadı. İşlem durduruldu.")
        return pd.DataFrame()

    print(f"\n{num_weeks} Haftalık Tarama Başlatılıyor...")
    print(f"Trend analizi: Son {TREND_GUN_SAYISI} iş günü vs önceki {TREND_GUN_SAYISI} iş günü")

    # Veri çekme başlangıcı: yeterli buffer ile
    genel_veri_cekme_baslangic_tarihi = today - timedelta(days=(num_weeks * 7) + 30)
    tasks = [(fon_kodu, genel_veri_cekme_baslangic_tarihi, today)
             for fon_kodu in all_fon_data_df['Fon Kodu'].unique()]

    weekly_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_data_for_fund_parallel, args): args[0] for args in tasks}

        for future in concurrent.futures.as_completed(future_to_fon):
            fon_kodu, fund_history = future.result()
            if fund_history is None or fund_history.empty:
                continue

            df = fund_history
            n = len(df)
            
            # ----- 1. HAFTALIK GETİRİ HESAPLAMA (DÜZELTİLDİ) -----
            # Her hafta için ayrı bitiş fiyatı kullan
            weekly_changes = []
            
            # Hafta_2 (en yeni): bugünkü fiyat / 1 hafta önceki fiyat
            price_now = get_value_on_or_before(df, today)
            price_1w_ago = get_value_on_or_after(df, today - timedelta(weeks=1))
            change_w2 = calculate_change(price_now, price_1w_ago)
            
            # Hafta_1 (eski): 1 hafta önceki fiyat / 2 hafta önceki fiyat
            price_2w_ago = get_value_on_or_after(df, today - timedelta(weeks=2))
            change_w1 = calculate_change(price_1w_ago, price_2w_ago)
            
            weekly_changes = [change_w1, change_w2]
            
            # ----- 2. SON 3 İŞ GÜNÜ TREND ANALİZİ (YENİ) -----
            trend_skoru = np.nan
            trend_yonu = "BELIRSIZ"
            son_3g_ort_getiri = np.nan
            onceki_3g_ort_getiri = np.nan
            
            if n >= TREND_GUN_SAYISI * 2 + 1:
                # Günlük getirileri hesapla
                df_trend = df.copy()
                df_trend['daily_return'] = df_trend['price'].pct_change() * 100
                df_trend = df_trend.dropna().tail(TREND_GUN_SAYISI * 2)
                
                if len(df_trend) >= TREND_GUN_SAYISI * 2:
                    # Son 3 iş günü
                    son_3g = df_trend.tail(TREND_GUN_SAYISI)['daily_return']
                    son_3g_ort_getiri = son_3g.mean()
                    
                    # Önceki 3 iş günü
                    onceki_3g = df_trend.head(TREND_GUN_SAYISI)['daily_return']
                    onceki_3g_ort_getiri = onceki_3g.mean()
                    
                    # Trend skoru: son 3g ortalaması / onceki 3g ortalaması
                    if pd.notna(onceki_3g_ort_getiri) and onceki_3g_ort_getiri != 0:
                        trend_skoru = son_3g_ort_getiri / onceki_3g_ort_getiri
                    elif son_3g_ort_getiri > 0:
                        trend_skoru = son_3g_ort_getiri  # Önceki sıfır, son pozitif → skor = son 3g ortalaması
                    
                    # Trend yönü (gelişmiş sınıflandırma)
                    if son_3g_ort_getiri > 0 and onceki_3g_ort_getiri > 0:
                        if son_3g_ort_getiri > onceki_3g_ort_getiri:
                            trend_yonu = "HIZLANAN"
                        else:
                            trend_yonu = "YUKSELEN"
                    elif son_3g_ort_getiri > 0 and onceki_3g_ort_getiri <= 0:
                        trend_yonu = "DONUS"
                    elif son_3g_ort_getiri <= 0 and onceki_3g_ort_getiri > 0:
                        trend_yonu = "DUSUS"
                    else:
                        trend_yonu = "DUSEN"

            if len(weekly_changes) == num_weeks and all(pd.notna(c) for c in weekly_changes):
                result = {
                    'Fon Kodu': fon_kodu,
                    'Hafta_1_Getiri': change_w1,
                    'Hafta_2_Getiri': change_w2,
                    'Toplam_Getiri': (change_w1 + change_w2) if pd.notna(change_w1) and pd.notna(change_w2) else np.nan,
                    'Son_3G_Ort_Getiri': round(son_3g_ort_getiri, 4) if pd.notna(son_3g_ort_getiri) else np.nan,
                    'Onceki_3G_Ort_Getiri': round(onceki_3g_ort_getiri, 4) if pd.notna(onceki_3g_ort_getiri) else np.nan,
                    'Trend_Skoru': round(trend_skoru, 4) if pd.notna(trend_skoru) else np.nan,
                    'Trend_Yonu': trend_yonu,
                }
                weekly_results.append(result)

    results_df = pd.DataFrame(weekly_results)
    print(f"Haftalık tarama tamamlandı. Toplam Süre: {time.time() - start_time_main:.2f} saniye")
    return results_df


# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    print("--- Tarama Script'i Başlatıldı (v3.0) ---")

    try:
        num_weeks_arg = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1].lower() == 'weekly' else 2
    except (ValueError, IndexError):
        num_weeks_arg = 2

    haftalik_sonuclar_df = run_weekly_scan(num_weeks=num_weeks_arg)

    if not haftalik_sonuclar_df.empty:
        print(f"\n{'='*60}")
        print("FİLTRELEME AŞAMASI")
        print(f"{'='*60}")
        
        print("\nFİLTRE 1: Son 2 haftanın toplam getirisi >= %2")
        # FİLTRE 1: Haftalık toplam getiri >= %2
        filtre1_df = haftalik_sonuclar_df[
            haftalik_sonuclar_df['Toplam_Getiri'].fillna(0) >= 2
        ].copy()
        print("  -> {} fon filtreyi geçti".format(len(filtre1_df)))

        print("\nFİLTRE 2: Son 3 iş günü yükseliş trendinde (HIZLANAN, YUKSELEN veya DONUS)")
        # FİLTRE 2: Trend analizi - yükselen trend veya dönüş
        filtre2_df = filtre1_df[
            filtre1_df['Trend_Yonu'].isin(['HIZLANAN', 'YUKSELEN', 'DONUS'])
        ].copy()
        print("  -> {} fon filtreyi geçti".format(len(filtre2_df)))

        # Ortak filtre
        print("\nFİLTRE 3 (KOMBİNE): Filtre 1 + Filtre 2 (tavsiye edilen)")
        filtre3_df = filtre2_df.copy()

        if not filtre3_df.empty:
            print("  -> {} fon tüm filtreleri geçti".format(len(filtre3_df)))
            
            # Sıralama: Trend skoru yüksek olanlar önce
            filtre3_df = filtre3_df.sort_values(
                by=['Trend_Skoru', 'Toplam_Getiri'],
                ascending=[False, False]
            )
            
            # Özet göster
            print(f"\n{'='*60}")
            print("EN YÜKSELEN TRENDDEKİ FONLAR (İLK 20)")
            print(f"{'='*60}")
            for i, (_, row) in enumerate(filtre3_df.head(20).iterrows()):
                print(f"{i+1:2d}. {row['Fon Kodu']:<6} | "
                      f"Hafta1: %{row['Hafta_1_Getiri']:>+6.2f} | "
                      f"Hafta2: %{row['Hafta_2_Getiri']:>+6.2f} | "
                      f"Toplam: %{row['Toplam_Getiri']:>+6.2f} | "
                      f"Trend: {row['Trend_Yonu']:<8} | "
                      f"Skor: {row['Trend_Skoru']:>7.2f}")
            
            # Filtrelenmiş fonları dosyaya yaz
            filtrelenmis_fon_listesi = filtre3_df['Fon Kodu'].tolist()
            print(f"\n{len(filtrelenmis_fon_listesi)} fon Fonaliz için seçildi.")
            try:
                with open('filtrelenmis_fonlar.txt', 'w', encoding='utf-8') as f:
                    for fon_kodu in filtrelenmis_fon_listesi:
                        f.write(f"{fon_kodu}\n")
                print("'filtrelenmis_fonlar.txt' dosyasına yazıldı.")
            except Exception as e:
                print(f"Hata: Dosyaya yazılırken sorun: {e}")
        else:
            print("\nFiltreleri geçen fon bulunamadı.")
            open('filtrelenmis_fonlar.txt', 'w').close()
            print("'filtrelenmis_fonlar.txt' dosyası boş olarak oluşturuldu.")
    else:
        print("Haftalık tarama sonucu boş.")
        open('filtrelenmis_fonlar.txt', 'w').close()
        print("'filtrelenmis_fonlar.txt' dosyası boş olarak oluşturuldu.")

    print("\n--- Tarama Script'i Tamamlandı ---")
