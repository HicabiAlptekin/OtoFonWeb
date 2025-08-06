# Fonaliz - Otomatik Fon Analiz Araci
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from tefas import Crawler
import time
import warnings
import concurrent.futures
import sys
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# --- AYARLAR ---
ANALIZ_SURESI_AY = 3
MAX_WORKERS = 10
TAKASBANK_EXCEL_URL = 'https://www.takasbank.com.tr/plugins/ExcelExportTefasFundsTradingInvestmentPlatform?language=tr'
TIMEZONE = pytz.timezone('Europe/Istanbul') # Gerekli import eklendi

# --- OTOFON'DAN ALINAN YARDIMCI FONKSİYONLAR ---

def load_takasbank_fund_list():
    """Takasbank'tan güncel tüm fonların listesini çeker."""
    print(f"Takasbank'tan güncel fon listesi yükleniyor...")
    try:
        df_excel = pd.read_excel(TAKASBANK_EXCEL_URL, engine='openpyxl')
        df_data = df_excel[['Fon Kodu']].copy()
        df_data['Fon Kodu'] = df_data['Fon Kodu'].astype(str).str.strip().str.upper()
        df_data.dropna(subset=['Fon Kodu'], inplace=True)
        df_data = df_data[df_data['Fon Kodu'] != '']
        print(f"✅ {len(df_data)} adet fon bilgisi okundu.")
        return df_data['Fon Kodu'].unique().tolist()
    except Exception as e:
        print(f"❌ Takasbank Excel yükleme hatası: {e}")
        return []

def get_price_on_or_before(df_fund_history, target_date: date):
    """Belirtilen tarihteki veya o tarihten önceki en son fiyatı bulur."""
    if df_fund_history is None or df_fund_history.empty or target_date is None: return np.nan
    df_filtered = df_fund_history[df_fund_history['date'] <= target_date].copy()
    if not df_filtered.empty: return df_filtered.sort_values(by='date', ascending=False)['price'].iloc[0]
    return np.nan

def calculate_change(current_price, past_price):
    """İki fiyat arasındaki yüzdesel değişimi hesaplar."""
    if pd.isna(current_price) or pd.isna(past_price) or past_price is None or current_price is None: return np.nan
    try:
        current_price_float, past_price_float = float(current_price), float(past_price)
        if past_price_float == 0: return np.nan
        return ((current_price_float - past_price_float) / past_price_float) * 100
    except (ValueError, TypeError): return np.nan

def fetch_history_for_filter(fon_kodu):
    """Filtreleme için bir fonun geçmiş verisini çeker."""
    try:
        crawler = Crawler()
        end_date = datetime.now(TIMEZONE).date()
        start_date = end_date - timedelta(days=60) # Trend analizi için yaklaşık 2 aylık veri yeterli
        df = crawler.fetch(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), name=fon_kodu, columns=["date", "price"])
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.date
        return fon_kodu, df
    except Exception:
        return fon_kodu, pd.DataFrame()

def get_otofon_trend_fonlari(num_weeks: int = 2):
    """
    OtoFon'daki haftalık getirisi düşüş trendinde olan fonları belirler ve
    bu fonların kodlarını bir liste olarak döndürür.
    """
    print(f"\n--- OtoFon Trend Filtrelemesi Başlatılıyor ({num_weeks} Hafta) ---")
    all_fon_kodlari = load_takasbank_fund_list()
    if not all_fon_kodlari:
        return []

    trend_fonlari = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(fetch_history_for_filter, fon_kodu): fon_kodu for fon_kodu in all_fon_kodlari}
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_fon), total=len(all_fon_kodlari), desc="Trend Fonları Taranıyor")

        for future in progress_bar:
            fon_kodu_completed, fund_history = future.result()
            if fund_history.empty:
                continue

            try:
                weekly_changes_list = []
                current_week_end_date_cal = datetime.now(TIMEZONE).date()

                for i in range(num_weeks):
                    current_week_start_date_cal = current_week_end_date_cal - timedelta(days=7)
                    price_end = get_price_on_or_before(fund_history, current_week_end_date_cal)
                    price_start = get_price_on_or_before(fund_history, current_week_start_date_cal)
                    weekly_change = calculate_change(price_end, price_start)
                    weekly_changes_list.append(weekly_change)
                    current_week_end_date_cal = current_week_start_date_cal

                valid_changes = [chg for chg in weekly_changes_list if not pd.isna(chg)]
                if len(valid_changes) == num_weeks and num_weeks >= 2:
                    # OtoFon'daki 'is_desired_trend' mantığı: Haftalık getiriler düşüş trendinde mi?
                    if all(valid_changes[j] > valid_changes[j+1] for j in range(num_weeks - 1)):
                        trend_fonlari.append(fon_kodu_completed)
            except Exception:
                continue # Hata durumunda bu fonu atla

    print(f"--- Trend Filtrelemesi Tamamlandı. {len(trend_fonlari)} adet uygun fon bulundu. ---")
    return trend_fonlari


# --- Fonaliz - ANA ANALİZ FONKSİYONLARI ---

def get_last_business_day():
    today = datetime.now()
    for i in range(7):
        day_to_check = today - timedelta(days=i)
        if day_to_check.weekday() < 5:
            return day_to_check
    return today - timedelta(days=1)

def get_fon_verileri_parallel(args):
    fon_kodu, start_date, end_date = args
    print(f"'{fon_kodu}' için detaylı analiz verisi çekiliyor...")
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

def calistir_analiz():
    """Fon analizini çalıştırır ve sonuçları bir DataFrame olarak döndürür."""
    # 1. ADIM: OtoFon trend mantığı ile fonları filtrele
    fon_kodlari_listesi = get_otofon_trend_fonlari(num_weeks=2)

    if not fon_kodlari_listesi:
        print("\nAnaliz edilecek kriterlere uygun fon bulunamadı. İşlem sonlandırılıyor.")
        return None, None

    # 2. ADIM: Filtrelenmiş liste ile detaylı analizi çalıştır
    print(f"\n--- FİLTRELENMİŞ LİSTE İLE DETAYLI ANALİZ BAŞLATILDI ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    end_date = get_last_business_day()
    start_date = end_date - pd.DateOffset(months=ANALIZ_SURESI_AY)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    print(f"Analiz Tarih Aralığı: {start_date_str} -> {end_date_str}")

    tasks = [(fon_kodu, start_date_str, end_date_str) for fon_kodu in fon_kodlari_listesi]
    analiz_sonuclari = []

    print(f"\n{len(tasks)} adet fon için detaylı veriler paralel olarak çekiliyor...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_fon = {executor.submit(get_fon_verileri_parallel, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(future_to_fon):
            fon_kodu, fon_adi, data = future.result()
            if data is not None:
                correlation_results, sentiment_score = analyze_daily_correlations(data, fon_kodu)
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
        excel_dosya_adi = f"Hisse_Senedi_Fon_Analizi_{end_date_str}.xlsx"
        with pd.ExcelWriter(excel_dosya_adi, engine='xlsxwriter') as writer:
            df_sonuc_sirali.to_excel(writer, sheet_name='Fon Analizi', index=False)
            worksheet = writer.sheets['Fon Analizi']
            for i, col in enumerate(df_sonuc_sirali.columns):
                column_len = max(df_sonuc_sirali[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, column_len)
        print(f"\nAnaliz sonuçları '{excel_dosya_adi}' dosyasına kaydedildi.")

if __name__ == '__main__':
    # Gerekli import'u main bloğuna taşıdık
    import pytz
    main()
