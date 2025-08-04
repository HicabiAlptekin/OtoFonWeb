# Fonaliz - Otomatik Fon Analiz Arac
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tefas import Crawler
import time
import warnings
import concurrent.futures

warnings.filterwarnings('ignore')

# --- AYARLAR ---
ANALIZ_SURESI_AY = 3
MAX_WORKERS = 10
MANUAL_FON_KODLARI = [
    "BDS", "BIH", "HDA", "HEH", "HIM", "SPN", "IAU", "FS5", "ZBO", "KLH",
    "HDK", "HMG", "LLA", "KHT", "MGH", "EC2", "HGM", "BST", "BDY", "ZJL",
    "AC5", "YHB", "GOH", "NNF", "YHZ", "GO9", "KTI", "HKH", "KHA", "KHC",
    "RBH", "FNT", "ALC", "BHI", "DKH", "ELZ", "AOJ", "KTS", "KPC", "ICZ",
    "GBJ", "IVF", "RHS", "TAU", "DTL", "YHK", "ZPE", "AK3", "ADP", "OHB",
    "SAS", "TTE", "KST", "HVZ", "GBH", "KPU", "ZJV", "GLG", "GKV", "TKF",
    "MPS", "TZD", "RDF", "HVS", "RPI", "BTZ", "YCP", "FPH", "TLH", "GHS",
    "IHA", "IDH", "GTM", "RTH", "GAF", "FSG", "MAC", "BUY", "ASJ", "PPB",
    "PHE", "ICF", "GIE", "BID", "MMH", "HMS", "GAE", "YEF", "HBU", "AKU",
    "FYD", "YUB", "TIE", "YHS", "GSP", "IIH", "ENJ", "KPH", "VAY", "GZR",
    "GL1", "DDA", "YLE", "FUA", "IHK", "BIO", "DZE", "YDI", "OHK", "DXP",
    "RBN", "DLD", "AYA", "HKR", "ST1", "DHJ", "IFN", "IHT", "DAH", "BVM",
    "AEV", "KHJ", "OPI", "UPH", "YAS", "OPH", "TYH", "TLZ", "KYA", "ZLH",
    "AAV", "GMR", "IHZ", "DPT", "YLY", "NHY", "HKM", "ONE", "GIH", "THV",
    "BUL", "THT", "TI2", "TI3", "NSH", "PPM", "HAT", "NPH", "HRZ", "MUT",
    "MTH", "HFI", "IML", "PHI", "KPA", "IHP", "BIG", "ZHH", "KVT", "POS",
    "BFT", "BSH", "BHL", "RKH", "BHA", "BNH", "BRT", "BV1", "DOH", "DUH",
    "FRC", "GKG", "GNH", "GNS", "GRT", "GTH", "HIH", "HNC", "IDI", "IMB",
    "KHB", "KIH", "KOT", "KPF", "MCU", "MGB", "MHF", "MKA", "MTF", "MTK",
    "NLE", "NST", "OMG", "PAO", "PBN", "PGS", "PHK", "PMP", "PPH", "PTO",
    "PYR", "RHI", "RIA", "SBH", "SKO", "SNY", "SRL", "SSS", "THF", "YHI",
    "YMH", "YPR"
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
    except Exception:
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
                metrikler = hesapla_metrikler(data)
                if metrikler:
                    sonuc = {'Fon Kodu': fon_kodu, 'Fon Adı': fon_adi, **metrikler}
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
        'Sortino Oranı (Yıllık)', 'Sharpe Oranı (Yıllık)', 'Getiri (%)', 'Standart Sapma (Yıllık %)'
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