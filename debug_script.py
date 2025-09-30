

import pandas as pd

FILE_PATH = r'C:\Users\autan\Downloads\Fon_Tarama_Sonuclari_20250907_072952.xlsx'

try:
    df = pd.read_excel(FILE_PATH)
    print("--- Dosyanın İlk 10 Satırı ---")
    print(df.head(10).to_string())
    print("\n--- Sütun Bilgileri ve Veri Tipleri ---")
    df.info()
except FileNotFoundError:
    print(f"HATA: Dosya bulunamadı -> {FILE_PATH}")
except Exception as e:
    print(f"Bir hata oluştu: {e}")

