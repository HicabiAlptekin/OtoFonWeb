import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web

# Analiz edilecek piyasaların ve sembollerinin tanımlandığı sözlük
markets = {
    "Altın (XAUUSD)": "GC=F",
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "DXY (Dolar Endeksi)": "DX-Y.NYB",
    "Ham Petrol (WTI)": "CL=F",
    "10Y Tahvil Faizi": "^TNX",
    "Bitcoin": "BTC-USD"
}

def get_user_selection():
    """
    Kullanıcıya mevcut piyasaları listeler ve bir seçim yapmasını ister.
    Geçerli bir seçim yapılana kadar sormaya devam eder.
    """
    print("--- M2V ile Karşılaştırmak İçin Bir Piyasa Seçin ---")
    
    # Seçenekleri numaralandırarak listele
    market_list = list(markets.keys())
    for i, market_name in enumerate(market_list):
        print(f"{i + 1}) {market_name}")
    
    while True:
        try:
            # Kullanıcıdan sayısal bir girdi al
            choice = int(input("\nLütfen seçiminizi yapın (sayı girin): "))
            # Girdinin geçerli bir aralıkta olup olmadığını kontrol et
            if 1 <= choice <= len(market_list):
                # Geçerli ise seçilen piyasanın adını döndür
                return market_list[choice - 1]
            else:
                print(f"Hata: Lütfen 1 ile {len(market_list)} arasında bir sayı girin.")
        except ValueError:
            # Eğer kullanıcı sayı dışında bir şey girerse hata ver
            print("Hata: Lütfen geçerli bir sayı girin.")

def plot_comparison(selected_market_name):
    """
    M2 Velocity (sabit sol eksende) ile kullanıcının seçtiği tek bir piyasayı (sağ eksende) karşılaştırır.
    ABD resesyon bölgelerini de grafikte gösterir.
    """
    # 1) FRED verilerini indir (M2V ve Resesyon verisi)
    start_date = "1960-01-01"
    end_date = "2025-08-01"
    m2v = web.DataReader("M2V", "fred", start=start_date, end=end_date)
    usrec = web.DataReader("USREC", "fred", start=start_date, end=end_date)

    # 2) Kullanıcının seçtiği piyasa verisini Yahoo Finance'ten indir
    ticker = markets[selected_market_name]
    # Aylık veriyi çek
    data = yf.download(ticker, start=start_date, end=end_date, interval="1mo")["Close"]
    # M2V ile uyumlu olması için çeyreklik ortalamaya dönüştür
    market_data_q = data.resample("QE").mean()
    market_data_q.name = selected_market_name # Seriye isim ver

    # 3) Tüm verileri tek bir DataFrame'de birleştir
    df = pd.concat([m2v, market_data_q, usrec], axis=1)
    df.columns = ["M2V", selected_market_name, "USREC"]
    df = df.ffill() # Eksik verileri önceki veriyle doldur
    df = df.dropna(subset=["M2V"]) # M2V verisi olmayan satırları kaldır

    # 4) Grafiği oluştur
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Sol eksen: M2 Velocity
    ax1.set_xlabel("Tarih")
    ax1.set_ylabel("M2 Velocity", color="tab:blue")
    ax1.plot(df.index, df["M2V"], color="tab:blue", linewidth=2, label="M2 Velocity")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Sağ eksen: Seçilen piyasa
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"{selected_market_name} Değeri", color="tab:red")
    ax2.plot(df.index, df[selected_market_name], color="tab:red", linewidth=2, label=selected_market_name)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Resesyon bölgelerini gri ile gölgelendir
    for i in range(len(df) - 1):
        if df["USREC"].iloc[i] == 1:
            ax1.axvspan(df.index[i], df.index[i+1], color="gray", alpha=0.3)

    # Başlık ve Lejant
    plt.title(f"M2 Para Hızı (M2V) ve {selected_market_name}\n(ABD Resesyon Bölgeleri ile Birlikte)", fontsize=14)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    fig.tight_layout()
    plt.show()


# --- Ana Kod Bloğu ---
if __name__ == "__main__":
    # Kullanıcıdan karşılaştırılacak piyasayı seçmesini iste
    selected_market = get_user_selection()
    
    # Seçilen piyasaya göre grafiği oluştur
    plot_comparison(selected_market)
