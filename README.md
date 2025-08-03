# Fonaliz - Otomatik Fon Analiz Aracı

Bu proje, TEFAS'ta işlem gören ve portföyünün en az %80'i hisse senetlerinden oluşan "Hisse Senedi Yoğun Fonları" otomatik olarak tespit eder ve bu fonlar için detaylı bir performans analizi yapar.

## Özellikler

- **Dinamik Fon Tespiti:** Her çalıştığında, varlık dağılımına göre en güncel hisse senedi yoğun fon listesini otomatik olarak oluşturur.
- **Dinamik Tarih Aralığı:** Analizi, çalıştırıldığı günden veya bir önceki iş gününden geriye doğru 3 aylık bir periyot için yapar.
- **Detaylı Performans Metrikleri:**
  - Sortino Oranı (Yıllık)
  - Sharpe Oranı (Yıllık)
  - Dönemsel Getiri (%)
  - Standart Sapma (Yıllık %)
  - Güncel Piyasa Değeri (TL)
  - Güncel Yatırımcı Sayısı
- **Paralel Veri Çekme:** Analiz sürecini hızlandırmak için birden çok fonun verisini eş zamanlı olarak çeker.
- **Okunaklı Excel Çıktısı:** Analiz sonuçlarını, sütunları otomatik olarak boyutlandırılmış, kolay okunabilir bir Excel dosyasına kaydeder.

## Kurulum

1.  Proje dosyalarını bilgisayarınıza klonlayın:
    ```bash
    git clone https://github.com/HicabiAlptekin/Fonaliz.git
    cd Fonaliz
    ```

2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım

Proje klasöründeyken aşağıdaki komutu çalıştırın:

```bash
python analiz_script.py
```

Script çalıştığında, analiz sonuçlarını içeren `Hisse_Senedi_Fon_Analizi_YYYY-AA-GG.xlsx` adında bir Excel dosyası oluşturacaktır.