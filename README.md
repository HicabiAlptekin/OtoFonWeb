# OtoFon - Otomatik Fon Tarama ve Analiz Sistemi

Bu proje, TEFAS'ta işlem gören yatırım fonlarını belirli kriterlere göre tarayan, filtreleyen ve seçilen fonlar için detaylı bir performans analizi yapan otomatik bir sistemdir. Sistem, GitHub Actions kullanılarak haftanın her iş günü otomatik olarak çalışacak şekilde tasarlanmıştır.

## Sistemin Çalışma Prensibi

Sistem iki ana script'ten oluşur ve bir GitHub Actions workflow'u ile yönetilir:

1.  **`tarama_script.py` (Fon Tarama):**
    *   Tüm TEFAS fonlarını listeler.
    *   Belirlenen haftalık periyot için fonların getirilerini hesaplar.
    *   Son iki haftalık toplam getirisi belirli bir eşiğin (örneğin %2) üzerinde olan fonları belirler.
    *   Bu kriterlere uyan fonların kodlarını `filtrelenmis_fonlar.txt` dosyasına kaydeder.

2.  **`analiz_script.py` (Detaylı Analiz):**
    *   `filtrelenmis_fonlar.txt` dosyasını okur.
    *   Listelenen her bir fon için son 3 aylık verileri çekerek detaylı bir performans analizi yapar.
    *   Hesaplanan metrikler şunlardır:
        - Yıllıklandırılmış Sortino ve Sharpe Oranları
        - Dönemsel Getiri (%)
        - Yıllıklandırılmış Standart Sapma (%)
        - Güncel Piyasa Değeri ve Yatırımcı Sayısı
    *   Sonuçları, tarih damgasıyla birlikte okunaklı bir Excel dosyasına (`Hisse_Senedi_Fon_Analizi_YYYY-AA-GG.xlsx`) yazar.

## Otomasyon (GitHub Actions)

Proje, `.github/workflows/main.yml` dosyasında tanımlanan bir iş akışı sayesinde otomatik olarak çalışır:
- **Zamanlama:** Her iş günü (Pazartesi-Cuma) sabah saat 09:00'da (TSI) otomatik olarak tetiklenir.
- **Manuel Tetikleme:** GitHub Actions arayüzünden manuel olarak da çalıştırılabilir.
- **Artifacts (Çıktılar):** Her çalıştırmanın sonunda oluşturulan Excel ve metin dosyaları, "analiz-sonuclari" adıyla bir artifact olarak kaydedilir ve GitHub arayüzünden indirilebilir.

## Kurulum (Lokal Çalıştırma İçin)

1.  Projeyi klonlayın:
    ```bash
    git clone https://github.com/HicabiAlptekin/OtoFon.git
    cd OtoFon
    ```

2.  Gerekli Python kütüphanelerini yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım (Lokal Çalıştırma İçin)

Scriptleri sırasıyla çalıştırın:

1.  Önce tarama script'ini çalıştırarak fon listesini oluşturun:
    ```bash
    python tarama_script.py
    ```

2.  Ardından analiz script'ini çalıştırın:
    ```bash
    python analiz_script.py
    ```

Analiz tamamlandığında, sonuçları içeren `.xlsx` uzantılı Excel dosyası proje ana dizininde oluşturulacaktır.
