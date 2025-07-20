import os
import sys # sys modülünü ekleyin

def scan_directory(directory_path):
    try:
        print(f"Scanning directory: {directory_path}\n")
        # Eğer taranacak dizin yoksa veya geçersizse hata yakalamak için kontrol edebiliriz
        if not os.path.isdir(directory_path):
            print(f"Error: The specified directory does not exist or is not a directory: {directory_path}")
            sys.exit(1) # Hata kodu ile çıkış yap

        for root, dirs, files in os.walk(directory_path):
            print(f"Current Directory: {root}")
            print(f"Subdirectories: {dirs}")
            print(f"Files: {files}\n")
    except Exception as e:
        print(f"An error occurred during scan: {e}")
        sys.exit(1) # Hata kodu ile çıkış yap

if __name__ == "__main__":
    # Komut satırı argümanlarını kontrol edin
    if len(sys.argv) > 1: # Eğer argüman olarak bir yol verilmişse
        directory_to_scan = sys.argv[1] # İlk argümanı al
        scan_directory(directory_to_scan)
    else:
        # Argüman verilmezse GitHub Actions'ın varsayılan çalışma dizinini tara
        # Bu genellikle deponun kök dizinidir.
        github_workspace = os.getenv('GITHUB_WORKSPACE')
        if github_workspace:
            print(f"No specific directory provided. Scanning GITHUB_WORKSPACE: {github_workspace}")
            scan_directory(github_workspace)
        else:
            print("Usage: python tarama_script.py <directory_path>")
            print("No directory path provided and GITHUB_WORKSPACE not found. Exiting.")
            sys.exit(1) # Argüman veya ortam değişkeni yoksa çıkış yap
