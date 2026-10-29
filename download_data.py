import os
import urllib.request
import tarfile

URL = "https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
TAR_PATH = "flower_data.tar.gz"
DATA_DIR = "flower_data"


def setup_data():
    if os.path.exists(DATA_DIR):
        print(f"[INFO] '{DATA_DIR}' klasörü zaten mevcut.")
        return

    if not os.path.exists(TAR_PATH):
        print("[INFO] Veri seti indiriliyor (~330 MB)...")
        urllib.request.urlretrieve(URL, TAR_PATH)
        print("[INFO] İndirme tamamlandı.")

    print("[INFO] Dosyalar çıkartılıyor...")
    os.makedirs(DATA_DIR, exist_ok=True)
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print(f"[INFO] Veri seti '{DATA_DIR}' içine başarıyla çıkartıldı!")


if __name__ == "__main__":
    setup_data()

