# 102 Sınıflı Çiçek Tanıma Projesi (PyTorch & TorchScript)

**Oxford 102 Flowers** ve **Kaggle Flower Classification** veri setlerinin 102 sınıf standardında birleştirilmesiyle eğitilmiş, düşük gecikmeli ve kompakt **EfficientNetV2-B0 (`tf_efficientnetv2_b0.in1k`)** derin öğrenme projesi.

---

## Performans Özeti

Resmi değerlendirme betiği (`evaluate.py`) üzerinde elde edilen sonuçlar:

| Kriter | Sonuç | Yarışma Hedefi | Puan | Ağırlık | Katkı |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test Doğruluğu (`Acc`)** | **`%99.63`** *(816 / 819)* | Maksimum | `99.63` | `%60` | `59.78` |
| **CPU Çıkarım Hızı (`1 Thread`)** | **`31.48 ms`** | `<= 50 ms` | `100.0` | `%25` | `25.00` |
| **TorchScript Model Boyutu** | **`22.79 MB`** | `<= 35 MB` | `100.0` | `%15` | `15.00` |
| **YARIŞMA FİNAL PUANI** | — | — | — | `%100` | **`99.78 / 100`** |

---

## Proje Dosya Yapısı

```text
flower_project/
├── inference.py                     # Saf PyTorch (timm bağımsız) tekil görsel tahmin betiği
├── evaluate.py                      # TorchScript dışa aktarım ve resmi yarışma puanlama betiği
├── train.py                         # Oxford 102 + Kaggle dengeli eğitim ve Model Soup betiği
├── download_data.py                 # Oxford 102 (flower_data) veri setini otomatik indirme betiği
├── cat_to_name.json                 # Sınıf ID (1-102) -> Çiçek ismi eşleştirme sözlüğü
├── RAPOR.pdf                        # Detaylı teknik ve performans raporu
└── runs/
    └── effv2b0/
        ├── model_submission.pt      # Teslime hazır optimize TorchScript modeli (22.79 MB)
        ├── best_weights.pth         # 3'lü Model Soup PyTorch ağırlık dosyası
        └── class_map.json           # Model indekslerini (0-101) sınıf ID'lerine (1-102) bağlayan harita
```

---

## Kurulum

Sanal ortam oluşturup aktif edin (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Gerekli kütüphaneleri yükleyin:

```powershell
pip install torch torchvision timm pillow tqdm
```

> **Not:** `inference.py` betiği TorchScript (`model_submission.pt`) kullandığı için `timm` kütüphanesi kurulu olmayan ortamlarda da yalnızca `torch`, `torchvision` ve `pillow` ile doğrudan çalışır.

---

## Kullanım

### 1. Tekil Görsel Tahmini (`inference.py`)

Herhangi bir çiçek fotoğrafı (dosya yolu, Masaüstü dosya adı veya doğrudan internet URL'si) ile tahmin yapmak için:

```powershell
python inference.py --image ornek_cicek.jpg
```

veya doğrudan:

```powershell
python inference.py "C:\Users\kullanici\Desktop\gul.jpg"
```

#### Tahmin Betiğinin Öne Çıkan Özellikleri

- **Çoklu Ölçek + Buket Yakınlaştırma & Aynalama (TTA):** Hem tekil yakın çekimleri (`256–512`) hem de sepet/buket/uzak çekimleri (`640–800`) tarayarak yatay aynalama ve ölçekler arası mutabakat ile en doğru bölgeyi seçer.
- **3 Kademeli Güven Analizi:**
  - `>= %50`: Yüksek güvenli doğrudan sınıf tahmini.
  - `%25 – %50`: Zor açı veya **102 sınıf dışındaki benzer çiçek türü** bilgilendirmesi.
  - `< %25`: Çiçek dışı nesne veya 102 sınıf dışı düşük güven uyarısı.

---

### 2. Resmi Yarışma Testi ve TorchScript Üretimi (`evaluate.py`)

Modelin hızını (`threads=1`), boyutunu (`MB`), test doğruluğunu (`flower_data/test`) ve ağırlıklı final skorunu ölçmek için:

```powershell
python evaluate.py
```

---

### 3. Veri Seti Hazırlığı ve Eğitim (`train.py`)

Oxford 102 veri seti klasörde yoksa otomatik indirmek için:

```powershell
python download_data.py
```

#### Sıfırdan Eğitim (Oxford 102 + Varsa Kaggle `jpeg-192x192`)

```powershell
python train.py --epochs-head 2 --epochs-ft 8 --combine-val
```

#### Mevcut Ağırlıklar Üzerinden `224x224` Temiz İnce Ayar ve Model Soup

```powershell
python train.py --resume runs/effv2b0/best_weights.pth --soup-with runs/effv2b0/best_weights.pth --epochs-head 0 --epochs-ft 2 --train-size 224 --clean-aug --lr 6e-5 --combine-val
```

---

## Uygulanan Mühendislik Teknikleri

1. **Hibrit Veri Birleştirme (22.257 Etiketli Görsel):** Oxford 102 (`flower_data`) ile Kaggle (`jpeg-192x192`) veri setlerindeki eşleşen 102 sınıf birleştirilmiş; isim farkları (`KAGGLE_SYNONYMS`) giderilmiş ve birleşik `marigold` sınıfı otomatik olarak Sınıf `5` (`english marigold`) ve Sınıf `47` (`marigold`) olarak ayrıştırılmıştır.
2. **Sınıf Dengeli & Görülmemiş Örnek Öncelikli Örnekleme (`BalancedCombinedDataset`):** Büyük sınıfların küçük sınıfları baskılamasını önlemek için her epoch'ta sınıf başına dengeli örnek çekilmiş ve daha önce görülmemiş görsellere öncelik verilmiştir.
3. **2 Aşamalı Transfer Öğrenme + Temiz Çözünürlük İnce Ayarı (`--clean-aug`):** Önce yalnızca sınıflandırıcı katman (`Head`), ardından tüm ağ (`CosineAnnealingLR` ve `label_smoothing=0.1` ile) eğitilmiş; son aşamada `224x224` test çözünürlüğünde ağır bozulmalar kapatılarak ince ayar yapılmıştır.
4. **3'lü Model Soup (Stochastic Weight Averaging):** Farklı eğitim aşamalarının ağırlıkları ağırlık uzayında birleştirilerek model boyutu (`22.79 MB`) ve çıkarım süresi (`31.48 ms`) sabit tutulurken doğruluk `%99.63`'e taşınmıştır.
