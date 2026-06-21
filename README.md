# Fruit Detection - YOLOv8

Sistem object detection untuk mendeteksi buah menggunakan YOLOv8 (Ultralytics).
Technical Test AI Training - Wrapstation.

## Spesifikasi Sistem

| Komponen | Spesifikasi |
|----------|-------------|
| OS | Windows 10 Home (Build 19045) |
| CPU | Intel Core i7-7700HQ @ 2.80GHz |
| RAM | 16 GB |
| GPU | NVIDIA GeForce GTX 1060 (6GB VRAM) |
| CUDA | 12.7 |
| Driver NVIDIA | 566.36 |
| Python | 3.10.11 |
| PyTorch | 2.6.0+cu124 |
| Ultralytics | 8.4.72 |
| OpenCV | 4.13.0.92 |

> Training dilakukan menggunakan NVIDIA GeForce GTX 1060 (6GB VRAM)
> karena keterbatasan VRAM pada komputer utama.

## Dataset

Dataset: [Fruits by YOLO - Kaggle](https://www.kaggle.com/datasets/kapturovalexander/fruits-by-yolo-fruits-detection)

9 kelas buah: Apple, Banana, Grapes, Kiwi, Mango, Orange, Pineapple, Sugerapple, Watermelon

## Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME_ANDA/1-ai-training-fruit-detection-yolo.git
cd 1-ai-training-fruit-detection-yolo
```

### 2. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Training

```bash
python train.py
```

### 4. Deteksi Buah

Letakkan gambar di folder `test-images/` lalu jalankan:

```bash
python detect.py
```

Popup window akan muncul dengan hasil deteksi.
Tekan `SPACE` untuk gambar berikutnya, `Q` untuk keluar.

## File Utama

| File | Keterangan |
|------|------------|
| `train.py` | Script training model |
| `detect.py` | Script inference + popup window |
| `best.pt` | Model hasil training |
| `requirements.txt` | Daftar dependencies |
