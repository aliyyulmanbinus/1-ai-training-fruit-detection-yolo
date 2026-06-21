# Mengimpor YOLO dari ultralytics
from ultralytics import YOLO
# Mengimpor os untuk operasi file dan folder
import os

# Konfigurasi path
DATA_YAML = os.path.join('dataset', 'data.yaml')

# Konfigurasi model training
MODEL = 'yolov8s.pt'
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8
PROJECT_NAME = 'runs'
EXPERIMENT_NAME = 'fruit-detection'

# Training
def train():
    print("=" * 60)
    print("  FRUIT DETECTION - YOLO Training")
    print("  Wrapstation Technical Test")
    print("=" * 60)
    
    # Cek apakah file data.yaml ada
    if not os.path.exists(DATA_YAML):
        print(f" ERROR: File {DATA_YAML} tidak ditemukan!")
        print("   Pastikan dataset sudah didownload dengan benar.")
        return
    
    print(f" Dataset  : {DATA_YAML}")
    print(f" Model    : {MODEL}")
    print(f" Epochs   : {EPOCHS}")
    print(f" Img Size : {IMG_SIZE}")
    print(f" Batch    : {BATCH_SIZE}")
    print("=" * 60)
    
    # Load model pre-trained
    # Jika file .pt belum ada, akan otomatis didownload dari internet
    print("\n Loading pre-trained model...")
    model = YOLO(MODEL)
    
    # Mulai training
    print("\n Memulai training...\n")
    results = model.train(
        data=DATA_YAML,         # Path ke data.yaml
        epochs=EPOCHS,          # Jumlah epoch
        imgsz=IMG_SIZE,         # Ukuran gambar
        batch=BATCH_SIZE,       # Batch size
        project=PROJECT_NAME,   # Folder output
        name=EXPERIMENT_NAME,   # Nama experiment
        device=0,               # 0 = GPU pertama (GTX 1050)
                                # 'cpu' = gunakan CPU
        workers=2,              # Jumlah worker untuk load data
                                # 2 aman untuk laptop
        patience=20,            # Early stopping: berhenti jika
                                # tidak ada improvement setelah 20 epoch
        save=True,              # Simpan model terbaik
        plots=True,             # Buat grafik hasil training
        verbose=True            # Tampilkan progress detail
    )
    
    print("\n" + "=" * 60)
    print("  Training Selesai!")
    print("=" * 60)
    
    # Tampilkan lokasi hasil training
    best_model = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, 'weights', 'best.pt')
    if os.path.exists(best_model):
        print(f"\n Model terbaik tersimpan di:")
        print(f"   {best_model}")
    
    print(f"\n Hasil lengkap tersimpan di:")
    print(f"   {os.path.join(PROJECT_NAME, EXPERIMENT_NAME)}")
    print("\n" + "=" * 60)


# Entry Point
if __name__ == '__main__':
    train()