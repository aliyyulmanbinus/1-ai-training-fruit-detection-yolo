# ============================================================
# detect.py - Script Inference Deteksi Buah menggunakan YOLO
# Wrapstation Technical Test - AI Training
# ============================================================

# --- BAGIAN 1: Import Library ---
from ultralytics import YOLO  # Library YOLO untuk deteksi objek
import cv2                     # OpenCV untuk proses gambar & popup
import os                      # Untuk operasi file dan folder

# ============================================================
# --- BAGIAN 2: Konfigurasi (Path ditentukan di sini!) ---
# ============================================================

# Path ke model hasil training
# best.pt = model terbaik hasil training kita
MODEL_PATH = os.path.join(
    'runs', 'detect', 'runs', 'detect',
    'fruit-detection', 'weights', 'best.pt'
)

# Path ke folder gambar yang akan dideteksi
# Sesuai requirement: path ditentukan dalam script, bukan dari terminal
IMAGE_FOLDER = 'test-images'

# Confidence threshold = batas minimum keyakinan deteksi
# 0.25 = hanya tampilkan deteksi dengan keyakinan >= 25%
CONFIDENCE = 0.25

# Warna untuk setiap kelas buah (format BGR untuk OpenCV)
# BGR = Blue, Green, Red (kebalikan dari RGB)
COLORS = {
    'Apple'      : (0, 255, 0),      # Hijau
    'Banana'     : (0, 255, 255),    # Kuning
    'Grapes'     : (128, 0, 128),    # Ungu
    'Kiwi'       : (0, 128, 0),      # Hijau tua
    'Mango'      : (0, 165, 255),    # Orange
    'Orange'     : (0, 69, 255),     # Orange tua
    'Pineapple'  : (255, 191, 0),    # Biru muda
    'Sugerapple' : (255, 0, 255),    # Magenta
    'Watermelon' : (0, 0, 255),      # Merah
}

# Warna default jika kelas tidak ada di dictionary
DEFAULT_COLOR = (255, 255, 255)  # Putih

# ============================================================
# --- BAGIAN 3: Fungsi Menggambar Hasil Deteksi ---
# ============================================================

def draw_detections(image, results):
    """
    Fungsi untuk menggambar bounding box dan label
    di atas gambar berdasarkan hasil deteksi YOLO
    
    Parameter:
    - image   : gambar asli (numpy array dari OpenCV)
    - results : hasil deteksi dari model YOLO
    
    Return:
    - image   : gambar yang sudah diberi anotasi
    """
    # Ambil nama-nama kelas dari model
    names = results[0].names
    
    # Loop setiap objek yang terdeteksi
    for box in results[0].boxes:
        # Ambil koordinat bounding box
        # xyxy = format (x1, y1, x2, y2) = sudut kiri atas & kanan bawah
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Ambil confidence score (keyakinan model)
        confidence = float(box.conf[0])
        
        # Ambil ID kelas dan nama kelasnya
        class_id = int(box.cls[0])
        class_name = names[class_id]
        
        # Tentukan warna berdasarkan nama kelas
        color = COLORS.get(class_name, DEFAULT_COLOR)
        
        # Gambar kotak (bounding box) di sekitar objek
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Buat teks label: "Apple 95%"
        label = f"{class_name} {confidence:.0%}"
        
        # Ukuran teks untuk menghitung background label
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  # Ukuran font
            2     # Ketebalan
        )
        
        # Gambar background kotak untuk teks label
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1  # -1 = fill (isi penuh)
        )
        
        # Tulis teks label di atas bounding box
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,           # Ukuran font
            (0, 0, 0),     # Warna teks: hitam
            2,             # Ketebalan
            cv2.LINE_AA    # Anti-aliasing untuk teks yang lebih halus
        )
    
    return image

# ============================================================
# --- BAGIAN 4: Fungsi Utama Deteksi ---
# ============================================================

def detect():
    print("=" * 60)
    print("  FRUIT DETECTION - Inference")
    print("  Wrapstation Technical Test")
    print("=" * 60)
    
    # --- Cek Model ---
    if not os.path.exists(MODEL_PATH):
        print(f" ERROR: Model tidak ditemukan di {MODEL_PATH}")
        return
    print(f" Model    : {MODEL_PATH}")
    
    # --- Cek Folder Gambar ---
    if not os.path.exists(IMAGE_FOLDER):
        print(f" ERROR: Folder {IMAGE_FOLDER} tidak ditemukan!")
        return
    
    # --- Ambil Semua Gambar dari Folder ---
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(extensions)
    ]
    
    if not image_files:
        print(f" ERROR: Tidak ada gambar di folder {IMAGE_FOLDER}")
        return
    
    print(f" Folder   : {IMAGE_FOLDER}")
    print(f" Gambar   : {len(image_files)} file ditemukan")
    print(f" Confidence: {CONFIDENCE}")
    print("=" * 60)
    print("\n Kontrol:")
    print("   SPACE / N  = Gambar berikutnya")
    print("   B          = Gambar sebelumnya")  
    print("   Q / ESC    = Keluar\n")
    
    # --- Load Model ---
    print(" Loading model...")
    model = YOLO(MODEL_PATH)
    print(" Model berhasil dimuat!\n")
    
    # --- Loop Setiap Gambar ---
    idx = 0
    while idx < len(image_files):
        image_file = image_files[idx]
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        
        print(f" Mendeteksi: {image_file} ({idx+1}/{len(image_files)})")
        
        # Baca gambar dengan OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            print(f" Gagal membaca: {image_file}, skip...")
            idx += 1
            continue
        
        # Jalankan deteksi
        results = model(image, conf=CONFIDENCE, verbose=False)
        
        # Hitung jumlah objek terdeteksi
        num_detected = len(results[0].boxes)
        print(f" Terdeteksi: {num_detected} objek")
        
        # Gambar hasil deteksi di atas gambar
        annotated_image = draw_detections(image.copy(), results)
        
        # Tambahkan info teks di pojok kiri atas gambar
        info_text = f"{image_file} | {num_detected} objek | {idx+1}/{len(image_files)}"
        cv2.putText(
            annotated_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Resize gambar jika terlalu besar untuk layar
        h, w = annotated_image.shape[:2]
        max_width = 1280
        max_height = 720
        if w > max_width or h > max_height:
            scale = min(max_width/w, max_height/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            annotated_image = cv2.resize(
                annotated_image, (new_w, new_h)
            )
        
        # ============================================
        # TAMPILKAN POPUP WINDOW
        # ============================================
        window_title = f"Fruit Detection - {image_file}"
        cv2.imshow(window_title, annotated_image)
        
        # Tunggu input keyboard
        key = cv2.waitKey(0) & 0xFF
        
        # Q atau ESC = keluar
        if key == ord('q') or key == 27:
            print("\n Keluar dari program.")
            break
        # B = kembali ke gambar sebelumnya
        elif key == ord('b') and idx > 0:
            idx -= 1
        # SPACE atau N = gambar berikutnya
        else:
            idx += 1
        
        # Tutup window sebelum buka yang baru
        cv2.destroyAllWindows()
    
    # Tutup semua window saat selesai
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print(" Deteksi selesai!")
    print("=" * 60)

# ============================================================
# --- BAGIAN 5: Entry Point ---
# ============================================================

if __name__ == '__main__':
    detect()