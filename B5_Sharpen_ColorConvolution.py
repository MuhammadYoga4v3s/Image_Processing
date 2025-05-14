# Nama File : ImageSharpening.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk menerapkan berbagai teknik penajaman Citra berwarna
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (15:55 WIB)

import cv2
import numpy as np

def Pertajam(image, kernel):
    # Buat padding untuk citra
    pad_size = 1  # Karena semua kernel 3x3
    padded = cv2.copyMakeBorder(image, 
                              pad_size, pad_size, 
                              pad_size, pad_size, 
                              cv2.BORDER_REFLECT)
    
    # Inisialisasi output
    output = np.zeros_like(image, dtype=float)
    
    # Iterasi melalui setiap piksel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):  # Untuk setiap channel warna
                # Ambil neighborhood 3x3
                neighborhood = padded[i:i+3, j:j+3, k]
                # Hitung konvolusi
                output[i, j, k] = np.sum(neighborhood * kernel)
    
    # Normalisasi dan konversi ke uint8
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

def Bingkai(original, results, start_idx, end_idx):
    # Ambil subset hasil yang ingin ditampilkan
    subset_results = results[start_idx:end_idx]
    
    # Resize semua gambar ke ukuran yang sama (ukuran gambar asli)
    h, w = original.shape[:2]
    resized_results = [cv2.resize(img, (w, h)) for img in subset_results]
    
    # Gabungkan gambar dalam satu baris
    combined = np.hstack([original] + resized_results)
    
    # Tambahkan label
    cv2.putText(combined, "Original", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Label untuk hasil
    for i in range(len(subset_results)):
        x_pos = (i+1)*w + 10
        cv2.putText(combined, f"K{start_idx+i+1}", (x_pos, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return combined

# Baca kedua citra berwarna
citra1 = cv2.imread('Lena_Ori-Colored.tif', cv2.IMREAD_COLOR)
citra2 = cv2.imread('aTree_inMyVillage.jpg', cv2.IMREAD_COLOR)

# Definisi semua kernel penajaman
kernels = [
    ("Kernel 1:\n-1 -1 -1\n-1  8 -1\n-1 -1 -1", 
     np.array([[-1, -1, -1],
               [-1,  8, -1],
               [-1, -1, -1]], dtype=float)),
    
    ("Kernel 2:\n-1 -1 -1\n-1  9 -1\n-1 -1 -1", 
     np.array([[-1, -1, -1],
               [-1,  9, -1],
               [-1, -1, -1]], dtype=float)),
    
    ("Kernel 3:\n 0 -1  0\n-1  5 -1\n 0 -1  0", 
     np.array([[ 0, -1,  0],
               [-1,  5, -1],
               [ 0, -1,  0]], dtype=float)),
    
    ("Kernel 4:\n 1 -2  1\n-2  5 -2\n 1 -2  1", 
     np.array([[ 1, -2,  1],
               [-2,  5, -2],
               [ 1, -2,  1]], dtype=float)),
    
    ("Kernel 5:\n 1 -2  1\n-2  4 -2\n 1 -2  1", 
     np.array([[ 1, -2,  1],
               [-2,  4, -2],
               [ 1, -2,  1]], dtype=float)),
    
    ("Kernel 6:\n 0  1  0\n 1 -4  1\n 0  1  0", 
     np.array([[ 0,  1,  0],
               [ 1, -4,  1],
               [ 0,  1,  0]], dtype=float))
]

# Proses setiap citra dengan semua kernel
results1 = [Pertajam(citra1, kernel) for _, kernel in kernels]
results2 = [Pertajam(citra2, kernel) for _, kernel in kernels]

# Buat tampilan hasil dalam 2 window (4 gambar dan 3 gambar)
gabung1_part1 = Bingkai(citra1, results1, 0, 3)  # Kernel 1-3
gabung1_part2 = Bingkai(citra1, results1, 3, 6)  # Kernel 4-6

gabung2_part1 = Bingkai(citra2, results2, 0, 3)  # Kernel 1-3
gabung2_part2 = Bingkai(citra2, results2, 3, 6)  # Kernel 4-6

# Tampilkan hasil dalam window terpisah
cv2.namedWindow('Hasil Penajaman Lena (K1-K3)', cv2.WINDOW_NORMAL)
cv2.imshow('Hasil Penajaman Lena (K1-K3)', gabung1_part1)

cv2.namedWindow('Hasil Penajaman Lena (K4-K6)', cv2.WINDOW_NORMAL)
cv2.imshow('Hasil Penajaman Lena (K4-K6)', gabung1_part2)

cv2.namedWindow('Hasil Penajaman Tree (K1-K3)', cv2.WINDOW_NORMAL)
cv2.imshow('Hasil Penajaman Tree (K1-K3)', gabung2_part1)

cv2.namedWindow('Hasil Penajaman Tree (K4-K6)', cv2.WINDOW_NORMAL)
cv2.imshow('Hasil Penajaman Tree (K4-K6)', gabung2_part2)

# Tampilkan deskripsi kernel di console
print("\nDeskripsi Kernel Penajaman:")
for i, (name, _) in enumerate(kernels):
    print(f"\nKernel {i+1}:")
    print(name)

cv2.waitKey(0)
cv2.destroyAllWindows()