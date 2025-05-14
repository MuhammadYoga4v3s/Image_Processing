# Nama File : SmoothingAverage.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk menerapkan smoothing average ke Citra berwarna
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (15:55 WIB)

import cv2
import numpy as np
from B10_NoiseSaltPepper_Color import NoiseSaltPepper

def averageFilter(image, kernel_size=3):
    # Pastikan kernel size ganjil
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Buat kernel average
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    kernel = kernel / (kernel_size * kernel_size)
    
    # Buat padding untuk citra
    pad_size = kernel_size // 2
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
                # Ambil neighborhood (area sesuai kernel size)
                neighborhood = padded[i:i+kernel_size, j:j+kernel_size, k]
                # Hitung average (dot product dengan kernel)
                output[i, j, k] = np.sum(neighborhood * kernel)
    
    # Normalisasi dan konversi ke uint8
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

# Baca kedua citra berwarna
citra1 = cv2.imread('Lena_Ori-Colored.tif', cv2.IMREAD_COLOR)
citra2 = cv2.imread('aTree_inMyVillage.jpg', cv2.IMREAD_COLOR)

# Tambahkan noise salt & pepper dengan probabilitas 10%
noisy1 = NoiseSaltPepper(citra1, prob=0.10)
noisy2 = NoiseSaltPepper(citra2, prob=0.10)

# Terapkan average filter dengan kernel 3x3
hasil_average1 = averageFilter(noisy1, 3)
hasil_average2 = averageFilter(noisy2, 3)

# Gabungkan gambar untuk masing-masing citra
gabung1 = np.hstack((citra1, noisy1, hasil_average1))
gabung2 = np.hstack((citra2, noisy2, hasil_average2))

# Tambahkan label
for gabung, width in [(gabung1, citra1.shape[1]), (gabung2, citra2.shape[1])]:
    cv2.putText(gabung, "Original", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(gabung, "Noisy (10%)", (width + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(gabung, "Average 3x3", (2*width + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Tampilkan hasil dalam window terpisah
cv2.namedWindow('Hasil Average Filter Lena', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Average Filter Lena', 1200, 400)
cv2.imshow('Hasil Average Filter Lena', gabung1)

cv2.namedWindow('Hasil Average Filter Tree', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Average Filter Tree', 1200, 400)
cv2.imshow('Hasil Average Filter Tree', gabung2)

cv2.waitKey(0)
cv2.destroyAllWindows()