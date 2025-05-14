# Nama File : GrayConvolution.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk menerapkan konvolusi ke Citra grayscale
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (15:55 WIB)

import cv2
import numpy as np

# Fungsi konvolusi (grayscale)
def konvolusiGrayscale(x, k):
    x = x.astype(float)
    M, N = x.shape
    m, n = k.shape
    a, b = m // 2, n // 2

    # Inisialisasi hasil konvolusi
    y = np.zeros((M, N), dtype=float)

    for i in range(a, M - a):
        for j in range(b, N - b):
            total = 0.0
            for u in range(-a, a + 1):
                for v in range(-b, b + 1):
                    total += k[u + a, v + b] * x[i + u, j + v]
            y[i, j] = total

    y = np.clip(y, 0, 255)  # Normalisasi
    return y.astype(np.uint8)

# Kernel blur 3x3
kernel3x3 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=float)
kernel3x3 = kernel3x3 / 9.0

# Kernel blur 5x5
kernel5x5 = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
], dtype=float)
kernel5x5 = kernel5x5 / 25.0

# Baca dan konversi citra ke grayscale
citra1 = cv2.imread('Lena_Ori-Colored.tif', cv2.IMREAD_GRAYSCALE)
citra2 = cv2.imread('aTree_inMyVillage.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan konvolusi dengan kernel 3x3 dan 5x5 pada kedua citra
hasil3x3_1 = konvolusiGrayscale(citra1, kernel3x3)
hasil5x5_1 = konvolusiGrayscale(citra1, kernel5x5)
hasil3x3_2 = konvolusiGrayscale(citra2, kernel3x3)
hasil5x5_2 = konvolusiGrayscale(citra2, kernel5x5)

# Gabungkan hasil secara horizontal
hasil_gabung1 = np.hstack((hasil3x3_1, hasil5x5_1))
hasil_gabung2 = np.hstack((hasil3x3_2, hasil5x5_2))

# Tampilkan hasil
cv2.namedWindow('Hasil Lena - Blur 3x3 dan 5x5', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Lena - Blur 3x3 dan 5x5', 800, 400)
cv2.imshow('Hasil Lena - Blur 3x3 dan 5x5', hasil_gabung1)

cv2.namedWindow('Hasil Tree - Blur 3x3 dan 5x5', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Tree - Blur 3x3 dan 5x5', 800, 400)
cv2.imshow('Hasil Tree - Blur 3x3 dan 5x5', hasil_gabung2)

cv2.waitKey(0)
cv2.destroyAllWindows()