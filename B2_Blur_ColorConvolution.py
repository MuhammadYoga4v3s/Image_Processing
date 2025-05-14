# Nama File : ColorConvolution.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk menerapkan konvolusi ke Citra berwarna
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (15:55 WIB)

import cv2
import numpy as np

# Fungsi konvolusi (berwarna)
def konvolusiWarna(x, k):
    x = x.astype(float)
    M, N, C = x.shape  # C adalah jumlah channel (3 untuk RGB)
    m, n = k.shape
    a, b = m // 2, n // 2

    # Inisialisasi hasil konvolusi
    y = np.zeros((M, N, C), dtype=float)

    for c in range(C):  # Loop untuk setiap channel warna
        for i in range(a, M - a):
            for j in range(b, N - b):
                total = 0.0
                for u in range(-a, a + 1):
                    for v in range(-b, b + 1):
                        total += k[u + a, v + b] * x[i + u, j + v, c]
                y[i, j, c] = total

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

# Baca kedua citra berwarna
citra1 = cv2.imread('Lena_Ori-Colored.tif', cv2.IMREAD_COLOR)
citra2 = cv2.imread('aTree_inMyVillage.jpg', cv2.IMREAD_COLOR)

# Terapkan konvolusi pada kedua citra
hasil3x3_1 = konvolusiWarna(citra1, kernel3x3)
hasil5x5_1 = konvolusiWarna(citra1, kernel5x5)
hasil3x3_2 = konvolusiWarna(citra2, kernel3x3)
hasil5x5_2 = konvolusiWarna(citra2, kernel5x5)

# Gabungkan hasil secara horizontal untuk masing-masing citra
gabung1 = np.hstack((citra1, hasil3x3_1, hasil5x5_1))
gabung2 = np.hstack((citra2, hasil3x3_2, hasil5x5_2))

# Tampilkan hasil dalam window terpisah
cv2.namedWindow('Hasil Lena - Original, Blur 3x3, Blur 5x5', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Lena - Original, Blur 3x3, Blur 5x5', 1200, 400)
cv2.imshow('Hasil Lena - Original, Blur 3x3, Blur 5x5', gabung1)

cv2.namedWindow('Hasil Tree - Original, Blur 3x3, Blur 5x5', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Tree - Original, Blur 3x3, Blur 5x5', 1200, 400)
cv2.imshow('Hasil Tree - Original, Blur 3x3, Blur 5x5', gabung2)

cv2.waitKey(0)
cv2.destroyAllWindows()