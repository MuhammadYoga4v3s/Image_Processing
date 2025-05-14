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

# Kernel eksperimental 1 (Edge Detection)
kernelEks1 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=float)

# Kernel eksperimental 2 (Sharpen)
kernelEks2 = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=float)

# Kernel eksperimental 3 (Emboss)
kernelEks3 = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]
], dtype=float)

# Baca kedua citra berwarna
citra1 = cv2.imread('Lena_Ori-Colored.tif', cv2.IMREAD_COLOR)
citra2 = cv2.imread('aTree_inMyVillage.jpg', cv2.IMREAD_COLOR)

# Terapkan konvolusi pada kedua citra
hasilEks1_1 = konvolusiWarna(citra1, kernelEks1)
hasilEks2_1 = konvolusiWarna(citra1, kernelEks2)
hasilEks3_1 = konvolusiWarna(citra1, kernelEks3)

hasilEks1_2 = konvolusiWarna(citra2, kernelEks1)
hasilEks2_2 = konvolusiWarna(citra2, kernelEks2)
hasilEks3_2 = konvolusiWarna(citra2, kernelEks3)

# Gabungkan hasil untuk masing-masing citra
gabung1 = np.hstack((hasilEks1_1, hasilEks2_1, hasilEks3_1))
gabung2 = np.hstack((hasilEks1_2, hasilEks2_2, hasilEks3_2))

# Tampilkan hasil dalam window terpisah
cv2.namedWindow('Hasil Eksperimental Lena', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Eksperimental Lena', 1200, 400)
cv2.imshow('Hasil Eksperimental Lena', gabung1)

cv2.namedWindow('Hasil Eksperimental Tree', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Eksperimental Tree', 1200, 400)
cv2.imshow('Hasil Eksperimental Tree', gabung2)

cv2.waitKey(0)
cv2.destroyAllWindows()