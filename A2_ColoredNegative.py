# Nama File : ColoredNegative.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk menerapkan efek Negatif ke Citra dalam berwarna
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (155:13 WIB)

import cv2
import numpy as np

# Fungsi negatif untuk gambar berwarna
def NegatifWarna(X):
    N, M = X.shape[0], X.shape[1]

    # Pecah channel
    B = np.zeros((N, M), dtype=np.uint8)
    G = np.zeros((N, M), dtype=np.uint8)
    R = np.zeros((N, M), dtype=np.uint8)

    for i in range(N):
        for j in range(M):
            B[i, j] = 255 - X[i, j][0]
            G[i, j] = 255 - X[i, j][1]
            R[i, j] = 255 - X[i, j][2]

    # Gabung kembali ketiga channel jadi citra warna
    hasil = cv2.merge([B, G, R])
    return hasil

# Aplikasi Fungsi
Citra = cv2.imread('Lena_Ori-Colored.tif')
CitraHasil = NegatifWarna(Citra)

# Tampilkan hasil
cv2.namedWindow('Hasil Negatif Berwarna', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Negatif Berwarna', 400, 400)
cv2.imshow('Hasil Negatif Berwarna', CitraHasil)
cv2.waitKey(0)
cv2.destroyAllWindows()
