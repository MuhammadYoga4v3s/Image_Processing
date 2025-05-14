# Nama File : PencerahanWarna.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk mencerahkan dan menggelapkan Citra berwarna
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (155:13 WIB)

import cv2
import numpy as np

# Fungsi Pencerahan untuk gambar warna (RGB/BGR)
def PencerahanWarna(X, k):
    N, M = X.shape[0], X.shape[1]
    hasil = np.zeros((N, M, 3), dtype=np.uint8)

    for i in range(N):
        for j in range(M):
            for c in range(3):  # untuk B, G, R
                nilai = int(X[i, j, c]) + k
                if nilai > 255:
                    nilai = 255
                elif nilai < 0:
                    nilai = 0
                hasil[i, j, c] = nilai

    return hasil

# Aplikasi Fungsi
Citra = cv2.imread('MuriaCountry.jpg')
K1, K2 = -80, 80
CitraHasil = PencerahanWarna(Citra, K1)

# Tampilkan hasil
cv2.namedWindow('Hasil Pencerahan Warna', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Pencerahan Warna', 400, 400)
cv2.imshow('Hasil Pencerahan Warna', CitraHasil)
cv2.waitKey(0)
cv2.destroyAllWindows()