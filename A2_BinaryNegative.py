# Nama File : BinaryNegatuve.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk menerapkan efek Negatif ke Citra dalam hitam putih
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (155:13 WIB)

import cv2
import numpy as np

# Fungsi negatif untuk gambar grayscale
def Negatif(X):
    N, M = X.shape[0], X.shape[1]
    GrayX = cv2.cvtColor(Citra, cv2.COLOR_BGR2GRAY)
    hasil = np.zeros((N, M), dtype=np.uint8)

    for i in range(N):
        for j in range(M):
            hasil[i, j] = 255 - GrayX[i, j]

    return hasil

# Aplikasi Fungsi
Citra = cv2.imread('Lena_Ori-Colored.tif')
CitraHasil = Negatif(Citra)

# Tampilkan hasil
cv2.namedWindow('Hasil Negatif HitamPutih', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Negatif HitamPutih', 400, 400)
cv2.imshow('Hasil Negatif HitamPutih', CitraHasil)
cv2.waitKey(0)
cv2.destroyAllWindows()
