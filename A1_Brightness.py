# Nama File : Pencerahan.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk mencerahkan dan menggelapkan Citra
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (155:13 WIB)

import cv2
import numpy as np

# Fungsi Pencerahan (Brightness Adjustment)
def Pencerahan(X, k):
    # Ambil ukuran gambar
    N, M = X.shape[0], X.shape[1]
    hasil = np.zeros((N, M), dtype=np.uint8)

    # Konversi ke grayscale dulu
    grayX = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)

    # Proses pencerahan
    for i in range(N):
        for j in range(M):
            gray = int(grayX[i, j])
            cerah = gray + k
            if cerah > 255:
                cerah = 255
            elif cerah < 0:
                cerah = 0
            hasil[i, j] = cerah

    return hasil

# Aplikasi Fungsi
Citra = cv2.imread('Lena_Ori-Colored.tif')
K1 = -80 , K2 = 40
CitraHasil = Pencerahan(Citra, K1)

# Tampilkan hasil
cv2.namedWindow('Hasil Pencerahan', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Pencerahan', 400, 400)
cv2.imshow('Hasil Pencerahan', CitraHasil)
cv2.waitKey(0)
cv2.destroyAllWindows()
