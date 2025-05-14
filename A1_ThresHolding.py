# Nama File : Thresholding.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk melakukan Thresholding terhadap Citra
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (155:13 WIB)

import cv2
import numpy as np

# Fungsi konversi RGB ke grayscale dan thresholding
def Thresholding(X, Nilai_TreshHold):
    # Ambil ukuran gambar dan simpan di variabel N dan M
    N, M = X.shape[0], X.shape[1]
    # Buat array hasil dengan ukuran N x M (grayscale)
    hasil = np.zeros((N, M), dtype=np.uint8)

    # Konversi gambar ke grayscale menggunakan cv2
    grayX = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    # Proses : ThresHolding
    for i in range(N):
        for j in range(M):
            gray = grayX[i, j]
            if gray > Nilai_TreshHold:
                hasil[i, j] = 255
            else:
                hasil[i, j] = 0

    return hasil

# Aplikasi Fungsi
Citra = cv2.imread('Lena_Ori-Colored.tif')
T1, T2, T3 = 127 , 60, 193
CitraHasil = Thresholding(Citra, T3)

# hasil
cv2.namedWindow('Hasil TreshHolding', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil TreshHolding', 400, 400)
cv2.imshow('Hasil TreshHolding', CitraHasil)
cv2.waitKey(0)
cv2.destroyAllWindows()

