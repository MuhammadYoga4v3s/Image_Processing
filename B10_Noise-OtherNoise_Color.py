# Nama File : ColorNoiseVarious.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk menerapkan berbagai jenis noise ke Citra berwarna
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (15:55 WIB)

import cv2
import numpy as np

def noiseGaussian(Citra):
    M, N, C = Citra.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (M, N, C))
    noisy = Citra + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def noisePoisson(Citra):
    vals = len(np.unique(Citra))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(Citra * vals) / float(vals)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def noiseSpeckle(Citra):
    M, N, C = Citra.shape
    sigma = 0.25
    speckle = np.random.randn(M, N, C) * sigma
    noisy = Citra + Citra * speckle
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Baca kedua citra berwarna
citra1 = cv2.imread('Lena_Ori-Colored.tif', cv2.IMREAD_COLOR)
citra2 = cv2.imread('aTree_inMyVillage.jpg', cv2.IMREAD_COLOR)

# Terapkan berbagai jenis noise pada citra pertama
hasilGaussian1 = noiseGaussian(citra1)
hasilPoisson1 = noisePoisson(citra1)
hasilSpeckle1 = noiseSpeckle(citra1)

# Terapkan berbagai jenis noise pada citra kedua
hasilGaussian2 = noiseGaussian(citra2)
hasilPoisson2 = noisePoisson(citra2)
hasilSpeckle2 = noiseSpeckle(citra2)

# Gabungkan hasil untuk masing-masing citra
gabung1 = np.hstack((citra1, hasilGaussian1, hasilPoisson1, hasilSpeckle1))
gabung2 = np.hstack((citra2, hasilGaussian2, hasilPoisson2, hasilSpeckle2))

# Tampilkan hasil dalam window terpisah
cv2.namedWindow('Hasil Noise - Lena', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Noise - Lena', 1600, 400)
cv2.imshow('Hasil Noise - Lena', gabung1)

cv2.namedWindow('Hasil Noise - Tree', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Noise - Tree', 1600, 400)
cv2.imshow('Hasil Noise - Tree', gabung2)

cv2.waitKey(0)
cv2.destroyAllWindows()