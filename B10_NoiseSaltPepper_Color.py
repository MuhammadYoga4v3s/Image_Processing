# Nama File : ColorNoiseSaltPepper.py
# Deskripsi : Algoritma pemrosesan Citra digital untuk menerapkan noise Salt and Pepper ke Citra berwarna
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (15:55 WIB)

import cv2
import numpy as np

def NoiseSaltPepper(image, prob=0.05):
    output = np.copy(image)
    
    # Salt noise
    salt = np.random.rand(*image.shape[:2])
    output[salt < prob/2] = [255, 255, 255]
    
    # Pepper noise
    pepper = np.random.rand(*image.shape[:2])
    output[pepper < prob/2] = [0, 0, 0]
    
    return output

# Baca kedua citra berwarna
citra1 = cv2.imread('Lena_Ori-Colored.tif', cv2.IMREAD_COLOR)
citra2 = cv2.imread('aTree_inMyVillage.jpg', cv2.IMREAD_COLOR)

# Tambahkan noise dengan probabilitas berbeda
hasilNoise1_1 = NoiseSaltPepper(citra1, prob=0.05)  # Noise 5%
hasilNoise1_2 = NoiseSaltPepper(citra1, prob=0.10)  # Noise 10%
hasilNoise1_3 = NoiseSaltPepper(citra1, prob=0.20)  # Noise 20%

hasilNoise2_1 = NoiseSaltPepper(citra2, prob=0.05)  # Noise 5%
hasilNoise2_2 = NoiseSaltPepper(citra2, prob=0.10)  # Noise 10%
hasilNoise2_3 = NoiseSaltPepper(citra2, prob=0.20)  # Noise 20%

# Gabungkan hasil untuk masing-masing citra
gabung1 = np.hstack((citra1, hasilNoise1_1, hasilNoise1_2, hasilNoise1_3))
gabung2 = np.hstack((citra2, hasilNoise2_1, hasilNoise2_2, hasilNoise2_3))

# Tampilkan hasil dalam window terpisah
cv2.namedWindow('Hasil Noise Salt & Pepper Lena', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Noise Salt & Pepper Lena', 1600, 400)
cv2.imshow('Hasil Noise Salt & Pepper Lena', gabung1)

cv2.namedWindow('Hasil Noise Salt & Pepper Tree', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hasil Noise Salt & Pepper Tree', 1600, 400)
cv2.imshow('Hasil Noise Salt & Pepper Tree', gabung2)

cv2.waitKey(0)
cv2.destroyAllWindows()