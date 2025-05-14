# Nama File : Histogram.py
# Deskripsi : Fungsi lokal untuk menghitung histogram citra grayscale
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (15:13 WIB)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi Histogram untuk citra grayscale
def Histogram(citra):
    m, n = citra.shape
    citra = citra.astype(np.float64)
    f = np.zeros(256, dtype=np.int32)
    
    for j in range(m):
        for k in range(n):
            nilai_piksel = int(citra[j, k])
            f[nilai_piksel] += 1
            
    return f

# Aplikasi Fungsi
Citra = cv2.imread('aTree_inMyVillage.jpg', cv2.IMREAD_GRAYSCALE)
# Citra = cv2.imread('Lena_Ori-Colored.tif', cv2.IMREAD_GRAYSCALE)
HistogramCitra = Histogram(Citra)

# Tampilkan histogram (contoh sederhana)
# print("Nilai Histogram:")
# print(HistogramCitra)

# Tampilkan histogram sebagai grafik
# plt.bar(range(256), HistogramCitra)
# plt.title('Histogram Citra Grayscale')
# plt.xlabel('Nilai Piksel')
# plt.ylabel('Frekuensi')
# plt.show()

# Tampilkan histogram sebagai garis
plt.plot(range(256), HistogramCitra, color='black', linewidth=1)
plt.title('Histogram Citra Grayscale (Garis)')
plt.xlabel('Nilai Piksel')
plt.ylabel('Frekuensi')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0, 255])
plt.show()