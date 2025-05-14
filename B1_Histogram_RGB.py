# Nama File : HistogramWarna.py
# Deskripsi : Fungsi untuk menghitung histogram citra berwarna (RGB/BGR)
# Nama      : Muhammad Yoga Aminudin (24060123130106)
# Tanggal   : 13-05-2025 (15:13 WIB)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi Histogram untuk gambar warna (RGB/BGR)
def HistogramWarna(X):
    N, M = X.shape[0], X.shape[1]
    histB = np.zeros(256, dtype=np.int32)
    histG = np.zeros(256, dtype=np.int32)
    histR = np.zeros(256, dtype=np.int32)
    
    for i in range(N):
        for j in range(M):
            # Channel Blue
            nilaiB = X[i, j, 0]
            histB[nilaiB] += 1
            
            # Channel Green
            nilaiG = X[i, j, 1]
            histG[nilaiG] += 1
            
            # Channel Red
            nilaiR = X[i, j, 2]
            histR[nilaiR] += 1
            
    return histB, histG, histR

# Aplikasi Fungsi
# Citra = cv2.imread('Lena_Ori-Colored.tif')
Citra = cv2.imread('aTree_inMyVillage.jpg')
HistB, HistG, HistR = HistogramWarna(Citra)

# Tampilkan histogram dalam satu grafik
plt.figure(figsize=(10, 6))

plt.plot(range(256), HistB, color='blue', linewidth=1, label='Channel Blue')
plt.plot(range(256), HistG, color='green', linewidth=1, label='Channel Green')
plt.plot(range(256), HistR, color='red', linewidth=1, label='Channel Red')

plt.title('Histogram Citra Warna')
plt.xlabel('Nilai Piksel')
plt.ylabel('Frekuensi')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0, 255])
plt.legend()

plt.tight_layout()
plt.show()