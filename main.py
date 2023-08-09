import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Masukkan data pelanggan ke dalam DataFrame
data = pd.DataFrame({
    'Pendapatan': [5000, 4500, 6000, 7500, 7000, 10000, 9000, 8000, 8500, 9500],
    'Pengeluaran': [1000, 2000, 1500, 2500, 3000, 3500, 4000, 3000, 3500, 4000]
})

# Visualisasi data
plt.scatter(data['Pendapatan'], data['Pengeluaran'])
plt.xlabel('Pendapatan')
plt.ylabel('Pengeluaran')
plt.title('Customer Data')
plt.show()

# Menentukan jumlah cluster yang diinginkan (dalam contoh ini, kita akan menggunakan 2 cluster)
num_clusters = 2

# Melakukan K-Means Clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data)

# Mendapatkan label hasil clustering untuk setiap data pelanggan
labels = kmeans.labels_

# Menambahkan kolom 'Cluster' ke dalam DataFrame
data['Cluster'] = labels

# Visualisasi hasil clustering
plt.scatter(data[data['Cluster'] == 0]['Pendapatan'], data[data['Cluster'] == 0]['Pengeluaran'], color='red', label='Cluster 1')
plt.scatter(data[data['Cluster'] == 1]['Pendapatan'], data[data['Cluster'] == 1]['Pengeluaran'], color='blue', label='Cluster 2')
#
plt.xlabel('Pendapatan')
plt.ylabel('Pengeluaran')
# Ini adalah perintah untuk memberikan label pada sumbu x dan y pada plot.
# Label sumbu x adalah "Pendapatan" dan label sumbu y adalah "Pengeluaran".
plt.title('Hasil Clustering dengan K-Means')
# Ini adalah perintah untuk memberikan judul pada plot. Judulnya adalah "Hasil Clustering dengan K-Means".
plt.legend()
# Ini adalah perintah untuk menampilkan legenda yang akan memberi tahu warna mana yang mewakili setiap kluster.
plt.show()