# Smart-Irigation-IoT · Tim ZiziNisa

**Kasus:** 07 · Irigasi Pintar Berbasis IoT · **Aplikasi:** https://smart-irigation-iot-machine-learning.streamlit.app/ · **Video:** [tautan video]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| Nurjihan Rizky Thahira | [NIM] | [misalnya model, evaluasi, dan aplikasi] |
| Annisa Zulfa Maulidia | 2403520 | [misalnya EDA dan pembersihan data] |
| [nama] | [NIM] | [misalnya laporan dan video] |

## Masalah dan pengguna

Pengguna aplikasi ini adalah pengelola rumah kaca sayuran di Lembang yang memiliki 16 petak dengan sensor IoT.
Selama ini keputusan menyiram dibuat oleh agronom yang tidak dapat berjaga 24 jam. Model ini meniru keputusan
agronom untuk menentukan, setiap jam dan untuk setiap petak, apakah petak **perlu disiram sekarang**, sekaligus
menunda penyiraman bila prakiraan hujan menunjukkan sebentar lagi akan hujan.

## Data dan pembersihan

Data: `raw.csv`, satu baris = satu petak pada satu jam, Agustus–Oktober 2025 (35.452 baris).

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Baris duplikat | 297 | Dihapus dengan `drop_duplicates` |
| Penulisan nama tanaman tidak konsisten (`Cabai`, `CABAI`, `cabe`, `Tomat`, `tomat ` dengan spasi, `Selada`, `lettuce`) | 1.063 | Disamakan dengan `strip`, `lower`, lalu `replace` (`cabe` → `cabai`, `lettuce` → `selada`) |
| Data kosong pada `suhu_udara_C` | 528 | Diisi nilai terakhir di petak yang sama (*forward fill*), karena hanya memakai data masa lalu |
| Data kosong pada `prakiraan_hujan_6jam_mm` | 354 | Diisi nilai terakhir di petak yang sama (*forward fill*) |
| Kelembapan tanah mustahil (0% atau ≥ 100%, termasuk 127,5%) | 103 | Ditandai sebagai error sensor, tidak dipakai untuk melatih dan menilai model |
| Sensor kelembapan macet (nilai sama ≥ 6 jam, terpanjang 47 jam di P05) | 54 | Ditandai sebagai error sensor, tidak dipakai untuk melatih dan menilai model |
| Cahaya bernilai negatif | 70 | Dianggap kosong, lalu diisi nilai terakhir (*forward fill*) |
| Jam yang hilang | 4–16 jam per petak (173 total) | Fitur laju pengeringan dihitung berdasarkan waktu, bukan baris sebelumnya, agar tidak salah ketika ada jam yang hilang |
| Label malam tidak dapat dipercaya: agronom hanya mencatat pukul 05–17, sehingga tanah kering di malam hari tetap berlabel 0 | 16.118 (jam 18–04) | Model dilatih dan dinilai hanya dengan data pukul 05–17; setelah jadi, model tetap dipakai 24 jam |
| Label "perlu siram" tercatat satu jam terlambat (sensor sudah membaca tanah basah setelah disiram) | 363 | Label dipindah ke jam sebelumnya saat tanah masih kering, jam setelahnya diberi label 0 |

Fitur tambahan: `turun_1j` dan `turun_3j` (perubahan kelembapan tanah dalam 1 dan 3 jam terakhir) untuk
menggambarkan seberapa cepat tanah mengering.

Pembagian data berdasarkan waktu: *train* 1 Agustus–30 September, *validation* 1–14 Oktober, *test* 15–31 Oktober 2025.

## Hasil

Metrik utama: F1 untuk kelas `perlu_siram = 1` (data pukul 05–17, sensor normal).

| Model | F1 validation | F1 test |
|---|---|---|
| Baseline: siram jika kelembapan tanah < 35% | 0,346 | 0,306 |
| Aturan buatan tangan (batas per tanaman + cek prakiraan hujan) | 0,822 | 0,754 |
| Gradient Boosting (model final) | 0,858 | 0,754 |

Threshold yang dipakai: **0,45**, karena memberi F1 tertinggi di data validation. Pada threshold ini precision
model tinggi (0,927 di data test), sehingga notifikasi siram jarang salah: air tidak terbuang, risiko akar busuk
kecil, dan pengelola tetap percaya pada notifikasi. Konsekuensinya recall lebih rendah (0,636), artinya sebagian
petak yang perlu disiram terlambat terdeteksi. Jika klien lebih mengutamakan mencegah tanaman stres, threshold
dapat diturunkan dengan konsekuensi lebih banyak notifikasi palsu.

Mengapa model jauh lebih baik dari baseline: agronom memakai batas yang berbeda untuk setiap tanaman (selada
sekitar 42%, tomat sekitar 36%, cabai sekitar 32%) dan hampir tidak pernah menyiram bila prakiraan hujan ≥ 2 mm.
Baseline memakai satu batas (35%) untuk semua tanaman dan tidak melihat prakiraan hujan. Di data test, model
setara dengan aturan buatan tangan yang juga ditemukan dari analisis data; artinya model pada dasarnya
mempelajari aturan agronom yang sama.

## Cara menjalankan

Aplikasi (memerlukan Python 3.12 karena model disimpan dengan scikit-learn 1.6.1):

```bash
pip install -r requirements.txt
streamlit run app.py
```

Notebook: buka `Kasus07_Irigasi_Pintar.ipynb` di Google Colab atau Jupyter, letakkan `raw.csv` di folder yang sama,
lalu jalankan semua sel dari atas. `raw.csv` tidak disertakan di repositori karena merupakan data klien.

## Struktur repositori

```
Kasus07_Irigasi_Pintar.ipynb   notebook dari EDA, pembersihan, pemodelan, evaluasi, sampai penyimpanan model
src/features.py                fungsi pembersihan data dan pembuatan fitur
app.py                         aplikasi web (Streamlit) yang memuat model dan memberi saran beserta alasannya
model_irigasi.joblib           model final (pipeline Gradient Boosting) beserta threshold, daftar fitur, dan aturan
metadata.json                  versi pustaka, fitur masukan, threshold, dan metrik validation dan test
requirements.txt               daftar pustaka yang dibutuhkan
```

## Keterbatasan
- Agronom hanya mencatat keputusan pukul 05–17, sehingga model belajar dari kondisi siang hari. Saran di malam
  hari belum dapat diverifikasi terhadap keputusan agronom.
- Data hanya mencakup 3 bulan (Agustus–Oktober 2025). Nilai F1 turun dari validation ke test, kemungkinan karena
  kondisi akhir Oktober (memasuki musim hujan) berbeda dari periode latih. Musim lain belum terwakili.
- Model meniru keputusan agronom, termasuk kesalahan yang mungkin ada di catatannya. Model tidak mengetahui
  kebutuhan air tanaman yang sebenarnya.
- Model bergantung pada kualitas sensor dan prakiraan hujan. Pembacaan sensor yang rusak ditolak oleh aplikasi,
  tetapi sensor yang meleset sedikit (misalnya selisih beberapa persen) tidak dapat dideteksi.
- Belum ada data volume air yang disiram, sehingga model hanya memutuskan perlu atau tidak, bukan berapa banyak.
