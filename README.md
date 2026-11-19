# Smart-Irigation-IoT · Tim ZiziNisa

> Salin berkas ini menjadi `README.md` di repositori tim kalian, lalu isi semua bagian di dalam kurung siku.

**Kasus:** 7 - Smart-Irigation-IoT · **Aplikasi:** https://smart-irigation-iot-machine-learning.streamlit.app/ · **Video:** [tautan video]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| [nama] | [NIM] | [misalnya EDA dan pembersihan data] |
| [nama] | [NIM] | [misalnya model dan evaluasi] |
| [nama] | [NIM] | [misalnya API dan aplikasi] |

## Masalah dan pengguna

[Dua atau tiga kalimat: siapa penggunanya dan keputusan apa yang dibantu oleh model ini.]

## Data dan pembersihan

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| baris duplikat | 297 | dihapus dengan drop_duplicates |
| penulisan nama tanaman yang tidak konsisten (Cabai, CABAI, cabe, Tomat, tomat, Selada, lettuce)  |---| penulisan nama disamakan berdasarkan jenisnya menggunakan strip, lower, dan replace |
| data kosong pada suhu_udara_C | 528 |---|
| Data kosong pada prakiraan_hujan_6jam_mm | 354 |---|
| nilai yang mustahil pada kelembapan tanah | 106 |---|
| sensor macet (nilai tidak berubah selama 47 jam) | 157 |---|
| jam yang hilang | 3-16 jam per petak |---|
## Hasil

| Model | [metrik utama] *validation* | [metrik utama] *test* |
|---|---|---|
| *Baseline*: [nama *baseline*] | | |
| [model final] | | |

*Threshold* yang dipakai: [angka], karena [alasan bisnis].

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py   # aplikasi
```

## Struktur repositori

```
Kasus07_Irigasi_Pintar.ipynb   notebook dari EDA, pembersihan, pemodelan, sampai penyimpanan model
app.py                         aplikasi web (Streamlit) yang memuat model dan memberi saran
model_irigasi.joblib           model final (pipeline Gradient Boosting) + threshold, daftar fitur, dan aturan
requirements.txt               daftar pustaka yang dibutuhkan
```

## Keterbatasan

[Dalam kondisi apa model ini tidak dapat dipercaya? Data apa yang belum tersedia?]

## Daftar periksa sebelum dikumpulkan

- [ ] Repositori bersifat publik dan setiap anggota memiliki *commit*
- [ ] *Notebook* dapat dijalankan ulang dari awal sampai akhir tanpa galat
- [ ] *Baseline* wajib dilaporkan dan dibandingkan
- [ ] Data *test* hanya dibuka satu kali, di akhir
- [ ] `artifacts/metadata.json` berisi versi model, fitur masukan, *threshold*, dan metrik
- [ ] Aplikasi dapat dibuka orang lain melalui tautan, termasuk dari ponsel
- [ ] Aplikasi tidak berhenti mendadak ketika menerima masukan yang tidak wajar
- [ ] Laporan paling banyak 6 halaman dan video paling lama 5 menit
