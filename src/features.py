"""
src/features.py — Pembersihan data dan pembuatan fitur untuk Kasus 07 (Irigasi Pintar).

Isi file ini sama dengan fungsi di notebook (Bagian 4 dan 5), dipisah supaya bisa dipakai ulang:
    from src.features import bersihkan, buat_fitur, FITUR, TARGET
    df = buat_fitur(bersihkan(pd.read_csv("raw.csv")))
"""

import numpy as np
import pandas as pd

FITUR_KATEGORI = ["tanaman", "jenis_tanah"]
FITUR_ANGKA = ["umur_tanaman_hari", "kelembaban_tanah_pct", "suhu_udara_C", "kelembaban_udara_pct",
               "cahaya_lux", "curah_hujan_mm", "prakiraan_hujan_6jam_mm", "turun_1j", "turun_3j"]
FITUR = FITUR_KATEGORI + FITUR_ANGKA
TARGET = "perlu_siram"

JAM_KERJA_AGRONOM = (5, 17)          # label hanya bisa dipercaya pada jam 05–17


def bersihkan(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data mentah dari sistem klien."""
    df = df.drop_duplicates().copy()                              # (1) hapus baris dobel
    df["waktu"] = pd.to_datetime(df["waktu"])
    df = df.sort_values(["petak", "waktu"]).reset_index(drop=True)

    # (2) rapikan nama tanaman: "CABAI", "cabe", "tomat ", "lettuce", dst.
    df["tanaman"] = (df["tanaman"].str.strip().str.lower()
                     .replace({"cabe": "cabai", "lettuce": "selada"}))

    # (3) tandai pembacaan kelembapan tanah yang rusak: 0, >= 100, atau nilai sama >= 6 jam (sensor macet)
    k = df["kelembaban_tanah_pct"]
    rusak = (k <= 0) | (k >= 100)
    sama = df.groupby("petak")["kelembaban_tanah_pct"].diff().eq(0)
    panjang = sama.groupby((~sama).cumsum()).transform("sum")
    df["sensor_rusak"] = rusak | (panjang >= 6)

    # (4) cahaya negatif tidak mungkin → dianggap kosong
    df.loc[df["cahaya_lux"] < 0, "cahaya_lux"] = np.nan

    # (5) isi data kosong dengan nilai terakhir di petak yang sama (forward fill).
    #     Hanya memakai masa lalu, sehingga tidak ada kebocoran data dari masa depan.
    for kol in ["suhu_udara_C", "cahaya_lux", "prakiraan_hujan_6jam_mm"]:
        df[kol] = df.groupby("petak")[kol].ffill()
    df["kelembaban_tanah_pct"] = df["kelembaban_tanah_pct"].where(~df["sensor_rusak"])
    df["kelembaban_tanah_pct"] = df.groupby("petak")["kelembaban_tanah_pct"].ffill()

    # (6) label "perlu siram" yang tercatat satu jam terlambat (sensor sudah membaca tanah basah)
    #     dipindah ke jam sebelumnya, saat tanah masih kering.
    k_prev = df.groupby("petak")["kelembaban_tanah_pct"].shift(1)
    petak_sama = df["petak"].eq(df["petak"].shift(1))
    jam_siram = ((df[TARGET] == 1) & (df["kelembaban_tanah_pct"] - k_prev >= 10)
                 & (k_prev < 45) & petak_sama)
    idx = df.index[jam_siram]
    df.loc[idx, TARGET] = 0
    df.loc[idx - 1, TARGET] = 1
    return df


def buat_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """Menambah fitur laju pengeringan tanah (turun_1j, turun_3j) dan penanda jam kerja agronom."""
    df = df.copy()
    # Kelembapan 1 dan 3 jam lalu diambil berdasarkan WAKTU, bukan baris sebelumnya,
    # supaya tetap benar walaupun ada jam yang hilang.
    for n in [1, 3]:
        lalu = df[["petak", "waktu", "kelembaban_tanah_pct"]].copy()
        lalu["waktu"] = lalu["waktu"] + pd.Timedelta(hours=n)
        lalu = lalu.rename(columns={"kelembaban_tanah_pct": f"k{n}"})
        df = df.merge(lalu, on=["petak", "waktu"], how="left")
    df["turun_1j"] = df["kelembaban_tanah_pct"] - df["k1"]
    df["turun_3j"] = df["kelembaban_tanah_pct"] - df["k3"]
    df["jam"] = df["waktu"].dt.hour
    df["jam_kerja"] = df["jam"].between(*JAM_KERJA_AGRONOM)
    return df.drop(columns=["k1", "k3"])


def fitur_dari_input(tanaman, jenis_tanah, umur, k_now, k_1j, k_3j, suhu, rh, lux, hujan, prakiraan) -> pd.DataFrame:
    """Menyusun satu baris input model dari isian aplikasi (pembacaan lama boleh None)."""
    return pd.DataFrame([{
        "tanaman": tanaman, "jenis_tanah": jenis_tanah, "umur_tanaman_hari": umur,
        "kelembaban_tanah_pct": k_now, "suhu_udara_C": suhu, "kelembaban_udara_pct": rh,
        "cahaya_lux": lux, "curah_hujan_mm": hujan, "prakiraan_hujan_6jam_mm": prakiraan,
        "turun_1j": k_now - k_1j if k_1j is not None else np.nan,
        "turun_3j": k_now - k_3j if k_3j is not None else np.nan,
    }])[FITUR]
