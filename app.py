# app.py — Aplikasi Irigasi Pintar untuk pengelola rumah kaca (Kasus 07)
# Jalankan di terminal:  py -3.12 -m streamlit run app.py

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Irigasi Pintar", layout="wide")


# ---------- 1. Memuat model dan metadata ----------
@st.cache_resource                      # model cukup dimuat sekali
def muat_model():
    return joblib.load("model_irigasi.joblib")

paket = muat_model()
model = paket["model"]                  # pipeline: preprocessing + Gradient Boosting
THRESHOLD = paket["threshold"]          # batas peluang, dipilih di data validation
FITUR = paket["fitur"]                  # urutan kolom input model
ATURAN = paket["aturan"]                # batas kelembapan per tanaman + batas hujan (untuk alasan)

# metadata.json dibuat di notebook (versi, metrik validation & test). Jika belum ada, bagian
# "Tentang model" disembunyikan.
META = json.loads(Path("metadata.json").read_text(encoding="utf-8")) if Path("metadata.json").exists() else None


# ---------- 2. Contoh kasus untuk mengisi form otomatis ----------
# Kunci (key) di sini sama dengan key setiap kolom input di form.
NILAI_AWAL = dict(tanaman="cabai", tanah="pasir", umur=60, k_now=None, k_1j=None, k_3j=None,
                  suhu=20.0, rh=85.0, lux=30000.0, hujan=0.0, prakiraan=0.0)

CONTOH = {
    "Isi sendiri": {},
    "Cabai kering, cuaca cerah": dict(tanaman="cabai", tanah="lempung", umur=45, k_now=29.5, k_1j=30.3,
                                      k_3j=31.8, suhu=24.5, rh=70.0, lux=58000.0, hujan=0.0, prakiraan=0.0),
    "Cabai kering, tapi akan hujan": dict(tanaman="cabai", tanah="lempung", umur=45, k_now=29.5, k_1j=30.0,
                                          k_3j=31.0, suhu=21.0, rh=92.0, lux=15000.0, hujan=0.0, prakiraan=6.0),
    "Selada mulai kering (38%)": dict(tanaman="selada", tanah="pasir", umur=30, k_now=38.0, k_1j=39.2,
                                      k_3j=41.0, suhu=23.0, rh=75.0, lux=50000.0, hujan=0.0, prakiraan=0.0),
    "Tomat, tanah masih lembap": dict(tanaman="tomat", tanah="liat", umur=70, k_now=52.0, k_1j=52.3,
                                      k_3j=53.0, suhu=21.0, rh=82.0, lux=35000.0, hujan=0.0, prakiraan=0.0),
    "Sensor bermasalah (127,5%)": dict(tanaman="tomat", tanah="pasir", umur=70, k_now=127.5, k_1j=41.0,
                                       k_3j=41.5, suhu=20.0, rh=85.0, lux=30000.0, hujan=0.0, prakiraan=0.0),
}

# st.session_state menyimpan isi setiap kolom. Nilai awal diisi sekali saat halaman pertama dibuka.
for kunci, nilai in NILAI_AWAL.items():
    st.session_state.setdefault(kunci, nilai)

def isi_contoh():
    # Dipanggil saat pilihan contoh berubah: salin nilai contoh ke kolom-kolom form
    for kunci, nilai in CONTOH[st.session_state["contoh"]].items():
        st.session_state[kunci] = nilai


# ---------- 3. Judul ----------
st.title("Irigasi Pintar")
st.write("Masukkan data petak dan pembacaan sensor terbaru. Sistem akan menyarankan apakah petak "
         "perlu disiram sekarang, dengan mempertimbangkan prakiraan hujan.")

kiri, kanan = st.columns([3, 2], gap="large")


# ---------- 4. Form input ----------
with kiri:
    st.selectbox("Contoh kasus (opsional)", list(CONTOH), key="contoh", on_change=isi_contoh,
                 help="Pilih salah satu untuk mengisi form secara otomatis, lalu tekan Periksa.")

    with st.form("input_petak"):
        st.subheader("Data petak")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Tanaman", ["cabai", "tomat", "selada"], key="tanaman")
        c2.selectbox("Jenis tanah", ["pasir", "lempung", "liat"], key="tanah")
        c3.number_input("Umur tanaman (hari)", min_value=0, max_value=365, key="umur")

        st.subheader("Kelembapan tanah (%)")
        c1, c2, c3 = st.columns(3)
        # Nilai awal None → kolom kosong; kekosongan dicek saat validasi
        c1.number_input("Saat ini", min_value=0.0, max_value=150.0, step=0.1, key="k_now",
                        placeholder="contoh: 38.5")
        c2.number_input("1 jam lalu (opsional)", min_value=0.0, max_value=150.0, step=0.1, key="k_1j")
        c3.number_input("3 jam lalu (opsional)", min_value=0.0, max_value=150.0, step=0.1, key="k_3j")

        st.subheader("Cuaca")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Suhu udara (°C)", min_value=-10.0, max_value=60.0, step=0.1, key="suhu")
        c2.number_input("Kelembapan udara (%)", min_value=0.0, max_value=100.0, step=1.0, key="rh")
        c3.number_input("Cahaya (lux)", min_value=0.0, max_value=150000.0, step=1000.0, key="lux")
        c1, c2 = st.columns(2)
        c1.number_input("Curah hujan jam ini (mm)", min_value=0.0, max_value=100.0, step=0.1, key="hujan")
        c2.number_input("Prakiraan hujan 6 jam ke depan (mm)", min_value=0.0, max_value=200.0, step=0.1,
                        key="prakiraan")

        tombol = st.form_submit_button("Periksa", type="primary", width="stretch")

s = st.session_state                    # singkatan: s.k_now = isi kolom "Saat ini", dst.


# ---------- 5. Validasi masukan ----------
def cek_masukan():
    """Kembalikan daftar masalah pada masukan. Daftar kosong = masukan aman dipakai."""
    masalah = []
    if s.k_now is None:
        masalah.append("Kelembapan tanah saat ini wajib diisi.")
        return masalah
    for nama, nilai in [("saat ini", s.k_now), ("1 jam lalu", s.k_1j), ("3 jam lalu", s.k_3j)]:
        # 0 dan >= 100 adalah kode error sensor yang ditemukan saat memeriksa data
        if nilai is not None and (nilai <= 0 or nilai >= 100):
            masalah.append(f"Kelembapan tanah {nama} ({nilai:g}%) tidak wajar. Nilai 0 atau 100% ke atas "
                           "biasanya berarti sensor rusak atau terputus.")
    if not 5 <= s.suhu <= 40:
        masalah.append(f"Suhu udara {s.suhu:g}°C di luar rentang wajar untuk rumah kaca (5–40°C).")
    return masalah


# ---------- 6. Prediksi, ringkasan, dan alasan ----------
with kanan:
    st.subheader("Hasil")
    if not tombol:
        st.write("Isi data di sebelah kiri, lalu tekan **Periksa**.")
    else:
        masalah = cek_masukan()
        if masalah:
            st.warning("Saran tidak dapat diberikan karena ada masukan yang perlu diperiksa:\n\n"
                       + "\n".join(f"- {m}" for m in masalah))
            st.write("Periksa sensor atau petak secara langsung, lalu masukkan ulang datanya.")
        else:
            # Laju pengeringan. Jika pembacaan lama tidak diisi, model menerima nilai kosong (NaN).
            turun_1j = s.k_now - s.k_1j if s.k_1j is not None else float("nan")
            turun_3j = s.k_now - s.k_3j if s.k_3j is not None else float("nan")

            baris = pd.DataFrame([{
                "tanaman": s.tanaman, "jenis_tanah": s.tanah, "umur_tanaman_hari": s.umur,
                "kelembaban_tanah_pct": s.k_now, "suhu_udara_C": s.suhu, "kelembaban_udara_pct": s.rh,
                "cahaya_lux": s.lux, "curah_hujan_mm": s.hujan, "prakiraan_hujan_6jam_mm": s.prakiraan,
                "turun_1j": turun_1j, "turun_3j": turun_3j,
            }])[FITUR]

            peluang = model.predict_proba(baris)[0, 1]      # peluang "perlu siram" menurut model
            perlu_siram = peluang >= THRESHOLD
            batas = ATURAN[s.tanaman]

            with st.container(border=True):
                if perlu_siram:
                    st.markdown("### :red[Perlu disiram sekarang]")
                else:
                    st.markdown("### :green[Belum perlu disiram]")
                st.caption(f"{s.tanaman.capitalize()} di tanah {s.tanah}")

            # Ringkasan angka
            m1, m2, m3 = st.columns(3)
            m1.metric("Kelembapan tanah", f"{s.k_now:.1f}%", f"{s.k_now - batas:+.1f} dari batas {batas}%",
                      delta_color="off", delta_arrow="off")
            m2.metric("Prakiraan hujan 6 jam", f"{s.prakiraan:.1f} mm", f"ditunda jika ≥ {ATURAN['hujan']} mm",
                      delta_color="off", delta_arrow="off")
            m3.metric("Keyakinan model", f"{peluang:.0%}", f"siram jika ≥ {THRESHOLD:.0%}",
                      delta_color="off", delta_arrow="off")

            # Alasan disusun dari pola keputusan agronom yang ditemukan di data latih
            alasan = []
            if s.k_now < batas:
                alasan.append(f"Kelembapan tanah {s.k_now:.1f}% sudah di bawah batas yang biasa dipakai agronom "
                              f"untuk {s.tanaman}, yaitu sekitar {batas}%.")
            else:
                alasan.append(f"Kelembapan tanah {s.k_now:.1f}% masih di atas batas untuk {s.tanaman} "
                              f"(sekitar {batas}%).")
            if s.prakiraan >= ATURAN["hujan"]:
                alasan.append(f"Diperkirakan turun hujan {s.prakiraan:.1f} mm dalam 6 jam ke depan. Penyiraman "
                              "sebaiknya ditunda agar air tidak terbuang.")
            elif s.k_now < batas:
                alasan.append("Tidak ada prakiraan hujan yang cukup untuk membasahi tanah.")
            if s.k_1j is not None and s.k_1j - s.k_now >= 1:
                alasan.append(f"Tanah mengering cukup cepat, turun {s.k_1j - s.k_now:.1f} poin dalam satu jam "
                              "terakhir.")
            if perlu_siram != (s.k_now < batas and s.prakiraan < ATURAN["hujan"]):
                alasan.append("Keputusan model berbeda dari pola umum di atas karena model juga menimbang jenis "
                              "tanah, cuaca, dan laju pengeringan. Sebaiknya petak dicek langsung.")

            st.markdown("**Alasan**")
            st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(alasan, 1)))
            if s.k_1j is None or s.k_3j is None:
                st.caption("Pembacaan 1 jam dan 3 jam lalu tidak diisi, sehingga laju pengeringan tidak ikut "
                           "dihitung.")


# ---------- 7. Tentang model ----------
if META:
    st.divider()
    st.subheader("Tentang model")
    k1, k2 = st.columns([3, 2], gap="large")
    with k1:
        tes = META["metrik"]["test"]
        tabel = pd.DataFrame([
            {"Metode": "Baseline wajib: siram jika kelembapan < 35", **tes["baseline"]},
            {"Metode": "Aturan per tanaman + cek prakiraan hujan", **tes["aturan"]},
            {"Metode": f"Model {META['nama_model']}", **tes["model"]},
        ]).rename(columns={"f1": "F1", "precision": "Precision", "recall": "Recall"})
        st.markdown(f"**Hasil pada data test** ({META['pembagian_data']['test']}, jam 05–17)")
        st.dataframe(tabel, hide_index=True, width="stretch")
        st.caption("F1 menyeimbangkan precision (notifikasi siram yang benar) dan recall (petak kering yang "
                   "tertangkap). Data test hanya dievaluasi satu kali setelah model dan threshold dipilih "
                   "memakai data validation.")
    with k2:
        st.markdown("**Cara kerja singkat**")
        st.markdown(
            f"- Dilatih dari keputusan agronom periode {META['pembagian_data']['train']}.\n"
            f"- Threshold {META['threshold']:.2f} dipilih di data validation "
            f"({META['pembagian_data']['validation']}).\n"
            "- Batas yang biasa dipakai agronom: "
            + ", ".join(f"{t} {META['aturan_alasan'][t]}%" for t in ["cabai", "tomat", "selada"])
            + f"; penyiraman ditunda jika prakiraan hujan ≥ {META['aturan_alasan']['hujan']} mm.")
        st.markdown("**Keterbatasan**")
        st.markdown(
            "- Agronom hanya mencatat keputusan pukul 05–17, sehingga model belajar dari data siang hari.\n"
            "- Data hanya 3 bulan (Agustus–Oktober); pola musim lain belum terwakili.\n"
            "- Model meniru keputusan agronom, termasuk kesalahan yang mungkin ada di catatannya.")


st.divider()
st.caption("Saran ini membantu, bukan menggantikan, penilaian agronom.")