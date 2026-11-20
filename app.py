# app.py — Aplikasi Irigasi Pintar untuk pengelola rumah kaca (Kasus 07)
# Jalankan di terminal:  py -3.12 -m streamlit run app.py

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Irigasi Pintar", layout="wide")


# ---------- 1. Memuat model ----------
@st.cache_resource                      # model cukup dimuat sekali
def muat_model():
    return joblib.load("model_irigasi.joblib")

paket = muat_model()
model = paket["model"]                  # pipeline: preprocessing + Gradient Boosting
THRESHOLD = paket["threshold"]          # batas peluang, dipilih di data validation
FITUR = paket["fitur"]                  # urutan kolom input model
ATURAN = paket["aturan"]                # batas kelembapan per tanaman + batas hujan (untuk alasan)


# ---------- 2. Judul ----------
st.title("Irigasi Pintar")
st.write("Masukkan data petak dan pembacaan sensor terbaru. Sistem akan menyarankan apakah petak "
         "perlu disiram sekarang, dengan mempertimbangkan prakiraan hujan.")

kiri, kanan = st.columns([3, 2], gap="large")


# ---------- 3. Form input ----------
with kiri:
    with st.form("input_petak"):
        st.subheader("Data petak")
        c1, c2, c3 = st.columns(3)
        tanaman = c1.selectbox("Tanaman", ["cabai", "tomat", "selada"])
        jenis_tanah = c2.selectbox("Jenis tanah", ["pasir", "lempung", "liat"])
        umur = c3.number_input("Umur tanaman (hari)", min_value=0, max_value=365, value=60)

        st.subheader("Kelembapan tanah (%)")
        c1, c2, c3 = st.columns(3)
        # value=None → kolom boleh kosong; kekosongan dicek di bawah
        k_now = c1.number_input("Saat ini", min_value=0.0, max_value=150.0, value=None, step=0.1,
                                placeholder="contoh: 38.5")
        k_1j = c2.number_input("1 jam lalu (opsional)", min_value=0.0, max_value=150.0, value=None, step=0.1)
        k_3j = c3.number_input("3 jam lalu (opsional)", min_value=0.0, max_value=150.0, value=None, step=0.1)

        st.subheader("Cuaca")
        c1, c2, c3 = st.columns(3)
        suhu = c1.number_input("Suhu udara (°C)", min_value=-10.0, max_value=60.0, value=20.0, step=0.1)
        rh = c2.number_input("Kelembapan udara (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        lux = c3.number_input("Cahaya (lux)", min_value=0.0, max_value=150000.0, value=30000.0, step=1000.0)
        c1, c2 = st.columns(2)
        hujan = c1.number_input("Curah hujan jam ini (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        prakiraan = c2.number_input("Prakiraan hujan 6 jam ke depan (mm)", min_value=0.0, max_value=200.0,
                                    value=0.0, step=0.1)

        tombol = st.form_submit_button("Periksa", type="primary", width="stretch")


# ---------- 4. Validasi masukan ----------
def cek_masukan():
    """Kembalikan daftar masalah pada masukan. Daftar kosong = masukan aman dipakai."""
    masalah = []
    if k_now is None:
        masalah.append("Kelembapan tanah saat ini wajib diisi.")
        return masalah
    for nama, nilai in [("saat ini", k_now), ("1 jam lalu", k_1j), ("3 jam lalu", k_3j)]:
        # 0 dan >= 100 adalah kode error sensor yang ditemukan saat memeriksa data
        if nilai is not None and (nilai <= 0 or nilai >= 100):
            masalah.append(f"Kelembapan tanah {nama} ({nilai:g}%) tidak wajar. Nilai 0 atau 100% ke atas "
                           "biasanya berarti sensor rusak atau terputus.")
    if not 5 <= suhu <= 40:
        masalah.append(f"Suhu udara {suhu:g}°C di luar rentang wajar untuk rumah kaca (5–40°C).")
    return masalah


# ---------- 5. Prediksi dan alasan ----------
with kanan:
    st.subheader("Hasil")
    if not tombol:
        st.write("Isi data di sebelah kiri, lalu tekan **Periksa**.")
        st.stop()

    masalah = cek_masukan()
    if masalah:
        st.warning("Saran tidak dapat diberikan karena ada masukan yang perlu diperiksa:\n\n"
                   + "\n".join(f"- {m}" for m in masalah))
        st.write("Periksa sensor atau petak secara langsung, lalu masukkan ulang datanya.")
        st.stop()

    # Laju pengeringan. Jika pembacaan lama tidak diisi, model menerima nilai kosong (NaN).
    turun_1j = k_now - k_1j if k_1j is not None else float("nan")
    turun_3j = k_now - k_3j if k_3j is not None else float("nan")

    baris = pd.DataFrame([{
        "tanaman": tanaman, "jenis_tanah": jenis_tanah, "umur_tanaman_hari": umur,
        "kelembaban_tanah_pct": k_now, "suhu_udara_C": suhu, "kelembaban_udara_pct": rh,
        "cahaya_lux": lux, "curah_hujan_mm": hujan, "prakiraan_hujan_6jam_mm": prakiraan,
        "turun_1j": turun_1j, "turun_3j": turun_3j,
    }])[FITUR]

    peluang = model.predict_proba(baris)[0, 1]      # peluang "perlu siram" menurut model
    perlu_siram = peluang >= THRESHOLD

    with st.container(border=True):
        if perlu_siram:
            st.markdown("### :red[Perlu disiram sekarang]")
        else:
            st.markdown("### :green[Belum perlu disiram]")
        st.caption(f"Tingkat keyakinan model {peluang:.0%}. Sistem menyarankan siram jika di atas {THRESHOLD:.0%}.")

    # Alasan disusun dari pola keputusan agronom yang ditemukan di data latih
    batas = ATURAN[tanaman]
    alasan = []
    if k_now < batas:
        alasan.append(f"Kelembapan tanah {k_now:.1f}% sudah di bawah batas yang biasa dipakai agronom untuk "
                      f"{tanaman}, yaitu sekitar {batas}%.")
    else:
        alasan.append(f"Kelembapan tanah {k_now:.1f}% masih di atas batas untuk {tanaman} (sekitar {batas}%).")

    if prakiraan >= ATURAN["hujan"]:
        alasan.append(f"Diperkirakan turun hujan {prakiraan:.1f} mm dalam 6 jam ke depan. Penyiraman sebaiknya "
                      "ditunda agar air tidak terbuang.")
    elif k_now < batas:
        alasan.append("Tidak ada prakiraan hujan yang cukup untuk membasahi tanah.")

    if k_1j is not None and k_1j - k_now >= 1:
        alasan.append(f"Tanah mengering cukup cepat, turun {k_1j - k_now:.1f} poin dalam satu jam terakhir.")

    sesuai_aturan = k_now < batas and prakiraan < ATURAN["hujan"]
    if perlu_siram != sesuai_aturan:
        alasan.append("Keputusan model berbeda dari pola umum di atas karena model juga menimbang jenis tanah, "
                      "cuaca, dan laju pengeringan. Sebaiknya petak dicek langsung.")

    st.markdown("**Alasan**")
    st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(alasan, 1)))

    if k_1j is None or k_3j is None:
        st.caption("Pembacaan 1 jam dan 3 jam lalu tidak diisi, sehingga laju pengeringan tidak ikut dihitung.")


st.divider()
st.caption("Saran ini dibuat oleh model yang mempelajari keputusan agronom pada Agustus–September 2025. "
           "Keputusan akhir tetap berada di tangan agronom.")