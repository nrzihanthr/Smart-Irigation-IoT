# app.py — Aplikasi web Irigasi Pintar untuk pengelola rumah kaca
# Cara menjalankan (di terminal VS Code):  streamlit run app.py

import joblib                # untuk membuka file model hasil latihan di Colab
import pandas as pd          # model menerima input berupa tabel (DataFrame)
import streamlit as st       # library untuk membuat halaman web dengan Python


# ---------- 1. Memuat model ----------
@st.cache_resource           # model cukup dimuat sekali, tidak setiap kali tombol ditekan
def muat_model():
    return joblib.load("model_irigasi.joblib")

paket = muat_model()
model = paket["model"]           # pipeline: preprocessing + Gradient Boosting
THRESHOLD = paket["threshold"]   # batas peluang yang dipilih di data validation
FITUR = paket["fitur"]           # urutan kolom yang diharapkan model
ATURAN = paket["aturan"]         # batas kelembapan per tanaman + batas hujan (untuk menyusun alasan)


# ---------- 2. Tampilan judul ----------
st.set_page_config(page_title="Irigasi Pintar", page_icon="💧")
st.title("💧 Irigasi Pintar")
st.write("Isi data petak dan pembacaan sensor, lalu sistem menyarankan apakah petak perlu disiram **sekarang**.")


# ---------- 3. Form input ----------
with st.form("input_petak"):     # form: hasil baru dihitung saat tombol ditekan
    st.subheader("Data petak")
    c1, c2, c3 = st.columns(3)
    tanaman = c1.selectbox("Tanaman", ["cabai", "tomat", "selada"])
    jenis_tanah = c2.selectbox("Jenis tanah", ["pasir", "lempung", "liat"])
    umur = c3.number_input("Umur tanaman (hari)", 0, 200, 60)

    st.subheader("Kelembapan tanah (%)")
    c1, c2, c3 = st.columns(3)
    k_now = c1.number_input("Sekarang", 0.0, 100.0, 40.0, step=0.1)
    k_1j = c2.number_input("1 jam lalu", 0.0, 100.0, 40.5, step=0.1)
    k_3j = c3.number_input("3 jam lalu", 0.0, 100.0, 41.5, step=0.1)

    st.subheader("Cuaca")
    c1, c2, c3 = st.columns(3)
    suhu = c1.number_input("Suhu udara (°C)", 0.0, 45.0, 20.0, step=0.1)
    rh = c2.number_input("Kelembapan udara (%)", 0.0, 100.0, 85.0, step=1.0)
    lux = c3.number_input("Cahaya (lux)", 0.0, 120000.0, 30000.0, step=1000.0)
    c1, c2 = st.columns(2)
    hujan = c1.number_input("Curah hujan jam ini (mm)", 0.0, 50.0, 0.0, step=0.1)
    prakiraan = c2.number_input("Prakiraan hujan 6 jam ke depan (mm)", 0.0, 50.0, 0.0, step=0.1)

    tombol = st.form_submit_button("Cek sekarang")


# ---------- 4. Prediksi + alasan ----------
if tombol:
    # Pembacaan 0 atau >= 100 adalah error sensor (ditemukan saat memeriksa data)
    if not all(0 < k < 100 for k in [k_now, k_1j, k_3j]):
        st.error("⚠️ Pembacaan sensor kelembapan tanah tidak wajar (0 atau ≥ 100%). "
                 "Kemungkinan sensor rusak. Silakan cek petak secara manual.")
        st.stop()               # hentikan di sini, jangan beri saran dari data rusak

    # Susun satu baris input dengan nama kolom yang SAMA dengan saat latihan
    baris = pd.DataFrame([{
        "tanaman": tanaman, "jenis_tanah": jenis_tanah, "umur_tanaman_hari": umur,
        "kelembaban_tanah_pct": k_now, "suhu_udara_C": suhu, "kelembaban_udara_pct": rh,
        "cahaya_lux": lux, "curah_hujan_mm": hujan, "prakiraan_hujan_6jam_mm": prakiraan,
        "turun_1j": k_now - k_1j, "turun_3j": k_now - k_3j,
    }])[FITUR]

    peluang = model.predict_proba(baris)[0, 1]    # peluang kelas 1 ("perlu siram")
    perlu_siram = peluang >= THRESHOLD

    if perlu_siram:
        st.error("💧 **SIRAM SEKARANG**")
    else:
        st.success("✅ **Belum perlu disiram**")
    st.caption(f"Keyakinan model: {peluang:.0%} (disarankan siram jika ≥ {THRESHOLD:.0%})")

    # Alasan disusun dari pola keputusan agronom yang ditemukan di data
    st.subheader("Alasan")
    batas = ATURAN[tanaman]
    if k_now < batas:
        st.write(f"- Kelembapan tanah **{k_now:.1f}%** sudah di bawah batas yang biasa dipakai agronom "
                 f"untuk {tanaman} (sekitar **{batas}%**).")
    else:
        st.write(f"- Kelembapan tanah **{k_now:.1f}%** masih di atas batas untuk {tanaman} "
                 f"(sekitar **{batas}%**).")
    if prakiraan >= ATURAN["hujan"]:
        st.write(f"- Prakiraan hujan **{prakiraan:.1f} mm** dalam 6 jam ke depan. "
                 "Agronom biasanya menunda penyiraman supaya air tidak terbuang.")
    if k_now - k_1j <= -1:
        st.write(f"- Tanah mengering cepat: turun **{k_1j - k_now:.1f} poin** dalam 1 jam terakhir.")
    if perlu_siram != (k_now < batas and prakiraan < ATURAN["hujan"]):
        st.write("- Catatan: keputusan model berbeda dari aturan sederhana di atas karena model juga "
                 "mempertimbangkan faktor lain (jenis tanah, cuaca, laju pengeringan). "
                 "Sebaiknya cek petak secara langsung.")

st.divider()
st.caption("Model dilatih dari keputusan agronom (Agustus–Oktober 2025). "
           "Saran ini membantu, bukan menggantikan, penilaian agronom.")
