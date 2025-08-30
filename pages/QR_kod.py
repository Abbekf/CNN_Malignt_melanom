# pages/QR_kod.py
# ---------------------------------------------------------
# Streamlit page that generates and lets the user download
# a QR code pointing to your deployed app (or any URL).
# UI text: Swedish. Code comments: English.
# Requires: qrcode[pil], pillow
# ---------------------------------------------------------

import io
import streamlit as st
from PIL import Image

# Try to import qrcode; fail gracefully if not installed
try:
    import qrcode
    from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
except Exception:
    qrcode = None

st.set_page_config(page_title="QR-kod", layout="centered")
st.title("📱 QR-kod till min app")

# If qrcode is missing, show a helpful error and stop the page
if qrcode is None:
    st.error(
        "QR-biblioteket saknas.\n\n"
        "Lägg till `qrcode[pil]` i `requirements.txt` och installera lokalt med:\n"
        "`pip install qrcode[pil] pillow`"
    )
    st.stop()

# --- Controls -----------------------------------------------------------
st.caption("Ange länken till din körande app nedan (eller valfri URL).")

# Replace this default after du har din riktiga Streamlit-URL
DEFAULT_URL = "https://din-app-url.streamlit.app"
app_url = st.text_input("Länk till appen:", value=DEFAULT_URL, placeholder="https://...")

col_a, col_b = st.columns(2)
with col_a:
    box_size = st.slider("Storlek (pixlar per ruta)", min_value=6, max_value=20, value=10, step=1)
    border = st.slider("Marginal (rutor)", min_value=2, max_value=8, value=4, step=1)
with col_b:
    ec_level = st.selectbox(
        "Felnivå (Error correction)",
        options=("M (standard)", "L (låg)", "Q (hög)", "H (max)"),
        index=0,
        help="Högre nivåer gör QR-koden mer robust men också tätare."
    )
    # Map UI selection to qrcode constants
    ec_map = {
        "L (låg)": ERROR_CORRECT_L,
        "M (standard)": ERROR_CORRECT_M,
        "Q (hög)": ERROR_CORRECT_Q,
        "H (max)": ERROR_CORRECT_H,
    }
    ec_value = ec_map[ec_level]

# Colors (keep defaults clean and printer-friendly)
col_c1, col_c2 = st.columns(2)
with col_c1:
    fg = st.color_picker("Färg (förgrund)", "#000000")
with col_c2:
    bg = st.color_picker("Bakgrund", "#FFFFFF")

# --- Generate button ----------------------------------------------------
generate = st.button("Generera QR-kod", type="primary", use_container_width=True)

# --- Build & display QR image ------------------------------------------
if generate:
    url = (app_url or "").strip()
    if not url:
        st.warning("Skriv in en giltig URL först.")
        st.stop()

    # Build QR object with selected parameters
    qr = qrcode.QRCode(
        version=None,                   # auto-size
        error_correction=ec_value,
        box_size=box_size,
        border=border,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # Create PIL image
    img: Image.Image = qr.make_image(fill_color=fg, back_color=bg)

    # Show result
    st.image(img, caption="Skanna mig med kameran 📷")

    # Offer download
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.download_button(
        label="Ladda ner QR-kod (PNG)",
        data=buf.getvalue(),
        file_name="qr_kod_streamlit_app.png",
        mime="image/png",
        use_container_width=True,
    )

st.info(
    "Tips: Lägg QR-koden på första sidan i presentationen eller skriv ut den. "
    "Testa alltid med flera telefoner i olika ljus."
)
