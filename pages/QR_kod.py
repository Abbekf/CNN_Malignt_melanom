# pages/QR_kod.py
# ---------------------------------------------------------
# Simple QR page for your deployed Streamlit app.
# UI language: Swedish; comments: English.
# Requires: qrcode[pil] in requirements.txt
# ---------------------------------------------------------

import io
import qrcode
import streamlit as st
from PIL import Image

st.set_page_config(page_title="QR-kod", layout="centered")

st.title("📱 QR-kod till min app")

# --- Put your deployed URL here once you have it ---
DEFAULT_URL = "https://din-app-url.streamlit.app"  # <-- ersätt efter deploy

# text input so you can change the link anytime
app_url = st.text_input("Länk till appen:", value=DEFAULT_URL)

# Size slider
box_size = st.slider("Storlek (pixlar per ruta)", 6, 20, 10, help="Hur tät QR-koden är")
border = st.slider("Marginal (rutor)", 2, 8, 4)

# Generate button
if st.button("Generera QR-kod"):
    # Build QR object
    qr = qrcode.QRCode(
        version=None,               # automatic size
        error_correction=qrcode.constants.ERROR_CORRECT_M,  # good balance
        box_size=box_size,
        border=border,
    )
    qr.add_data(app_url.strip())
    qr.make(fit=True)

    # Create image (black on white)
    img: Image.Image = qr.make_image(fill_color="black", back_color="white")

    # Show in app
    st.image(img, caption="Skanna mig med kameran 📷", use_container_width=False)

    # Offer download as PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.download_button(
        label="Ladda ner QR-kod (PNG)",
        data=buf.getvalue(),
        file_name="qr_kod_streamlit_app.png",
        mime="image/png",
    )

st.info("Tips: Lägg QR-koden på första sidan i din presentation eller skriv ut den för publiken.")
