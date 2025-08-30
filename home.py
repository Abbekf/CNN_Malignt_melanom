# home.py  — Start page / Overview
from pathlib import Path
import io
import streamlit as st
from streamlit.components.v1 import html


# Måste ligga tidigt
st.set_page_config(page_title="Malignt melanom", layout="wide", initial_sidebar_state="collapsed")
html('<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">', height=0)

# --- MOBIL ULTRA-COMPACT OVERRIDES ---
st.markdown("""
<style>
/* Mindre grundtyp på mobil (gör allt kompaktare) */
@media (max-width: 820px){
  html { font-size: 14px !important; }                      /* sänker basen */
  .block-container { padding: 10px !important; }            /* mindre sidopadding */
  section.main > div { max-width: 100% !important; }        /* låt innehåll använda hela bredden */
}

/* Rubriker (tvinga ned storleken ordentligt) */
h1, .stMarkdown h1, [data-testid="stMarkdownContainer"] h1 {
  font-size: clamp(1.2rem, 4.2vw, 1.7rem) !important;
  line-height: 1.15 !important;
  margin: 0.35rem 0 0.5rem 0 !important;
}
h2, .stMarkdown h2, [data-testid="stMarkdownContainer"] h2 {
  font-size: clamp(1.05rem, 3.6vw, 1.35rem) !important;
  line-height: 1.2 !important;
  margin: 0.4rem 0 0.45rem 0 !important;
}
h3, .stMarkdown h3 {
  font-size: clamp(0.95rem, 3.2vw, 1.2rem) !important;
}

/* Brödtext & listor */
p, .stMarkdown p, [data-testid="stMarkdownContainer"] p, li {
  font-size: clamp(0.95rem, 3.3vw, 1.02rem) !important;
  line-height: 1.45 !important;
  margin: 0.35rem 0 !important;
}

/* Kort/containers med mycket luft: pressa ned padding */
[data-testid="stVerticalBlock"] > div { padding: 0.6rem !important; }

/* Knappar / inputs i full bredd på mobil */
@media (max-width: 820px){
  .stButton>button, .stDownloadButton>button,
  .stTextInput input, .stNumberInput input, .stSelectbox, .stFileUploader {
    width: 100% !important;
  }
}

/* Bilder & diagram: fyll bredd men inte mer än skärmen höjdmässigt */
.stImage img, .stPyplot, .stPlotlyChart, .stAltairChart {
  width: 100% !important; height: auto !important; max-height: 80vh !important;
}

/* Tabs: gör listan scroll-bar istället för att trycka ihop titlar */
[data-baseweb="tab-list"] {
  overflow-x: auto !important;
  flex-wrap: nowrap !important;
  scrollbar-width: none;
}
[data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
[data-baseweb="tab"] { min-width: max-content !important; }

/* Tab-panel: lite tajtare padding */
[data-baseweb="tab-panel"] { padding: 0.5rem 0 0 0 !important; }

/* Tabeller – horisontell scroll vid behov */
[data-testid="stDataFrame"] { overflow: auto hidden; }

/* Dölj Streamlit-menyer på mobil för ren vy (frivilligt) */
@media (max-width: 820px){ #MainMenu, footer { display: none !important; } }
</style>
""", unsafe_allow_html=True)



# ---------------- Page config ----------------
st.set_page_config(
    page_title="Malignt Melanom – Start",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CSS ----------------
css_path = Path(__file__).resolve().parent / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ---------------- Model meta (optional, cached) ----------------
# We try to load the model once to show learned params/layers on the start page.
MODEL_PATH = Path(__file__).resolve().parent / "exported_models" / "keras_tuner_best_finetuned.h5"

@st.cache_resource(show_spinner=False)
def _load_model_meta():
    try:
        from tensorflow import keras  # lazy import so start page stays fast if TF not installed
        model = keras.models.load_model(str(MODEL_PATH))
        param_count = model.count_params()
        layer_count = len(model.layers)
        return int(param_count), int(layer_count)
    except Exception:
        return None, None

param_count, layer_count = _load_model_meta()

# ---------------- HERO ----------------
st.markdown(
    """
<div class="mdc-hero">
  <h1>Malignt Melanom – Klassificeringsprojekt</h1>
  <p>CNN-baserad klassificering (studieprojekt). Byggt av Abbe – för demo och utbildning. <b>Ej för medicinskt bruk.</b></p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------- Quick KPIs ----------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Dataset (Kaggle)", "≈ 10 000 bilder")
with k2:
    st.metric("Keras-Tuner tid", "13,5 timmar")
with k3:
    st.metric("Lärda parametrar", f"{param_count:,}" if param_count else "—")
with k4:
    st.metric("AUC", "0.976")

st.markdown("---")

# ---------------- Intro + Navigation ----------------
cL, cR = st.columns([1.2, 1])

with cL:
    st.subheader("Vad är detta?")
    st.write(
        """
Det här är en pedagogisk demo som visar hur ett Convolutional Neural Network (CNN) kan klassificera
hudförändringar som **benigna** eller **maligna**. Projektet fokuserar på hela ML-flödet:
*data → träning → utvärdering → inferens (live)*.
        """
    )

    st.subheader("Hur använder jag sidan?")
    st.markdown(
        """
- **Klassificerare** – ladda upp en bild och se modellens prediktion live.  
- **Utvärdering** – utforska resultat: *Loss*, *Accuracy*, *Confusion Matrix*, *ROC-kurva*, *UMAP*.  
- **Tips för egna foton** – croppa så att fläcken fyller bilden, använd jämnt ljus och undvik hår/skuggor.
        """
    )

    # Direct links to subpages (works in modern Streamlit)
    try:
        st.page_link("pages/1_Klassificerare.py", label="🩺 Öppna Klassificerare", icon="🩺")
        st.page_link("pages/2_Utvardering.py", label="📊 Se Utvärdering", icon="📊")
    except Exception:
        st.info("Använd sidomenyn till vänster för att öppna **Klassificerare** och **Utvärdering**.")

with cR:
    import streamlit as st
    from urllib.parse import quote_plus

    st.subheader("📱 Dela appen med QR-kod")
    st.caption("Skanna QR-koden nedan för att öppna appen direkt:")

    app_url = "https://abbekf-cnn-malignt-melanom-home-dpiqho.streamlit.app/"  # byt till din riktiga URL

    qr_img_url = f"https://api.qrserver.com/v1/create-qr-code/?size=360x360&data={quote_plus(app_url)}"

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.image(qr_img_url, caption="Öppna appen genom att scanna QR-koden", use_container_width=True)
        st.markdown(f"[Öppna appen här]({app_url})")

    st.markdown("---")

# ---------------- Project details ----------------
st.subheader("Projekt i korthet")
st.markdown(
    """
- **Data**: Nedladdat **Kaggle-dataset (~10 000 bilder)** med hudförändringar.  
- **Träning & tuning**: Keras-Tuner kördes i **13,5 timmar** för att hitta en bra konfiguration;
  bästa modellen finjusterades (transfer learning).  
- **Inferens**: Appen använder en **låst inputstorlek 224×224 px** och beräknar sannolikhet för *malign*.  
- **Syfte**: Visa ML-flödet och ge ett pedagogiskt demo – **inte** att ersätta klinisk diagnostik.
"""
)

# ---------------- What you can do here ----------------
st.subheader("Vad kan du göra här?")
st.markdown(
    """
- **Testa egna bilder** (helst croppade så fläcken fyller bilden) och se hur sannolikheten påverkas.  
- **Justera beslutsgränsen (threshold)** i klassificeraren och visa hur *precision/recall-trade-off* ändras.  
- **Visa träningskurvor** (Loss/Accuracy) för att diskutera under/överanpassning.  
- **Förklara modellen** med *Confusion Matrix*, *Classification report* och *ROC-AUC*.  
- **Illustrera representationer** med *UMAP* (hur modellen separerar klasser).
"""
)

# ---------------- Limitations ----------------
st.subheader("Begränsningar att känna till")
st.markdown(
    """
- Träningsdata består främst av **dermatoskopiska** bilder. **Mobilbilder** med varierande ljus/bakgrund kan ge missvisande resultat.  
- Sannolikheten är **inte** en medicinsk riskbedömning; den anger hur mycket bilden liknar datasetets *maligna* exempel.  
- Modellen är ett **studieprojekt** och ska inte användas för kliniska beslut.
"""
)

# ---------------- Roadmap / Future work ----------------
st.subheader("Förbättringar jag vill göra")
st.markdown(
    """
- **Datadiversitet**: Lägg till fler **mobilbilder** + olika hudtoner/ljussättningar (domain adaptation).  
- **Aktiv inlärning**: Låt användare flagga osäkra fall → bygg kurering/aktiv inlärning för nästa träningsrunda.  
- **Implementera** *transfer learning* (t.ex. EfficientNet/ResNet) för att kunna hantera egna mobilbilder bättre.  
    Mobilbilder skiljer sig ofta från Kaggle-datat (ljus, vinkel, bakgrund), och förtränade nätverk kan därför förbättra generaliseringen.
- **MLOps**: Versionera data/modeller, automatisera utvärdering och export (t.ex. DVC + GitHub Actions).
"""
)

# ---------------- Tech stack ----------------
st.subheader("Tech-stack")
st.markdown(
    """
**Python · TensorFlow/Keras · Keras-Tuner · scikit-learn · NumPy · Matplotlib**  
**UMAP-learn** (för visualisering) · **Streamlit** (UI)  
"""
)

# ---------------- Disclaimer ----------------
st.warning("⚠️ Endast för utbildning/demo. Kontakta sjukvården vid oro över hudförändringar.")
