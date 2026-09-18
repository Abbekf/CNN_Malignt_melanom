from pathlib import Path
import io
import numpy as np
from PIL import Image
import streamlit as st

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None

if tf is None:
    st.warning("Modellen är avstängd i moln-deploy (ingen TensorFlow). Demo visar bara UI.")
    st.stop()


MODEL_PATH = Path(__file__).resolve().parent.parent / "exported_models" / "keras_tuner_best_finetuned.h5"
CLASS_NAMES = ["benign", "malignant"]
FIXED_INPUT_SIZE = (224, 224)

st.set_page_config(page_title="Klassificerare", layout="wide", initial_sidebar_state="expanded")

css_path = Path(__file__).resolve().parents[1] / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

def _preprocess_pil(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """PIL -> normaliserat batch-tensor [1,H,W,3] i [0,1]."""
    img = img.convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def _predict(model: keras.Model, img_tensor: np.ndarray) -> float:
    """Returnera sannolikhet för klass index 1 = 'malignant'."""
    pred = model.predict(img_tensor, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] == 1:
        return float(pred[0, 0])
    if pred.ndim == 2 and pred.shape[1] == 2:
        return float(pred[0, 1])
    if pred.ndim == 1 and pred.shape[0] >= 1:
        return float(pred[0])
    raise ValueError(f"Unexpected model output shape: {pred.shape}")

def _pct(x: float) -> str:
    return f"{x*100:.1f}%"

@st.cache_resource(show_spinner=True)
def load_model_fixed() -> keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Hittar inte modellen:\n{MODEL_PATH}")
    # H5-filen innehåller en gammal träningskonfiguration som inte behövs för inferens.
    return keras.models.load_model(str(MODEL_PATH), compile=False)

def _candidate_conv_layers(model: keras.Model):
    """Lista alla lager som producerar 4D feature maps (oavsett lagertyp)."""
    names = []
    for layer in model.layers:
        out = getattr(layer, "output", None)
        if out is None:
            continue
        for o in tf.nest.flatten(out):
            try:
                if len(o.shape) == 4:
                    names.append(layer.name)
                    break
            except Exception:
                pass
    return names

def _score_cam(model: keras.Model, img_tensor: np.ndarray, layer_name: str,
               class_index: int = 1, max_channels: int = 32) -> np.ndarray:
    """
    Score-CAM (förenklad):
      1) Hämta feature maps från valt lager
      2) Normalisera varje kanal och skala upp till inputstorlek
      3) Använd varje karta som mask på indata och kör prediktion
      4) Viktad summa av kartor med respektive klass-score
    """
    # 1) Feature maps
    conv_layer = model.get_layer(layer_name)
    fmap_model = keras.Model(inputs=model.inputs, outputs=conv_layer.output)
    fmap = fmap_model(tf.convert_to_tensor(img_tensor), training=False)
    if isinstance(fmap, (list, tuple)):
        fmap = fmap[0]
    fmap = tf.convert_to_tensor(fmap)
    if len(fmap.shape) != 4 or fmap.shape[-1] is None:
        raise RuntimeError(f"Lagret {layer_name} gav en ogiltig feature-map: {fmap.shape}")
    c = int(tf.shape(fmap)[-1])

    use_c = min(c, max_channels)
    fmap = fmap[:, :, :, :use_c]
    fmap_min = tf.reduce_min(fmap, axis=(1, 2), keepdims=True)
    fmap_max = tf.reduce_max(fmap, axis=(1, 2), keepdims=True)
    denom = tf.where(fmap_max - fmap_min > 1e-9, fmap_max - fmap_min, tf.ones_like(fmap_max))
    fmap_norm = (fmap - fmap_min) / denom
    fmap_up = tf.image.resize(fmap_norm[0], size=FIXED_INPUT_SIZE, method="bilinear")
    fmap_up = tf.clip_by_value(fmap_up, 0.0, 1.0)

    img = tf.convert_to_tensor(img_tensor[0])
    masks = tf.transpose(fmap_up, [2, 0, 1])[:, :, :, None]
    img_batch = img[None, ...]
    masked_batch = img_batch * masks

    # Direkt modellanrop fungerar även med Keras 3:s nya predict-adapter.
    preds = model(masked_batch, training=False)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = tf.convert_to_tensor(preds)

    if preds.shape.rank == 2 and preds.shape[-1] >= 2:
        scores = preds[:, class_index]
    elif preds.shape.rank == 2 and preds.shape[-1] == 1:
        scores = preds[:, 0]
    elif preds.shape.rank == 1:
        scores = preds
    else:
        flat = tf.reshape(preds, (tf.shape(preds)[0], -1))
        scores = flat[:, 0]

    weighted = tf.einsum("chw, c -> hw", tf.transpose(fmap_up, [2, 0, 1]), scores)
    weighted = tf.nn.relu(weighted)
    weighted = weighted / (tf.reduce_max(weighted) + 1e-8)
    return weighted.numpy()

def _overlay_heatmap_on_pil(pil_img: Image.Image, heatmap: np.ndarray,
                            intensity: float = 0.45, cmap: str = "jet") -> Image.Image:
    """Färglägg heatmap med matplotlib colormap och blanda med originalbilden."""
    import matplotlib.cm as cm
    heat_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    colored = cm.colormaps.get_cmap(cmap)(heat_uint8)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    heat_pil = Image.fromarray(colored).resize(pil_img.size, resample=Image.BILINEAR)
    return Image.blend(pil_img.convert("RGB"), heat_pil, alpha=float(intensity))

st.title("🩺 Live-klassificerare")
st.caption(
    "Den här sidan använder **endast** modellen `exported_models/keras_tuner_best_finetuned.h5`."
    " Ladda upp en bild för att få ett resultat. **Inputstorlek:** 224×224 px (låst)."
)

with st.sidebar:
    debug = st.toggle("Visa debug-info", value=False)

    try:
        model = load_model_fixed()
        st.write(f"Lärda parametrar: **{model.count_params():,}**")
        st.write(f"Lager: **{len(model.layers)}**")
    except Exception as e:
        st.error(str(e)); st.stop()

    thr = st.slider("Beslutsgräns (malign om sannolikhet ≥ gräns)", 0.05, 0.95, 0.50, 0.01)

    st.markdown("---")
    st.markdown("**Modellens inputstorlek:** 224 × 224 px (låst)")

    st.markdown("---")
    st.markdown("### Bildvisning")
    fit_to_container = st.toggle("Skala till sidans bredd", value=False)
    preview_width = st.slider("Förhandsgranskningens bredd (px)", 320, 1400, 900, step=10, disabled=fit_to_container)
    st.caption("Endast visningsstorlek. Modellen använder alltid 224×224 för inferens.")

    st.markdown("---")
    st.markdown("### Score-CAM (förklaring)")
    cam_intensity = st.slider("Heatmap-intensitet", 0.05, 0.95, 0.45, 0.05)
    cam_map = st.selectbox("Färgskala", ["jet", "magma", "inferno", "plasma", "viridis"], index=3)
    max_chan = st.slider("Antal kanaler (fart vs. detalj)", 8, 64, 32, step=8)
    st.caption("Fler kanaler = tydligare karta men långsammare.")

st.divider()

uploaded = st.file_uploader(
    "Ladda upp en bild (JPG/PNG). Bilden beskärs inte – endast skalas.",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)
if uploaded is None:
    st.info("📷 Välj en bild för att köra klassificering.")
    st.stop()

# Förhandsgranskning
try:
    pil_img = Image.open(io.BytesIO(uploaded.read()))
except Exception:
    st.error("Kunde inte läsa bilden. Försök med en JPG/PNG."); st.stop()

kwargs = {"use_container_width": True} if fit_to_container else {"width": int(preview_width)}
st.markdown('<div class="img-frame">', unsafe_allow_html=True)
st.image(pil_img, caption="Förhandsgranskning", **kwargs)
st.markdown('</div>', unsafe_allow_html=True)

# Prediktion
with st.spinner("Klassificerar..."):
    x = _preprocess_pil(pil_img, target_size=FIXED_INPUT_SIZE)
    p_mal = _predict(model, x)
    p_ben = 1.0 - p_mal
    is_mal = p_mal >= thr
    pred_label = CLASS_NAMES[1 if is_mal else 0]

st.subheader("Resultat")
c1, c2, c3 = st.columns(3)
c1.metric("Benign", _pct(p_ben))
c2.metric("Malignant", _pct(p_mal))
c3.metric("Beslutsgräns", _pct(thr))

if is_mal:
    st.error(f"**Prediktion: {pred_label.upper()}**  (≥ gräns)")
else:
    st.success(f"**Prediktion: {pred_label.capitalize()}**  (< gräns)")

st.caption(
    "Notera: Sannolikheterna kommer direkt från modellen. "
    "Beslutsgränsen (threshold) styr hur strikt vi är när vi kallar något 'malign'."
)

# ---- Score-CAM (alltid på) ----
try:
    candidates = _candidate_conv_layers(model)
    if not candidates:
        raise RuntimeError("Modellen innehåller inga 4D-lager att använda för CAM.")
    used_layer = candidates[-1]  # sista 4D-lagret (vanligtvis närmast klassificeraren)

    with st.spinner("Beräknar Score-CAM…"):
        heatmap = _score_cam(model, x, used_layer, class_index=1, max_channels=max_chan)

    overlay = _overlay_heatmap_on_pil(pil_img, heatmap, intensity=cam_intensity, cmap=cam_map)
    st.markdown(f"#### CAM (områden som bidrog mest till 'malignant')  \n*Lager:* `{used_layer}` · *Metod:* `Score-CAM`")
    oc1, oc2 = st.columns(2)
    with oc1:
        st.image(overlay, caption="Överlagrad heatmap", **kwargs)
    with oc2:
        hm_img = Image.fromarray(np.uint8(heatmap * 255)).resize(pil_img.size, Image.BILINEAR)
        st.image(hm_img, caption="Heatmap (gråskala, 0–255)", **kwargs)

    st.caption("Tips: Croppa så att fläcken fyller större del av bilden för tydligare aktiveringar.")
except Exception as e:
    st.info(f"CAM kunde inte beräknas: {e}")
    if debug:
        try:
            st.write("• 4D-kandidater (bakifrån):", list(reversed(_candidate_conv_layers(model))))
        except Exception as ee:
            st.write("• Kunde inte lista kandidater:", ee)
