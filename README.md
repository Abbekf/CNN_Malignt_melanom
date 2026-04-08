# Malignt Melanom – CNN-klassificering

Studieprojekt som visar hur ett CNN kan klassificera hudförändringar som **benigna** eller **maligna**.  
Byggt med TensorFlow/Keras och presenterat via en Streamlit-app.

> ⚠️ **Endast för utbildning/demo – inte för medicinskt bruk.**

---

## Projektstruktur

| Fil / Mapp | Beskrivning |
|---|---|
| `home.py` | Startsida (Streamlit) med projektöversikt och KPI:er |
| `pages/1_Klassificerare.py` | Live-klassificerare med Score-CAM-förklaring |
| `pages/2_utvärdering.py` | Utvärderingssida: Loss, Accuracy, ROC, Confusion Matrix, UMAP |
| `ui_mobile.py` | Responsiv CSS-hjälpmodul (mobil/desktop) |
| `style.css` | Holo-MedTech-tema (mörkt futuristiskt UI) |
| `main.ipynb` | Notebook: egen CNN + Keras-Tuner hyperparameteroptimering |
| `transfer_efficient.ipynb` | Notebook: transfer learning med EfficientNetB3 (2-fas) |
| `exported_models/` | Sparade modeller, vikter och tröskelkonfiguration |
| `figures/` | Träningskurvor, ROC, Confusion Matrix, UMAP m.m. |
| `images1/` | Dataset (train/test, benign/malignant) |
| `kt_tuning/` | Keras-Tuner-resultat (hyperband-trials) |

## Modeller

### 1. Egen CNN (`main.ipynb`)
- Arkitektur: Conv2D → MaxPooling → GlobalAveragePooling → Dense
- Input: **224×224 px**
- Hyperparametertuning med **Keras-Tuner (Hyperband)** – ~13,5 timmar
- Tröskeloptimering via Youden's J-statistik

### 2. EfficientNetB3 Transfer Learning (`transfer_efficient.ipynb`)
- Förtränad backbone (ImageNet), 2-fas träning (frusen → finjusterad)
- Input: **300×300 px**
- Exporterad i flera format: `.keras`, `.h5`, SavedModel

## Streamlit-app

Appen använder modellen `exported_models/keras_tuner_best_finetuned.h5` och har tre sidor:

1. **Startsida** – projektöversikt, KPI:er, QR-kod
2. **Klassificerare** – ladda upp en bild, se prediktion + Score-CAM heatmap
3. **Utvärdering** – träningskurvor, ROC, Confusion Matrix, Classification Report, UMAP

### Kör lokalt

```bash
pip install -r requirements.txt
streamlit run home.py
```

## Tech-stack

- **Python** · **TensorFlow/Keras** · **Keras-Tuner** · **scikit-learn** · **NumPy** · **Matplotlib**
- **UMAP-learn** (visualisering) · **Streamlit** (UI)

## Resultat

| Metrik | Värde |
|---|---|
| AUC | 0.976 |
| Dataset | ~10 000 bilder (Kaggle) |
| Keras-Tuner träningstid | 13,5 timmar |
