import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT_S = 60

st.set_page_config(page_title="Promy", page_icon="🧾", layout="wide")
st.title("🧾 Promy | Extraction sur images vers données structurées")
st.caption(
    "Image brute → données structurées | "
    "DET : RapidOCR (DBNet ONNX) | REC : PaddleOCR CRNN fine-tuné"
)

# ── Session state ──────────────────────────────────────────────────────────────
if "ocr_data" not in st.session_state:
    st.session_state.ocr_data = None
    st.session_state.ocr_elapsed = None
    st.session_state.ocr_filename = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres")
    conf_threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=(
            "Filtre les lignes dont la confiance est inférieure au seuil. "
            "Baisser le seuil affiche plus de lignes, au risque d'inclure des erreurs."
        ),
    )
    show_preprocessing = st.checkbox(
        "Détails preprocessing",
        value=False,
        help="Affiche les métadonnées produites (deskew, résolution).",
    )
    st.divider()
    st.caption(f"API : `{API_URL}`")

# ── Upload ────────────────────────────────────────────────────────────────────
MAX_SIZE_MB = 10

uploaded = st.file_uploader(
    "Déposez une facture",
    type=["jpg", "jpeg", "png"],
    help=f"Formats acceptés : JPG, PNG · Taille max : {MAX_SIZE_MB} Mo · 1 fichier à la fois",
)
st.caption(f"Formats : JPG, PNG · Taille max : {MAX_SIZE_MB} Mo · 1 fichier")

if not uploaded:
    st.info("Déposez une image ci-dessus pour démarrer l'extraction.")
    st.stop()

if uploaded.size > MAX_SIZE_MB * 1024 * 1024:
    st.error(
        f"Fichier trop volumineux ({uploaded.size / 1024 / 1024:.1f} Mo). "
        f"Limite : {MAX_SIZE_MB} Mo. Réduisez la résolution de l'image et réessayez."
    )
    st.stop()

# Nouvelle image → effacer les résultats précédents
if uploaded.name != st.session_state.ocr_filename:
    st.session_state.ocr_data = None
    st.session_state.ocr_elapsed = None
    st.session_state.ocr_filename = uploaded.name

# ── Layout deux colonnes ───────────────────────────────────────────────────────
col_img, col_result = st.columns([1, 1], gap="large")

with col_img:
    st.markdown("**Entrée : image brute**")
    st.image(uploaded, caption=uploaded.name, use_container_width=True)

with col_result:
    st.markdown("**Sortie : données structurées**")

    if st.button("Lancer l'OCR", type="primary", use_container_width=True):
        t0 = time.perf_counter()
        with st.spinner("Analyse en cours === comptez 5 à 15 secondes..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ocr",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=TIMEOUT_S,
                )
                if resp.status_code == 200:
                    st.session_state.ocr_data = resp.json()
                    st.session_state.ocr_elapsed = time.perf_counter() - t0
                else:
                    st.error(f"Erreur API {resp.status_code} : {resp.text}")
            except requests.exceptions.Timeout:
                st.error(
                    f"Délai dépassé ({TIMEOUT_S}s). "
                    "La facture est peut-être trop volumineuse : essayez avec une image plus petite."
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Impossible de joindre l'API ({API_URL}). "
                    "Vérifiez que le service api tourne."
                )

    # ── Résultats ──────────────────────────────────────────────────────────────
    data = st.session_state.ocr_data
    if data is not None:
        filtered = [
            (line, conf)
            for line, conf in zip(data["lines"], data["confidences"])
            if conf >= conf_threshold
        ]
        n_shown = len(filtered)
        n_hidden = len(data["lines"]) - n_shown

        st.success(
            f"{n_shown} lignes extraites · "
            f"confiance moyenne : {data['mean_confidence']:.3f} · "
            f"traitement : {st.session_state.ocr_elapsed:.1f}s"
            + (f" · {n_hidden} ligne(s) masquée(s)" if n_hidden else "")
        )

        if show_preprocessing:
            meta = data.get("preprocessing", {})
            st.info(
                f"Résolution originale : {meta.get('original_size')}  |  "
                f"Après preprocessing : {meta.get('processed_size')}  |  "
                f"Angle deskew corrigé : {meta.get('deskew_angle', 0):.2f}°"
            )

        tab_table, tab_json = st.tabs(["Données structurées", "Export JSON"])

        with tab_table:
            if filtered:
                df = pd.DataFrame({
                    "Ligne": range(1, n_shown + 1),
                    "Texte extrait": [line for line, _ in filtered],
                    "Confiance": [conf for _, conf in filtered],
                })
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Confiance": st.column_config.ProgressColumn(
                            "Confiance",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f",
                        )
                    },
                )
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Télécharger CSV",
                    data=csv,
                    file_name=f"{Path(uploaded.name).stem}_ocr.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Aucune ligne au-dessus du seuil de confiance. "
                    "Essayez de baisser le seuil dans la barre latérale."
                )

        with tab_json:
            st.json(data)
