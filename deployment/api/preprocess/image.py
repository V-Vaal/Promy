import os
import sys
import tempfile
from pathlib import Path

# preprocessing.py est à la racine de deployment/ (source unique partagée avec NB2)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocessing import preprocess_invoice  # noqa: E402


def preprocess_image_bytes(image_bytes: bytes) -> tuple:
    """Décode les bytes image et applique le pipeline preprocessing documenté dans NB2.

    Returns:
        gray (np.ndarray), metadata (dict) avec original_size, processed_size, deskew_angle
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        gray, metadata = preprocess_invoice(tmp_path)
    finally:
        os.unlink(tmp_path)
    return gray, metadata
