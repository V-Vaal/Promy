"""
Module de preprocessing pour les images de factures.

Genere depuis NB2.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def to_grayscale_lab(img_bgr):
    """Convertit en niveaux de gris via le canal L de l'espace LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    channel_l, _, _ = cv2.split(lab)
    return channel_l


def resize_to_width(gray_img, target_width=2480):
    """Redimensionne en conservant le ratio et une largeur cible fixe."""
    height, width = gray_img.shape[:2]
    if width == target_width:
        return gray_img

    scale = target_width / width
    new_height = int(height * scale)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(gray_img, (target_width, new_height), interpolation=interpolation)


def apply_clahe(gray_img, clip_limit=2.0, tile_size=(8, 8)):
    """Applique CLAHE pour renforcer le contraste local."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def deskew(gray_img, angle_threshold=0.5):
    """Corrige l'inclinaison de l'image si l'angle detecte est significatif."""
    # Binarisation par seuillage Otsu pour isoler les traits sombres sur fond clair
    _, binary = cv2.threshold(
        gray_img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Detection des segments de droite predominants dans l'image binarisee
    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=gray_img.shape[1] // 4,  # segments au moins 1/4 de la largeur
        maxLineGap=10,
    )
    if lines is None:
        return gray_img, 0.0

    # Calcul de l'angle de chaque segment ; on conserve uniquement les angles proches
    # de l'horizontale (< 15 deg) pour eviter de confondre bords verticaux et inclinaison
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return gray_img, 0.0

    # La mediane est plus robuste que la moyenne face aux faux segments
    median_angle = np.median(angles)
    if abs(median_angle) < angle_threshold:
        # Inclinaison negligeable : on renvoie l'image intacte
        return gray_img, median_angle

    # Rotation de correction centree sur l'image, remplissage par replication des bords
    height, width = gray_img.shape
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        gray_img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, median_angle


def denoise(gray_img, kernel_size=3):
    """Applique un filtre median leger pour reduire le bruit."""
    return cv2.medianBlur(gray_img, kernel_size)


def _load_image_for_line_crops(img_or_path):
    """Charge une image depuis un chemin, une PIL.Image ou un array numpy."""
    if isinstance(img_or_path, (str, Path)):
        img_bgr = cv2.imread(str(img_or_path))
        if img_bgr is None:
            raise ValueError(f"Impossible de lire : {img_or_path}")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return img_bgr, gray

    if isinstance(img_or_path, Image.Image):
        rgb = np.array(img_or_path.convert("RGB"))
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return img_bgr, gray

    if isinstance(img_or_path, np.ndarray):
        if img_or_path.ndim == 2:
            gray = img_or_path.copy()
            img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return img_bgr, gray
        if img_or_path.ndim == 3 and img_or_path.shape[2] == 3:
            img_bgr = img_or_path.copy()
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            return img_bgr, gray

    raise TypeError("img_or_path doit etre un chemin, une PIL.Image ou un np.ndarray")


def extract_text_line_crops(img_or_path, min_line_height=15, padding=4, return_heights=False):
    """
    Extrait des lignes de texte par projection horizontale.

    Returns:
        list[PIL.Image] ou tuple(list[PIL.Image], list[int])
    """
    img_bgr, gray = _load_image_for_line_crops(img_or_path)
    height, width = gray.shape

    # Binarisation puis dilatation horizontale pour fusionner les mots d'une meme ligne
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, width // 10), 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Projection horizontale : nombre de pixels actifs par ligne de l'image
    horizontal_projection = np.sum(dilated, axis=1)

    crops = []
    heights = []
    in_line = False
    line_start = 0

    # Parcours ligne par ligne pour detecter le debut et la fin de chaque bande de texte
    for y, value in enumerate(horizontal_projection):
        if not in_line and value > 0:
            in_line = True
            line_start = y
        elif in_line and value == 0:
            in_line = False
            line_height = y - line_start
            if line_height >= min_line_height:
                # Ajout d'un padding vertical pour ne pas couper les ascendantes/descendantes
                y0 = max(0, line_start - padding)
                y1 = min(height, y + padding)
                crop_rgb = cv2.cvtColor(img_bgr[y0:y1, :], cv2.COLOR_BGR2RGB)
                crops.append(Image.fromarray(crop_rgb))
                heights.append(line_height)

    # Cas limite : derniere ligne sans transition vers zero
    if in_line and height - line_start >= min_line_height:
        y0 = max(0, line_start - padding)
        crop_rgb = cv2.cvtColor(img_bgr[y0:height, :], cv2.COLOR_BGR2RGB)
        crops.append(Image.fromarray(crop_rgb))
        heights.append(height - line_start)

    if return_heights:
        return crops, heights
    return crops


def crop_to_padded_square(pil_img, size=384, bg=(255, 255, 255)):
    """Place le crop redimensionne dans un carre blanc sans deformer le ratio."""
    image = pil_img.convert("RGB")
    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError("Image vide recue dans crop_to_padded_square")

    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (size, size), bg)
    offset = ((size - new_width) // 2, (size - new_height) // 2)
    canvas.paste(resized, offset)
    return canvas


def preprocess_invoice(img_path, target_width=2480):
    """
    Pipeline complet de preprocessing pour une image de facture.

    Returns:
        gray_processed (np.ndarray)
        metadata (dict)
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Impossible de lire : {img_path}")

    original_height, original_width = img_bgr.shape[:2]

    # 1. Conversion en niveaux de gris via le canal L (LAB) : meilleur rendu que BGR direct
    gray = to_grayscale_lab(img_bgr)
    # 2. Normalisation de la resolution pour homogeneiser les inputs du modele
    gray = resize_to_width(gray, target_width)
    # 3. Rehaussement du contraste local (CLAHE) avant deskew pour des segments plus nets
    gray = apply_clahe(gray)
    # 4. Correction de l'inclinaison
    gray, angle = deskew(gray)
    # 5. Reduction du bruit residuel
    gray = denoise(gray)

    metadata = {
        "original_size": (original_width, original_height),
        "processed_size": (gray.shape[1], gray.shape[0]),
        "deskew_angle": angle,
    }
    return gray, metadata
