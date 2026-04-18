from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from api.preprocess.image import preprocess_image_bytes
from api.vendor import TextRecognizer, init_args, load_config

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = DEPLOYMENT_ROOT / "models" / "rec_infer"


def _get_rotate_crop_image(image: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    pts = np.array(points, dtype="float32")
    crop_width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    crop_height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if crop_width <= 0 or crop_height <= 0:
        return None
    destination_points = np.array(
        [[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]],
        dtype=np.float32,
    )
    transform_matrix = cv2.getPerspectiveTransform(pts, destination_points)
    crop = cv2.warpPerspective(
        image,
        transform_matrix,
        (crop_width, crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if crop.shape[0] / max(crop.shape[1], 1) >= 1.5:
        crop = np.rot90(crop)
    return crop


def _resolve_rec_image_shape(rec_model_dir: Path) -> str:
    config = load_config(str(rec_model_dir / "inference.yml"))
    for transform_op in config.get("PreProcess", {}).get("transform_ops", []):
        resize_config = transform_op.get("RecResizeImg")
        if resize_config and resize_config.get("image_shape"):
            return ",".join(str(value) for value in resize_config["image_shape"])
    return "3,32,100"


def _build_local_infer_args(use_gpu: bool = False):
    parser = init_args()
    args = parser.parse_args([])
    args.use_gpu = use_gpu
    args.enable_mkldnn = False
    args.show_log = False
    args.warmup = False
    args.rec_algorithm = "CRNN"
    args.rec_model_dir = str(MODEL_DIR)
    args.rec_char_dict_path = str(MODEL_DIR / "en_dict.txt")
    args.rec_image_shape = _resolve_rec_image_shape(MODEL_DIR)
    args.rec_batch_num = 12
    return args


def load_ocr_model() -> tuple[RapidOCR, TextRecognizer]:
    detector = RapidOCR()
    recognizer = TextRecognizer(_build_local_infer_args(use_gpu=False))
    print("OCR chargé | DET: RapidOCR (ONNX embarqué) | REC: TextRecognizer vendorisé aligné NB3")
    return detector, recognizer


def predict_ocr(engines: tuple[RapidOCR, TextRecognizer], image_bytes: bytes) -> dict:
    detector, recognizer = engines
    gray_image, metadata = preprocess_image_bytes(image_bytes)
    image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    detection_result, _ = detector(image_bgr)

    lines: list[str] = []
    confidences: list[float] = []
    for item in detection_result or []:
        if not item or len(item) < 1:
            continue
        crop = _get_rotate_crop_image(image_bgr, np.asarray(item[0], dtype="float32"))
        if crop is None:
            continue
        prediction, _ = recognizer([crop])
        text, score = prediction[0] if prediction else ("", 0.0)
        if text.strip():
            lines.append(text.strip())
            confidences.append(round(float(score), 4))

    mean_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    return {
        "lines": lines,
        "confidences": confidences,
        "mean_confidence": mean_confidence,
        "n_segments": len(lines),
        "preprocessing": {
            "original_size": list(metadata["original_size"]),
            "processed_size": list(metadata["processed_size"]),
            "deskew_angle": round(float(metadata["deskew_angle"]), 3),
        },
    }
