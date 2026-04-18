from pydantic import BaseModel
from typing import List


class PreprocessingMeta(BaseModel):
    original_size: List[int]
    processed_size: List[int]
    deskew_angle: float


class OCRResponse(BaseModel):
    lines: List[str]
    confidences: List[float]
    mean_confidence: float
    n_segments: int
    preprocessing: PreprocessingMeta
