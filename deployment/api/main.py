from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from api.models.ocr_model import load_ocr_model, predict_ocr
from api.schemas import OCRResponse

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["ocr"] = load_ocr_model()
    yield
    models.clear()


app = FastAPI(
    title="Promy OCR API",
    description=(
        "Pipeline OCR factures — DET : RapidOCR (DBNet ONNX) | "
        "REC : PaddleOCR CRNN fine-tuné sur 1 413 factures Kaggle. "
        "CER proxy validation : 0.19% | Architecture REC : MobileNetV3 + BiLSTM + CTC"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_route(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Format accepté : JPG ou PNG")
    image_bytes = await file.read()
    result = predict_ocr(models["ocr"], image_bytes)
    return OCRResponse(**result)
