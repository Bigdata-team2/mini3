import torch
import yolov5
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from PIL import Image
import io
import warnings
import pathlib

# PyTorch 2.6+ 호환성 설정
torch.serialization.add_safe_globals(['models.yolo.DetectionModel'])
warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")

# pathlib 호환성 패치 (WindowsPath -> PosixPath)
class PatchedPath(pathlib.PurePath):
    def __new__(cls, *args, **kwargs):
        if cls is pathlib.WindowsPath:
            return pathlib.PosixPath(*args, **kwargs)
        return super().__new__(cls, *args, **kwargs)

# pathlib 모듈 패치
pathlib.WindowsPath = PatchedPath

# torch.load 함수 패치
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

app = FastAPI(title="Mini3 - Ingredient Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLOv5 모델 로드
MODEL_PATH = "../models/best.pt"

try:
    yolo_model = yolov5.load(MODEL_PATH)
    yolo_model.conf = 0.25
    yolo_model.iou = 0.45
    CLASS_NAMES = yolo_model.names
    print("✅ YOLOv5 모델 로드 완료:", MODEL_PATH)
    print("📋 클래스 목록:", CLASS_NAMES)
except Exception as e:
    print("❌ YOLO 모델 로드 실패:", e)
    raise RuntimeError(f"YOLO 모델을 로드할 수 없습니다: {e}")

def run_yolo_inference(image_bytes: bytes, conf_threshold: float = 0.25) -> Dict[str, Any]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = yolo_model(img, size=640)
    
    detections = []
    unique_ingredients = []
    
    predictions = results.pred[0]
    if len(predictions) > 0:
        for pred in predictions:
            x1, y1, x2, y2, conf, cls = pred.cpu().numpy()
            if conf >= conf_threshold:
                cls_id = int(cls)
                label = CLASS_NAMES.get(cls_id, str(cls_id))
                
                detections.append({
                    "label": label,
                    "confidence": round(float(conf), 4),
                    "box_xyxy": [float(x1), float(y1), float(x2), float(y2)]
                })
                unique_ingredients.append(label)

    # 중복 제거
    unique_ingredients = list(dict.fromkeys(unique_ingredients))
    
    return {
        "ingredients": unique_ingredients,
        "detections": detections,
        "count": len(detections)
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일을 업로드하세요.")
    
    try:
        image_bytes = await image.read()
        result = run_yolo_inference(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "class_count": len(CLASS_NAMES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)