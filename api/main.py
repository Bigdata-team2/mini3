# 필요한 라이브러리 임포트
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
# Windows 경로를 POSIX 경로로 변환하는 클래스
class PatchedPath(pathlib.PurePath):
    def __new__(cls, *args, **kwargs):
        if cls is pathlib.WindowsPath:
            return pathlib.PosixPath(*args, **kwargs)
        return super().__new__(cls, *args, **kwargs)

# pathlib 모듈의 WindowsPath를 PatchedPath로 대체
pathlib.WindowsPath = PatchedPath

# torch.load 함수를 패치하여 weights_only=False 옵션 추가
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# FastAPI 앱 생성 및 CORS 미들웨어 설정
app = FastAPI(title="Mini3 - Ingredient Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLOv5 모델 파일 경로 설정
MODEL_PATH = "../models/best.pt"

# YOLOv5 모델 로드 및 설정
try:
    yolo_model = yolov5.load(MODEL_PATH)
    yolo_model.conf = 0.25  # 신뢰도 임계값 설정
    yolo_model.iou = 0.45   # IoU 임계값 설정
    CLASS_NAMES = yolo_model.names
    print("✅ YOLOv5 모델 로드 완료:", MODEL_PATH)
    print("📋 클래스 목록:", CLASS_NAMES)
except Exception as e:
    print("❌ YOLO 모델 로드 실패:", e)
    raise RuntimeError(f"YOLO 모델을 로드할 수 없습니다: {e}")

# YOLO 모델을 사용하여 이미지에서 객체 감지를 수행하는 함수
def run_yolo_inference(image_bytes: bytes, conf_threshold: float = 0.25) -> Dict[str, Any]:
    # 바이트 데이터를 PIL 이미지로 변환
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # YOLO 모델로 추론 수행
    results = yolo_model(img, size=640)
    
    detections = []
    unique_ingredients = []
    
    # 예측 결과 처리
    predictions = results.pred[0]
    if len(predictions) > 0:
        for pred in predictions:
            x1, y1, x2, y2, conf, cls = pred.cpu().numpy()
            if conf >= conf_threshold:
                cls_id = int(cls)
                label = CLASS_NAMES.get(cls_id, str(cls_id))
                
                # 감지된 객체 정보 저장
                detections.append({
                    "label": label,
                    "confidence": round(float(conf), 4),
                    "box_xyxy": [float(x1), float(y1), float(x2), float(y2)]
                })
                unique_ingredients.append(label)

    # 중복 제거된 재료 목록 생성
    unique_ingredients = list(dict.fromkeys(unique_ingredients))
    
    return {
        "ingredients": unique_ingredients,
        "detections": detections,
        "count": len(detections)
    }

# 이미지 예측 엔드포인트
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # 이미지 파일 검증
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일을 업로드하세요.")
    
    try:
        # 이미지 읽기 및 추론 수행
        image_bytes = await image.read()
        result = run_yolo_inference(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류: {e}")

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "class_count": len(CLASS_NAMES)
    }

# 직접 실행 시 uvicorn 서버 구동
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)