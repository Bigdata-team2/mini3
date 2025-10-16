# 필요한 라이브러리 임포트
import torch
import sys
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from PIL import Image
import io
import warnings

# YOLOv7 경로 추가
sys.path.append('/Users/shindongeun/bigdata_lacture/team_project/mini3/yolov7')

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

# 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

# FastAPI 앱 생성 및 CORS 미들웨어 설정
app = FastAPI(title="Mini3 - Ingredient Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLOv7 모델 파일 경로 설정
MODEL_PATH = "/Users/shindongeun/bigdata_lacture/team_project/mini3/models/yolov7_custom2/weights/best.pt"
IMG_SIZE = 640
IOU_THRESHOLD = 0.45

# YOLOv7 모델 로드 및 설정
try:
    device = select_device('')  # cuda 또는 cpu 자동 선택
    yolo_model = attempt_load(MODEL_PATH, map_location=device)
    yolo_model.eval()
    CLASS_NAMES = yolo_model.names
    
    print("✅ YOLOv7 모델 로드 완료:", MODEL_PATH)
    print("📋 클래스 목록:", CLASS_NAMES)
    print(f"🖥️  디바이스: {device}")
except Exception as e:
    print("❌ YOLO 모델 로드 실패:", e)
    raise RuntimeError(f"YOLO 모델을 로드할 수 없습니다: {e}")


# YOLO 모델을 사용하여 이미지에서 객체 감지를 수행하는 함수
def run_yolo_inference(image_bytes: bytes, conf_threshold: float = 0.25) -> Dict[str, Any]:
    """
    YOLOv7 모델로 이미지에서 재료 감지
    """
    # PIL 이미지로 변환
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img0 = np.array(img_pil)  # 원본 이미지 (RGB)
    
    # 전처리: letterbox로 리사이즈 + 패딩
    img = letterbox(img0, IMG_SIZE, stride=32, auto=True)[0]
    
    # RGB to BGR, HWC to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    
    # Tensor 변환 및 정규화
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # 추론 수행
    with torch.no_grad():
        pred = yolo_model(img, augment=False)[0]
    
    # NMS (Non-Maximum Suppression) 적용
    pred = non_max_suppression(pred, conf_threshold, IOU_THRESHOLD, classes=None, agnostic=False)
    
    # 결과 처리
    detections = []
    unique_ingredients = []
    
    for det in pred:  # 배치의 각 이미지
        if len(det):
            # 좌표를 원본 이미지 크기로 스케일링
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            for *xyxy, conf, cls in det:
                cls_id = int(cls)
                label = CLASS_NAMES[cls_id]
                
                # 감지된 객체 정보 저장
                detections.append({
                    "label": label,
                    "confidence": round(float(conf), 4),
                    "box_xyxy": [float(x) for x in xyxy]
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
        "model_version": "YOLOv7",
        "device": str(device), 
        "class_count": len(CLASS_NAMES),
        "classes": CLASS_NAMES 
    }

# 직접 실행 시 uvicorn 서버 구동
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
