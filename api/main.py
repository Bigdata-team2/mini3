#------------------------------------------------------------------------------
# 1. 필요한 라이브러리 임포트
#------------------------------------------------------------------------------
# 기본 라이브러리
import torch  # 딥러닝 프레임워크
import sys    # 시스템 경로 조작
import cv2    # 이미지 처리
import numpy as np  # 수치 연산
import logging  # 로깅
import io      # 입출력 처리
import warnings  # 경고 메시지 제어

# FastAPI 관련
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 기타 유틸리티
from typing import List, Dict, Any
from PIL import Image

#------------------------------------------------------------------------------
# 2. YOLOv7 관련 설정
#------------------------------------------------------------------------------
# YOLOv7 모듈 경로 추가
sys.path.append('/Users/shindongeun/bigdata_lacture/team_project/mini3/yolov7')

# YOLOv7 모듈 임포트
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

#------------------------------------------------------------------------------
# 3. 로깅 설정
#------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PyTorch 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)

#------------------------------------------------------------------------------
# 4. 기본 설정값 정의
#------------------------------------------------------------------------------
# YOLOv7 설정
DEFAULT_IMG_SIZE = 640     # 입력 이미지 크기
DEFAULT_CONF_THRESH = 0.25 # 객체 검출 신뢰도 임계값
DEFAULT_IOU_THRESH = 0.45  # NMS IOU 임계값

# 클래스명 한글 매핑
ENGLISH_TO_KOREAN = {
    "cab": "배추", "cab2": "양배추", "car": "당근",
    "cuc": "오이", "egg": "달걀", "gar": "마늘",
    "lee": "대파", "oni": "양파", "pork": "돼지고기",
    "pota": "감자", "rad": "무"
}

def translate_class_name(english_name: str) -> str:
    """영어 클래스명을 한국어로 변환"""
    return ENGLISH_TO_KOREAN.get(english_name, english_name)

#------------------------------------------------------------------------------
# 5. FastAPI 앱 설정
#------------------------------------------------------------------------------
# FastAPI 인스턴스 생성
app = FastAPI(title="Mini3 - Ingredient Detector", version="1.0.0")

# CORS 미들웨어 설정 (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#------------------------------------------------------------------------------
# 6. 모델 초기화
#------------------------------------------------------------------------------
# 모델 파일 경로
MODEL_PATH = "/Users/shindongeun/bigdata_lacture/team_project/mini3/models/yolov7_custom2/weights/best.pt"

# YOLOv7 모델 로드
try:
    logger.info("YOLOv7 모델 로딩 시작...")
    device = select_device('')  # GPU/CPU 자동 선택
    yolo_model = attempt_load(MODEL_PATH, map_location=device)
    yolo_model.eval()
    
    # GPU 사용시 FP16 설정
    half = device.type != 'cpu'
    if half:
        yolo_model.half()
        logger.info("Half precision (FP16) 활성화")
    
    # 모델 설정 가져오기
    stride = int(yolo_model.stride.max())
    from utils.general import check_img_size
    DEFAULT_IMG_SIZE = check_img_size(DEFAULT_IMG_SIZE, s=stride)
    CLASS_NAMES = yolo_model.names
    
    # 초기화 상태 로깅
    logger.info("✅ YOLOv7 모델 로드 완료")
    logger.info(f"📋 클래스 목록: {CLASS_NAMES}")
    logger.info(f"🖥️  디바이스: {device}")
    logger.info(f"📐 모델 stride: {stride}")
    logger.info(f"📏 이미지 크기: {DEFAULT_IMG_SIZE}")
    logger.info(f"⚙️  기본 설정값:")
    logger.info(f"   - img-size: {DEFAULT_IMG_SIZE}")
    logger.info(f"   - conf-thres: {DEFAULT_CONF_THRESH}")
    logger.info(f"   - iou-thres: {DEFAULT_IOU_THRESH}")
    logger.info(f"   - half precision: {half}")

except Exception as e:
    logger.error(f"❌ YOLO 모델 로드 실패: {e}")
    raise RuntimeError(f"YOLO 모델을 로드할 수 없습니다: {e}")

#------------------------------------------------------------------------------
# 7. 추론 함수 정의
#------------------------------------------------------------------------------
def run_yolo_inference(image_bytes: bytes, conf_threshold: float = DEFAULT_CONF_THRESH, 
                      img_size: int = DEFAULT_IMG_SIZE) -> Dict[str, Any]:
    """YOLOv7 모델로 이미지에서 재료를 감지하는 함수"""
    try:
        # 이미지 전처리
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img0 = np.array(img_pil)
        img = letterbox(img0, img_size, stride=stride, auto=True)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        # Tensor 변환 및 정규화
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # 모델 추론
        with torch.no_grad():
            pred = yolo_model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_threshold, DEFAULT_IOU_THRESH, 
                                 classes=None, agnostic=False)
        
        # 결과 처리
        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], 
                                        img0.shape).round()
                
                for *xyxy, conf, cls in det:
                    cls_id = int(cls)
                    english_label = CLASS_NAMES[cls_id]
                    korean_label = translate_class_name(english_label)
                    
                    detections.append({
                        "label": korean_label,
                        "label_en": english_label,
                        "confidence": float(conf),
                        "box_xyxy": [float(x) for x in xyxy],
                        "class_id": cls_id
                    })
        
        logger.info(f"감지된 객체 수: {len(detections)}")
        return {
            "raw_detections": detections,
            "count": len(detections)
        }
        
    except Exception as e:
        logger.error(f"이미지 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, 
                          detail=f"이미지 처리 중 오류 발생: {str(e)}")

#------------------------------------------------------------------------------
# 8. API 엔드포인트 정의
#------------------------------------------------------------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...), 
                 img_size: int = DEFAULT_IMG_SIZE,
                 conf_threshold: float = DEFAULT_CONF_THRESH):
    """이미지에서 재료를 감지하는 API 엔드포인트"""
    # 이미지 유효성 검사
    if not image.content_type or not image.content_type.startswith("image/"):
        logger.warning(f"잘못된 파일 타입: {image.content_type}")
        raise HTTPException(status_code=400, detail="이미지 파일을 업로드하세요.")
    
    try:
        logger.info(f"이미지 추론 요청 - img_size: {img_size}, conf_threshold: {conf_threshold}")
        image_bytes = await image.read()
        result = run_yolo_inference(image_bytes, conf_threshold=conf_threshold, 
                                  img_size=img_size)
        
        ingredients = list(set([det['label'] for det in result['raw_detections']]))
        logger.info(f"감지된 재료: {ingredients}")
        
        return {
            "ingredients": ingredients,
            "detections": result['raw_detections'],
            "count": result['count'],
            "classes": CLASS_NAMES
        }
    except Exception as e:
        logger.error(f"추론 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"추론 중 오류: {str(e)}")

@app.post("/predict-and-recommend")
async def predict_and_recommend(image: UploadFile = File(...), top_k: int = 5):
    """이미지에서 재료를 감지하고 레시피를 추천하는 API 엔드포인트"""
    # 이미지 유효성 검사
    if not image.content_type or not image.content_type.startswith("image/"):
        logger.warning(f"잘못된 파일 타입: {image.content_type}")
        raise HTTPException(status_code=400, detail="이미지 파일을 업로드하세요.")
    
    try:
        logger.info(f"레시피 추천 요청 - top_k: {top_k}")
        image_bytes = await image.read()
        result = run_yolo_inference(image_bytes)
        
        ingredients = list(set([det['label'] for det in result['raw_detections']]))
        recommendations = []  # 향후 구현 예정
        
        logger.info(f"감지된 재료: {ingredients}, 추천 레시피: {len(recommendations)}개")
        
        return {
            "ingredients": ingredients,
            "detections": result['raw_detections'],
            "count": result['count'],
            "recommendations": recommendations,
            "classes": CLASS_NAMES
        }
    except Exception as e:
        logger.error(f"추론 및 추천 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"추론 및 추천 중 오류: {str(e)}")

@app.get("/health")
async def health_check():
    """서버와 모델의 상태를 확인하는 헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "YOLOv7",
        "device": str(device),
        "class_count": len(CLASS_NAMES),
        "classes": CLASS_NAMES
    }

#------------------------------------------------------------------------------
# 9. 메인 실행
#------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("FastAPI 서버 시작...")
    uvicorn.run(app, host="0.0.0.0", port=8081)