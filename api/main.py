# main.py
# ------------------------------------------------------------
# FastAPI + YOLO(best.pt) 추론 + (옵션) Chroma 레시피 검색
# - 비전 파트: ultralytics 만 사용 (YOLOv5 레포 import X)
# - 모델파일: ./best.pt (동일 경로)
# - 엔드포인트:
#   1) POST /predict              : 이미지 → 재료 인식 결과(JSON)
#   2) POST /recipes/search       : 재료 리스트 → 레시피 질의
#   3) POST /predict-and-recommend: 이미지 → 재료 인식 → 레시피 추천
# ------------------------------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from PIL import Image
import io
from python_multipart import multipart  # Add python-multipart import

# ✅ 비전: YOLO(best.pt만 사용) - ultralytics
from ultralytics import YOLO

# ✅ 레시피 검색: ChromaDB (벡터 임베딩은 sentence-transformers)
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# -----------------------------
# 0) 앱 & CORS
# -----------------------------
app = FastAPI(title="Mini3 - Ingredient Detector & Recipe Search", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 개발 단계 전부 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 1) YOLO 모델 로드 (pt 외 의존 X)
# -----------------------------
MODEL_PATH = "models/best.pt"  # 같은 폴더에 둔다
try:
    yolo_model = YOLO(MODEL_PATH)  # ultralytics가 내부적으로 가중치만 사용
    CLASS_NAMES = yolo_model.names  # id→label 매핑
    print("✅ YOLO 모델 로드 완료:", MODEL_PATH)
    print("📋 클래스 목록:", CLASS_NAMES)
except Exception as e:
    print("❌ YOLO 모델 로드 실패:", e)
    yolo_model = None
    CLASS_NAMES = {}

# -----------------------------
# 2) ChromaDB 초기화 (옵션)
#    - persist_directory: 로컬에 영속 저장(개발용)
#    - 임베딩: all-MiniLM-L6-v2
#    - collection 이름: recipes
#    - 사전 구축은 별도 스크립트에서 upsert 해두기!
# -----------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)

chroma_client = chromadb.Client(
    Settings(
        anonymized_telemetry=False,
        persist_directory="./chroma_data"  # 폴더 자동 생성
    )
)

# 존재하면 가져오고, 없으면 새로 만든다
try:
    recipes_col = chroma_client.get_or_create_collection(
        name="recipes",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    print("✅ Chroma 컬렉션 준비 완료: recipes")
except Exception as e:
    print("❌ Chroma 컬렉션 생성 실패:", e)
    recipes_col = None

# -----------------------------
# 3) 스키마
# -----------------------------
class RecipeSearchIn(BaseModel):
    ingredients: List[str]
    top_k: int = 5

# -----------------------------
# 4) 유틸: YOLO 추론 → 결과 정리
# -----------------------------
def run_yolo_inference(image_bytes: bytes, conf_threshold: float = 0.25) -> Dict[str, Any]:
    if yolo_model is None:
        raise RuntimeError("YOLO model is not loaded.")

    # PIL 로 로드 (RGB)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ultralytics 예측 (single-image)
    # 반환: list[ultralytics.engine.results.Results]
    results = yolo_model.predict(source=img, conf=conf_threshold, verbose=False)
    result = results[0]

    detections = []
    unique_ingredients = []

    # 박스, cls, conf 추출
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = CLASS_NAMES.get(cls_id, str(cls_id))
            xyxy = [float(v) for v in box.xyxy[0].tolist()]  # [x1,y1,x2,y2]

            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "box_xyxy": xyxy
            })
            unique_ingredients.append(label)

    # 중복 제거 순서 유지
    seen = set()
    unique_ingredients = [x for x in unique_ingredients if not (x in seen or seen.add(x))]

    return {
        "ingredients": unique_ingredients,
        "detections": detections,
        "count": len(detections)
    }

# -----------------------------
# 5) 엔드포인트: 이미지 → 재료 인식
#     Flutter에서 멀티파트로 'image' 키로 업로드하면 됨
# -----------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일을 업로드하세요.")

    try:
        image_bytes = await image.read()
        out = run_yolo_inference(image_bytes)
        # Flask 호환 간단 형태(ingredients만)도 포함해 반환
        return {
            "ingredients": out["ingredients"],
            "detections": out["detections"],  # UI에서 박스/확률 쓰고 싶으면 사용
            "count": out["count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류: {e}")

# -----------------------------
# 6) 엔드포인트: 재료 리스트 → 레시피 검색 (Chroma)
#     - 사전 upsert된 recipes 컬렉션에 질의
#     - metadata 예시: { "recipe_id": "...", "title": "...", "ingredients": "...", ... }
# -----------------------------
@app.post("/recipes/search")
def search_recipes(body: RecipeSearchIn):
    if recipes_col is None:
        raise HTTPException(status_code=500, detail="Chroma 컬렉션이 준비되지 않았습니다.")
    if not body.ingredients:
        return {"results": []}

    # 질의 텍스트 단순 결합 (예: "egg, tomato, onion")
    query_text = ", ".join(body.ingredients)

    try:
        q = recipes_col.query(
            query_texts=[query_text],
            n_results=body.top_k,
            include=["metadatas", "distances", "documents", "ids"]
        )
        # Chroma 포맷 → 간단 리스트로 변환
        hits = []
        for i in range(len(q["ids"][0])):
            hits.append({
                "id": q["ids"][0][i],
                "distance": float(q["distances"][0][i]) if q.get("distances") else None,
                "metadata": q["metadatas"][0][i],
                "document": q["documents"][0][i] if q.get("documents") else None
            })
        return {"query": query_text, "results": hits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"레시피 검색 오류: {e}")

# -----------------------------
# 7) 엔드포인트: 이미지 → 재료 인식 → 레시피 추천(원샷)
# -----------------------------
@app.post("/predict-and-recommend")
async def predict_and_recommend(image: UploadFile = File(...), top_k: int = 5):
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일을 업로드하세요.")
    if recipes_col is None:
        raise HTTPException(status_code=500, detail="Chroma 컬렉션이 준비되지 않았습니다.")

    try:
        image_bytes = await image.read()
        det = run_yolo_inference(image_bytes)
        ingredients = det["ingredients"]

        if not ingredients:
            return {
                "ingredients": [],
                "recommendations": []
            }

        # Chroma 질의
        query_text = ", ".join(ingredients)
        q = recipes_col.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["metadatas", "distances", "documents", "ids"]
        )
        hits = []
        for i in range(len(q["ids"][0])):
            hits.append({
                "id": q["ids"][0][i],
                "distance": float(q["distances"][0][i]) if q.get("distances") else None,
                "metadata": q["metadatas"][0][i],
                "document": q["documents"][0][i] if q.get("documents") else None
            })

        return {
            "ingredients": ingredients,
            "detections": det["detections"],
            "recommendations": hits
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {e}")

# -----------------------------
# 8) 로컬 실행
# -----------------------------
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload