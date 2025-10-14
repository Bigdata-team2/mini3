# 필요한 라이브러리 임포트
from fastapi import FastAPI, Request, HTTPException  # FastAPI 웹 프레임워크
from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어
from pydantic import BaseModel  # 데이터 검증을 위한 Pydantic
import chromadb  # 벡터 데이터베이스
from chromadb.config import Settings  # ChromaDB 설정
from sentence_transformers import SentenceTransformer  # 텍스트 임베딩 모델
import pandas as pd  # 데이터 처리
import os
from typing import List, Dict, Any  # 타입 힌팅

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="Recipe Search API", version="1.0.0")

# CORS 미들웨어 설정
# Flutter 앱에서 API 호출을 허용하기 위한 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# ChromaDB 클라이언트 설정
# 로컬 파일 시스템에 데이터를 영구 저장
client = chromadb.PersistentClient(
    path="./chroma_db",  # 데이터 저장 경로
    settings=Settings(
        anonymized_telemetry=False,  # 텔레메트리 비활성화
        allow_reset=True  # 컬렉션 리셋 허용
    )
)

# 문장 임베딩을 위한 모델 로드
# all-MiniLM-L6-v2: 다국어 지원, 빠른 속도, 적절한 성능의 경량 모델
model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB 컬렉션 생성/로드
# 코사인 유사도를 사용하는 HNSW 인덱스 설정
recipes_info = client.get_or_create_collection(
    "recipes_info",  # 레시피 기본 정보 컬렉션
    metadata={"hnsw:space": "cosine"}
)
recipes_steps = client.get_or_create_collection(
    "recipes_steps",  # 레시피 조리 단계 컬렉션
    metadata={"hnsw:space": "cosine"}
)
user_allergy = client.get_or_create_collection(
    "user_allergy",  # 사용자 알러지 정보 컬렉션
    metadata={"hnsw:space": "cosine"}
)

# Pydantic 모델 정의
# API 요청/응답의 데이터 구조와 유효성 검사를 위한 모델
class RecipeInfo(BaseModel):
    recipe_id: str  # 레시피 고유 ID
    요리명: str  # 레시피 이름
    요리별재료: str  # 재료 목록
    조리시간: str = ""  # 예상 조리 시간
    난이도: str = ""  # 레시피 난이도
    카테고리: str = ""  # 음식 카테고리

class RecipeStep(BaseModel):
    recipe_id: str  # 레시피 ID
    recipe_num: int  # 조리 단계 번호
    recipe_text: str  # 조리 방법 설명
    recipe_img_url: str = ""  # 조리 단계 이미지 URL

class UserAllergy(BaseModel):
    이메일: str  # 사용자 이메일
    알러지정보: str  # 알러지 정보 (쉼표로 구분)

class BuildChromaRequest(BaseModel):
    recipes_info: List[RecipeInfo]  # 레시피 기본 정보 목록
    recipes_steps: List[RecipeStep]  # 레시피 조리 단계 목록
    user_allergy: List[UserAllergy]  # 사용자 알러지 정보 목록

class SearchRequest(BaseModel):
    query: str  # 검색 쿼리
    top_k: int = 5  # 반환할 결과 수
    collection_type: str = "recipes_info"  # 검색할 컬렉션 타입

# 루트 엔드포인트: API 상태 확인
@app.get("/")
async def root():
    return {"message": "Recipe Search API", "status": "running"}

# 헬스체크 엔드포인트: 각 컬렉션의 데이터 수 반환
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "collections": {
            "recipes_info": recipes_info.count(),
            "recipes_steps": recipes_steps.count(),
            "user_allergy": user_allergy.count()
        }
    }

# ChromaDB 데이터 구축 엔드포인트
@app.post("/build_chroma/")
async def build_chroma(request: BuildChromaRequest):
    try:
        # 1. 레시피 기본 정보 저장
        if request.recipes_info:
            # DataFrame 생성 및 텍스트 결합
            df_info = pd.DataFrame([r.dict() for r in request.recipes_info])
            info_texts = (df_info["요리명"] + " " + df_info["요리별재료"]).tolist()
            # 텍스트 임베딩 생성
            info_embeddings = model.encode(info_texts, normalize_embeddings=True).tolist()
            
            # ChromaDB에 데이터 저장
            recipes_info.upsert(
                ids=df_info["recipe_id"].tolist(),
                documents=info_texts,
                metadatas=df_info.to_dict("records"),
                embeddings=info_embeddings
            )

        # 2. 레시피 조리 단계 저장
        if request.recipes_steps:
            df_steps = pd.DataFrame([r.dict() for r in request.recipes_steps])
            steps_text = df_steps["recipe_text"].tolist()
            step_embeddings = model.encode(steps_text, normalize_embeddings=True).tolist()
            
            recipes_steps.upsert(
                ids=[f"{rid}_{num}" for rid, num in zip(df_steps["recipe_id"], df_steps["recipe_num"])],
                documents=steps_text,
                metadatas=df_steps.to_dict("records"),
                embeddings=step_embeddings
            )

        # 3. 사용자 알러지 정보 저장
        if request.user_allergy:
            df_user = pd.DataFrame([u.dict() for u in request.user_allergy])
            allergy_texts = df_user["알러지정보"].tolist()
            allergy_embeddings = model.encode(allergy_texts, normalize_embeddings=True).tolist()
            
            user_allergy.upsert(
                ids=df_user["이메일"].tolist(),
                documents=allergy_texts,
                metadatas=df_user.to_dict("records"),
                embeddings=allergy_embeddings
            )

        # 성공 응답 반환
        return {
            "status": "success",
            "message": "ChromaDB 데이터 구축 완료",
            "counts": {
                "recipes_info": recipes_info.count(),
                "recipes_steps": recipes_steps.count(),
                "user_allergy": user_allergy.count()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 구축 실패: {str(e)}")

# 검색 엔드포인트
@app.post("/search/")
async def search(request: SearchRequest):
    try:
        # 검색할 컬렉션 선택
        collection_map = {
            "recipes_info": recipes_info,
            "recipes_steps": recipes_steps,
            "user_allergy": user_allergy
        }
        
        collection = collection_map.get(request.collection_type)
        if not collection:
            raise HTTPException(status_code=400, detail="잘못된 컬렉션 타입")

        # 검색 쿼리를 임베딩 벡터로 변환
        query_embedding = model.encode([request.query], normalize_embeddings=True).tolist()
        
        # 벡터 유사도 검색 실행
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=request.top_k,
            include=["metadatas", "distances", "documents", "ids"]
        )
        
        # 검색 결과 포맷팅
        hits = []
        if results and results.get("ids"):
            for i in range(len(results["ids"][0])):
                hits.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": float(results["distances"][0][i]),
                    "similarity": 1 - float(results["distances"][0][i])  # 코사인 거리를 유사도 점수로 변환
                })
        
        return {
            "query": request.query,
            "collection": request.collection_type,
            "hits": hits,
            "total_hits": len(hits)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

# 알러지 필터링이 적용된 검색 엔드포인트
@app.post("/search_with_allergy_filter/")
async def search_with_allergy_filter(request: SearchRequest, user_email: str):
    """알러지 정보를 고려한 레시피 검색"""
    try:
        # 1. 기본 레시피 검색 (필터링을 위해 요청된 것보다 2배 많은 결과 조회)
        recipe_results = recipes_info.query(
            query_embeddings=model.encode([request.query], normalize_embeddings=True).tolist(),
            n_results=request.top_k * 2,
            include=["metadatas", "distances", "documents", "ids"]
        )
        
        # 2. 사용자의 알러지 정보 조회
        user_allergy_results = user_allergy.get(ids=[user_email])
        user_allergies = []
        if user_allergy_results and user_allergy_results.get("documents"):
            user_allergies = user_allergy_results["documents"][0].split(",")
        
        # 3. 알러지 성분 필터링
        filtered_hits = []
        if recipe_results and recipe_results.get("ids"):
            for i in range(len(recipe_results["ids"][0])):
                recipe_ingredients = recipe_results["metadatas"][0][i].get("요리별재료", "")
                
                # 알러지 성분이 포함된 레시피 제외
                has_allergy = any(allergy.strip() in recipe_ingredients for allergy in user_allergies)
                
                if not has_allergy:
                    filtered_hits.append({
                        "id": recipe_results["ids"][0][i],
                        "document": recipe_results["documents"][0][i],
                        "metadata": recipe_results["metadatas"][0][i],
                        "distance": float(recipe_results["distances"][0][i]),
                        "similarity": 1 - float(recipe_results["distances"][0][i])
                    })
                    
                    # 요청된 수만큼 결과를 찾으면 중단
                    if len(filtered_hits) >= request.top_k:
                        break
        
        return {
            "query": request.query,
            "user_email": user_email,
            "user_allergies": user_allergies,
            "hits": filtered_hits,
            "total_hits": len(filtered_hits)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"알러지 필터 검색 실패: {str(e)}")

# 직접 실행 시 uvicorn 서버 구동
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)