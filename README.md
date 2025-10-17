# 🍳 재료 감지 API 설명서

> 사진 속 음식 재료를 자동 인식하는 AI API입니다.
> 
> 
> 업로드된 이미지를 **YOLOv7**로 분석해 재료의 **종류, 위치(박스), 신뢰도**를 반환합니다.
> 

---

## 📱 서비스 개요

사용자가 보내는 음식 사진을 분석하여 포함된 **재료명 리스트**와 **탐지 박스 좌표**를 제공합니다. Flutter 앱과 연동해 **레시피 추천**, **알러지 필터링** 등으로 확장 가능합니다.

---

## 🛠 기술 스택

### ⚡ FastAPI

- Python 기반 고성능 웹 프레임워크
- 자동 문서(Swagger/Redoc) 지원
- 비동기 처리로 빠른 응답

### 🤖 YOLOv7

- 실시간 객체 탐지 모델
- 다중 객체(재료) 동시 탐지
- **Bounding Box / Class / Confidence** 제공

### 🔐 CORS

- Flutter ↔ API **교차 출처 요청 허용**
- 개발 단계: `allow_origins=["*"]` (배포 시 화이트리스트 권장)

### 🧹 NMS (Non-Max Suppression)

- **중복 박스 자동 제거**
- 가장 신뢰도 높은 박스만 남김

---

## 📸 처리 프로세스

1. **이미지 업로드 수신** (`multipart/form-data`)
2. **리사이즈 640×640(원본 비율 유지, Letterbox)**
3. **YOLOv7 추론** (GPU/CPU 자동 선택, FP16 지원)
4. **NMS 후처리** + **원본 좌표로 복원**
5. **JSON 결과 반환** (재료 목록, 박스, 신뢰도)

---

## 🧭 모델/환경 설정(요약)

- 기본 입력 크기: `640`
- 신뢰도 임계값: `0.25`
- IoU 임계값: `0.45`
- FP16: GPU 사용 시 자동 활성화
- 클래스 매핑(영→한, 일부 예):
    
    ```json
    {
      "cab": "배추", "cab2": "양배추", "car": "당근",
      "cuc": "오이", "egg": "달걀", "gar": "마늘",
      "lee": "대파", "oni": "양파", "pork": "돼지고기",
      "pota": "감자", "rad": "무"
    }
    
    ```
    

---

## 🔄 API 목록

### 1) `POST /predict` — 사진에서 재료 찾기

- **요청**
    - Content-Type: `multipart/form-data`
    - Body:
        - `image`: (필수) 이미지 파일(JPEG/PNG)
        - `img_size`: (선택) 입력 크기, 기본 `640`
        - `conf_threshold`: (선택) 신뢰도 임계값, 기본 `0.25`
- **응답(200 OK)**
    
    ```json
    {
      "ingredients": ["양파","돼지고기","감자"],
      "detections": [
        {
          "label": "양파",
          "label_en": "oni",
          "confidence": 0.962,
          "box_xyxy": [50.0, 30.0, 200.0, 180.0],
          "class_id": 7
        }
      ],
      "count": 3,
      "classes": ["cab","cab2","car","cuc","egg","gar","lee","oni","pork","pota","rad"]
    }
    
    ```
    
- **예시(cURL)**
    
    ```bash
    curl -X POST "http://<HOST>:8081/predict" \
         -H "Accept: application/json" \
         -F "image=@/path/sample.jpg" \
         -F "img_size=640" \
         -F "conf_threshold=0.25"
    
    ```
    

---

### 2) `POST /predict-and-recommend` — 재료 감지 + 레시피 추천(스텁)

- **설명**: 감지 후 추천 로직(추후 구현)까지 한 번에 호출
- **요청**
    - `image`: (필수) 이미지 파일
    - `top_k`: (선택) 추천 개수, 기본 `5`
- **응답(200 OK)**
    
    ```json
    {
      "ingredients": ["양파","감자"],
      "detections": [ /* predict와 동일 구조 */ ],
      "count": 2,
      "recommendations": [],   // TODO: 추후 구현
      "classes": [ "... 클래스 목록 ..." ]
    }
    
    ```
    

---

### 3) `GET /health` — 서버 상태 확인

- **응답(200 OK)**
    
    ```json
    {
      "status": "healthy",
      "model_loaded": true,
      "model_version": "YOLOv7",
      "device": "cuda:0",
      "class_count": 11,
      "classes": ["cab","cab2","car","cuc","egg","gar","lee","oni","pork","pota","rad"]
    }
    
    ```
    

---

## 🧯 오류 응답 형식

- 잘못된 파일 타입
    
    ```json
    { "detail": "이미지 파일을 업로드하세요." }
    
    ```
    
- 내부 처리 오류
    
    ```json
    { "detail": "추론 중 오류: <메시지>" }
    
    ```
    

HTTP 상태코드: `400`(요청 오류), `500`(서버 오류)

---

## 🔒 보안/네트워크

- **CORS**: 개발용 , 배포 시 **앱 도메인 화이트리스트**로 제한 권장
- **TLS/HTTPS** 권장
- (선택) `Authorization: Bearer <Firebase ID Token>` 검증 로직 추가 가능

---

## ⚙️ 성능 팁

- 업로드 전 **클라이언트에서 JPEG 압축(품질 0.8)** 추천
- 긴 변을 1280px 정도로 **선압축**하면 네트워크 지연 ↓
- GPU 사용 시 **FP16 활성화**로 지연 시간 ↓

---

## 🧩 통합 가이드(Flutter)

- `multipart/form-data`로 이미지 전송
- 예시 응답의 `box_xyxy`는 **원본 좌표 기준** → 바로 캔버스 오버레이 가능
- 반환된 `ingredients`로 **레시피 API/파이어스토어** 조회

---

## 📚 참고 자료

- FastAPI 문서: https://fastapi.tiangolo.com/
- YOLOv7 GitHub: https://github.com/WongKinYiu/yolov7

---

**버전**: 1.0 (2025-10-16)

**팀**: Mini3

**문의**: sde0110@naver.com
