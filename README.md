# 🧠 RecruitAI - AI Servers Repository  

> 🎯 **RecruitAI** 플랫폼의 AI 서버 모듈 모노레포  
> 각 폴더는 독립적인 **FastAPI 기반 마이크로서비스**로 구성되어 있으며,  
> Docker & AWS EC2/Fargate 환경에서 실행 가능합니다.  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-EC2%20%7C%20Fargate-FF9900?logo=amazonaws&logoColor=white" />
</p>

---

## 📂 Repository Structure  

### 1️⃣ `emotion` 💡  
> 얼굴 이미지를 분석하여 **감정 상태(😊 😡 😢 등)** 를 분류  

- 🛠️ **기술 스택:** DeepFace / OpenCV / FastAPI  
- 🌟 **특징:**  
  - `POST /evaluate` → JSON 응답 (`score`, `timestamp` , 'reason')   
  - `GET /health` 헬스 체크 지원
  

---

### 2️⃣ `evaluate` 📝  
> LLM을 활용한 **면접 답변 평가 서버**  
> 점수, 요약, 피드백을 자동 생성  

- 🛠️ **기술 스택:** vLLM (Llama-3 AWQ) / FastAPI  
- 🌟 **특징:**  
  - `POST /evaluate` → JSON 응답 (`score`, `feedback`, `improve`)  
  - GPU 버전 Dockerfile

---

### 3️⃣ `interview` 🎤  
> **면접 세션 제어 및 Q&A 관리 서버**  
> AI 분석 서버와 통신하며 결과를 집계  

- 🛠️ **기술 스택:** FastAPI / Langgrpah / ChromaDB / koelectra
- 🌟 **특징:**  
  - 면접 진행 상태 추적  
  - 사용자 맞춤 질문
  - 꼬리질문 판단  
  - 자기소개서, 포트폴리오 등 문서 OCR
---

### 4️⃣ `tracking` 👀  
> 영상 기반 **시선 추적 / 눈 깜빡임 / 얼굴 터치 / 고개 움직임 감지**  

- 🛠️ **기술 스택:** Google Mediapipe / OpenCV / FastAPI  
- 🌟 **특징:**  
  - WebM → MP4 변환 지원 (ffmpeg)  
  - 실시간 비디오 분석 REST API
  - 각 로직에 쿨다운/연속 프레임 안정화를 적용하여 중복 방지

