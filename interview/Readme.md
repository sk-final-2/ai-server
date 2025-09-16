## 프로젝트 개요 (Overview)

- 면접자의 답변을 처리하고 후속 질문을 생성하는 AI 서버 모듈

- Whisper + LangGraph + ChromaDB 사용

## 주요 기능 (Features)

- 음성 인식 (STT): Whisper를 이용해 면접자의 음성을 텍스트로 변환

- 질문 흐름 제어: LangGraph로 면접 질문 생성 과정을 노드 단위로 관리 (첫 질문 → 후속 질문)

- 대화 맥락 저장: ChromaDB에 답변을 저장하고, 의미 기반 검색으로 맥락 유지

- 후속 질문 생성: 답변을 분석해 같은 주제 더 깊이 물어보거나, 새로운 주제로 전환

## 작동 방식(Flow)
**1. 사용자가 문서 업로드**

- 음성 → Whisper로 텍스트 변환

**2. 면접자의 응답을 STT로 전처리하여 LangGraph에 전달**

- 문서 → OCR로 텍스트 추출

**3. LangGraph**

- 초기 질문 생성 (자기소개서 기반 or 기본 직무 기반)

- 답변을 저장 & 다음 질문 경로 선택

- 후속 질문 생성

- 같은 주제 파고들기 / 주제 전환 / 기술 ↔ 인성 질문 변환


## 📂 Project Structure
```
interview/
├── .dockerignore
├── .env
├── .gitignore
├── Dockerfile
├── graph.mmd # LangGraph 설계 다이어그램 (Mermaid)
├── graph.png # LangGraph 설계 이미지
├── poetry.lock
├── pyproject.toml # Poetry 환경 설정
├── Readme.md
├── vis.py # 시각화 관련 코드
├── src/ # 메인 소스 코드
│ ├── main.py # FastAPI 실행 진입점
│ └── test_main.py # 테스트 코드
├── temp/ # STT 처리용 임시 파일 저장 (m4a 등)
│ └── *.m4a
├── .git/ # Git 설정
└── .venv/ # 가상환경
```
##