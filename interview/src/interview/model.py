from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class ResumeItem(BaseModel):
    key: str  # 키워드 (없을 수도 있음)
    desc: str = "" # 설명은 반드시 필요
class InterviewState(BaseModel):
    question: str = "" # 마지막 질문
    questions: List[str] = Field(default_factory=list)      # 전체 질문 히스토리
    answer: List[str] = []
    last_analysis: Dict[str, Any] = Field(default_factory=dict) # ✅ 분석 결과를 딕셔너리로 저장
    last_answer: Optional[str] = None
    step: int = 0
    seq: int = 0
    keepGoing: bool = True # ✅ 계속 진행 여부
    ocrText: Optional[str] = None
    job: Optional[str] = None
    career: Optional[str] = None
    resume: str = ""
    last_question_for_dynamic: Optional[str] = None
    interviewId: Optional[str] = None  # ✅ 유지
    language: Literal["KOREAN","ENGLISH"] = "KOREAN"    # ✅ 유지
    level: Literal["상","중","하"] = "중"
    interviewType: Optional[str] = None
    count: int = 0
    options_locked: bool = False  # ✅ 유지
    aspect_index: int = 0         # 질문 측면 라운드로빈 인덱스
    dup_streak: int = 0 
    #-----------------------------------------------------------------------------------------------
    topics: List[Dict[str, Any]] = Field(default_factory=list)  # 추출된 토픽
    current_topic_index: Optional[int] = None                  # 현재 토픽 인덱스
    raw_llm_response: Optional[str] = None  # 유사문 판정 연속 실패 카운트
    aspects: List[str] = []  # 질문 측면 (예: "기술적", "인성", "경험")
    resume_summary: List[ResumeItem] = [] # 자기소개서 요약문 리스트
    bridge_switched: bool = False  # ✅ 브릿지 전환 플래그
    bridge_done: bool = False 
    last_label: Optional[str] = None# ✅ 브릿지 전환 완료 플래그
    topic: Optional[str] = None
    aspect: Optional[str] = None
    subtype: Optional[str] = None
    class MyModel(BaseModel):
        # ✅ 외부에서 dict → 모델 생성할 때, alias 없이도 작동하게 허용
        model_config = {
        "validate_by_name": True,
        "populate_by_name": True
        }
