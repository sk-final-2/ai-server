from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class InterviewState(BaseModel):
    question: List[str] = []
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
    dup_streak: int = 0           # 유사문 판정 연속 실패 카운트

    class Config:
        # ✅ 외부에서 dict → 모델 생성할 때, alias 없이도 작동하게 허용
        allow_population_by_field_name = True
        populate_by_name = True
