from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class InterviewState(BaseModel):
    questions: List[str] = []
    answer: List[str] = []
    last_analysis: Dict[str, Any] = Field(default_factory=dict) # ✅ 분석 결과를 딕셔너리로 저장
    last_answer: Optional[str] = None
    step: int = 0
    seq: int = 0
    is_finished: bool = False
    text: Optional[str] = None
    job: Optional[str] = None
    career: Optional[str] = None
    resume: str = ""
    interviewId: Optional[str] = None  # ✅ 유지
    Language: Literal["KOREAN","ENGLISH"] = "KOREAN"    # ✅ 유지
    level: Literal["상","중","하"] = "중"
    interviewType: Optional[str] = None
    count: int = 0
    options_locked: bool = False  # ✅ 유지
    class Config:
        # ✅ 외부에서 dict → 모델 생성할 때, alias 없이도 작동하게 허용
        allow_population_by_field_name = True
        populate_by_name = True
