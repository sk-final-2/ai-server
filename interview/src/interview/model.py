from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class InterviewState(BaseModel):
    questions: List[str] = Field(default_factory=list)
    answers: List[str] = Field(default_factory=list)
    last_analysis: Optional[Dict] = None
    step: int = 0
    seq: int = 0
    is_finished: bool = False
    text: Optional[str] = None
    job: Optional[str] = None
    interview_id: Optional[str] = None