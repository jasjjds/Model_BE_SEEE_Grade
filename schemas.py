from pydantic import BaseModel, field_validator
from typing import List, Optional

# ================= INPUT CHO /predict_subject (MỚI) =================
class GradeEntry(BaseModel):
    subject: str
    grade: str

class SubjectPredictionRequest(BaseModel):
    major: str
    current_grades: List[GradeEntry]
    target_subjects: List[str]

    @field_validator('major')
    def validate_major(cls, v):
        if v not in ["ET1", "EE2"]:
            raise ValueError("Chỉ hỗ trợ ngành ET1 hoặc EE2")
        return v

# ================= INPUT CHO /predict_cpa (CŨ) =================
class CPAPredictionRequest(BaseModel):
    student_type: str  # "Cử nhân" hoặc "Kỹ sư"
    current_semester: int
    gpa_list: List[float]
    tc_list: List[int]

    @field_validator('gpa_list')
    def check_gpa(cls, v):
        if any(not (0.0 <= s <= 4.0) for s in v):
            raise ValueError("GPA phải từ 0.0 - 4.0")
        return v

# ================= OUTPUT CHUNG =================
class SubjectResult(BaseModel):
    subject: str
    predicted_letter: str
    predicted_score: float
    note: Optional[str] = None

class CPAResult(BaseModel):
    cpa_grad_predict: Optional[float]
    next_gpa_predict: Optional[float]