# main.py (Phiên bản mới: API Backend)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List
import joblib
import numpy as np
from pathlib import Path

# ================= KHOI TAO APP =================
app = FastAPI(title="Dự đoán kết quả học tập API")

# Cấu hình CORS (Cho phép Next.js gọi vào)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên để domain cụ thể của Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= DATA MODELS (Pydantic) =================
class StudentInput(BaseModel):
    student_type: str  # "Cử nhân" hoặc "Kỹ sư"
    current_semester: int
    gpa_list: List[float]
    tc_list: List[int]

    # Validate dữ liệu cơ bản ngay từ đầu vào
    @field_validator('gpa_list')
    def check_gpa_range(cls, v):
        for score in v:
            if not (0.0 <= score <= 4.0):
                raise ValueError(f'GPA {score} không hợp lệ (phải từ 0.0 - 4.0)')
        return v

    @field_validator('tc_list')
    def check_tc_range(cls, v):
        for tc in v:
            if tc < 0:
                raise ValueError(f'Tín chỉ {tc} không hợp lệ (phải >= 0)')
        return v

# ================= HELPER FUNCTIONS (Giữ lại từ code cũ) =================
def load_model_from_disk(path: str):
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model không tìm thấy tại: {path}")
    return joblib.load(path_obj)

def build_feature_vector(gpa_list, tc_list):
    # Logic cũ: gộp GPA và Tín chỉ xen kẽ nhau
    features = []
    for gpa, tc in zip(gpa_list, tc_list):
        features.append(gpa)
        features.append(tc)
    return np.array(features).reshape(1, -1)

def get_model_path(student_type: str, model_type: str):
    prefix = "8" if student_type == "Cử nhân" else "10"
    if model_type == "cpa":
        return f"models_streamlit/final_cpa_{prefix}_ki.joblib"
    else:
        return f"models_streamlit/next_gpa_{prefix}_ki.joblib"

# ================= API ENDPOINTS =================
@app.get("/")
def health_check():
    return {"status": "running", "message": "API Dự đoán học tập đã sẵn sàng"}

@app.post("/predict/cpa")
async def predict(data: StudentInput):
    # 1. Validate số lượng
    if len(data.gpa_list) != data.current_semester or len(data.tc_list) != data.current_semester:
        raise HTTPException(status_code=400, detail="Số lượng điểm GPA và Tín chỉ không khớp với số kỳ đã khai báo.")

    # 2. Chuẩn bị vector
    try:
        input_vector = build_feature_vector(data.gpa_list, data.tc_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý dữ liệu: {str(e)}")

    response_data = {
        "success": True,
        "cpa_grad_predict": None,
        "next_gpa_predict": None
    }

    # 3. Dự đoán CPA
    try:
        cpa_path = get_model_path(data.student_type, "cpa")
        cpa_dict = load_model_from_disk(cpa_path)
        
        group_key_cpa = f"GPA_TC_1_{data.current_semester}" if data.current_semester > 1 else "GPA_TC_1"
        
        if group_key_cpa in cpa_dict:
            model_cpa = cpa_dict[group_key_cpa]
            pred = model_cpa.predict(input_vector)[0]
            response_data["cpa_grad_predict"] = round(float(pred), 2)
        else:
            print(f"Warning: Không tìm thấy key {group_key_cpa} trong model CPA")

    except Exception as e:
        print(f"Lỗi dự đoán CPA: {e}")
    max_semester = 6 if data.student_type == "Cử nhân" else 8
    if data.current_semester < max_semester:
        try:
            next_path = get_model_path(data.student_type, "next")
            next_dict = load_model_from_disk(next_path)
            
            group_key_gpa = f"GPA_{data.current_semester + 1}"
            
            if group_key_gpa in next_dict:
                model_next = next_dict[group_key_gpa]
                pred_next = model_next.predict(input_vector)[0]
                response_data["next_gpa_predict"] = round(float(pred_next), 2)
        except Exception as e:
             print(f"Lỗi dự đoán Next GPA: {e}")

    return response_data