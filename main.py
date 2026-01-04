# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from schemas import SubjectPredictionRequest, CPAPredictionRequest, SubjectResult, CPAResult
import services

# ==========================================
# 1. LIFESPAN: Quản lý vòng đời ứng dụng
# (Load model khi bật, giải phóng RAM khi tắt)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load toàn bộ model và data vào RAM
    services.load_all_resources()
    yield
    # Dọn dẹp RAM khi tắt server
    services.loaded_resources["subjects_data"].clear()
    services.loaded_resources["general_models"].clear()

# Khởi tạo App
app = FastAPI(title="Grade Prediction System", lifespan=lifespan)

# ==========================================
# 2. MIDDLEWARE: Khắc phục lỗi CORS (405 Options)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Cho phép mọi nguồn (Frontend từ port 3000, mobile app...)
    allow_credentials=True,
    allow_methods=["*"],      # Cho phép mọi method: GET, POST, OPTIONS, PUT...
    allow_headers=["*"],      # Cho phép mọi header
)

# ==========================================
# 3. API ENDPOINTS
# ==========================================

@app.get("/")
def health_check():
    """Kiểm tra server có sống không"""
    return {"status": "ok", "message": "API is running"}

# --- API 1: Dự đoán điểm môn học cụ thể (GGM) ---
@app.post("/predict_subject", response_model=list[SubjectResult])
def predict_subject(payload: SubjectPredictionRequest):
    # 1. Chuyển đổi danh sách điểm từ Input (List Object) sang Dictionary {Môn: Điểm số}
    user_grades = {}
    for item in payload.current_grades:
        # Convert điểm chữ (A, B+) sang số (4.0, 3.5)
        s = services.convert_letter_to_score(item.grade)
        # Chỉ lấy điểm hợp lệ (không phải NaN)
        if not services.np.isnan(s):
            user_grades[item.subject] = s
            
    # 2. Validate: Cần tối thiểu 3 môn đầu vào
    if len(user_grades) < 3:
        raise HTTPException(status_code=400, detail="Cần ít nhất 3 môn đã có điểm để dự đoán.")

    # 3. Gọi Service xử lý
    try:
        results = services.predict_subject_score(
            major=payload.major, 
            user_grades=user_grades, 
            target_subjects=payload.target_subjects
        )
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Server: {str(e)}")

# --- API 2: Dự đoán CPA ra trường & GPA kỳ tới (Regression) ---
@app.post("/predict_cpa", response_model=CPAResult)
def predict_cpa(payload: CPAPredictionRequest):
    # 1. Validate số lượng phần tử khớp với số kỳ
    if len(payload.gpa_list) != payload.current_semester or len(payload.tc_list) != payload.current_semester:
        raise HTTPException(
            status_code=400, 
            detail=f"Số lượng điểm GPA ({len(payload.gpa_list)}) và Tín chỉ ({len(payload.tc_list)}) không khớp với số kỳ đã khai báo ({payload.current_semester})."
        )

    # 2. Gọi Service xử lý
    try:
        return services.predict_cpa_general(
            student_type=payload.student_type,
            current_semester=payload.current_semester,
            gpa_list=payload.gpa_list,
            tc_list=payload.tc_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Server: {str(e)}")