import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models_general"  # Folder chứa model cũ

# ================= GLOBALS =================
# Store models in memory
loaded_resources = {
    "subjects_data": {},  # Chứa data ET1, EE2
    "general_models": {}  # Chứa data CPA, NextGPA cũ
}

LETTER_TO_GPA = {
    "A+": 4.0, "A": 4.0, "B+": 3.5, "B": 3.0,
    "C+": 2.5, "C": 2.0, "D+": 1.5, "D": 1.0,
}

# ================= HELPER FUNCTIONS =================
def convert_letter_to_score(letter: str):
    if not letter: return np.nan
    return LETTER_TO_GPA.get(str(letter).strip().upper(), np.nan)

def numeric_to_letter(score: float) -> str:
    if np.isnan(score): return "N/A"
    if score >= 3.75: return "A / A+"
    if score >= 3.25: return "B+"
    if score >= 2.75: return "B"
    if score >= 2.25: return "C+"
    if score >= 1.75: return "C"
    if score >= 1.25: return "D+"
    return "D"

# ================= LOAD MODELS (LIFESPAN) =================
def load_all_resources():
    print("⏳ Đang tải tài nguyên...")
    
    # 1. Load Data Môn học (GGM)
    for major in ["ET1", "EE2"]:
        try:
            folder = BASE_DIR / major
            loaded_resources["subjects_data"][major] = {
                "subjects": json.loads((folder / f"{major}-subjects.json").read_text(encoding="utf-8")),
                "scaler": joblib.load(folder / f"{major}-scaler.joblib"),
                "ggm": joblib.load(folder / f"{major}-ggm.joblib")
            }
            # Pre-calc means/stds
            loaded_resources["subjects_data"][major]["means"] = pd.Series(loaded_resources["subjects_data"][major]["scaler"]["means"])
            loaded_resources["subjects_data"][major]["stds"] = pd.Series(loaded_resources["subjects_data"][major]["scaler"]["stds"]).replace(0, 1.0)
            print(f"✅ Loaded {major}")
        except Exception as e:
            print(f"⚠️ Lỗi load {major}: {e}")

    # 2. Load Data CPA cũ
    # Format key: "cunhan_cpa", "kysu_next", v.v.
    configs = [
        ("Cử nhân", "8"), ("Kỹ sư", "10")
    ]
    for stype, prefix in configs:
        key_prefix = "cunhan" if stype == "Cử nhân" else "kysu"
        try:
            # Load CPA model
            cpa_path = MODELS_DIR / f"final_cpa_{prefix}_ki.joblib"
            if cpa_path.exists():
                loaded_resources["general_models"][f"{key_prefix}_cpa"] = joblib.load(cpa_path)
            
            # Load Next GPA model
            next_path = MODELS_DIR / f"next_gpa_{prefix}_ki.joblib"
            if next_path.exists():
                loaded_resources["general_models"][f"{key_prefix}_next"] = joblib.load(next_path)
                
            print(f"✅ Loaded General Models cho {stype}")
        except Exception as e:
             print(f"⚠️ Lỗi load model CPA {stype}: {e}")

# ================= LOGIC 1: DỰ ĐOÁN MÔN (GGM) =================
def predict_subject_score(major: str, user_grades: dict, target_subjects: list):
    data = loaded_resources["subjects_data"].get(major)
    if not data:
        raise ValueError(f"Chưa load data ngành {major}")

    means, stds, ggm, all_subjects = data["means"], data["stds"], data["ggm"], data["subjects"]
    idx_map = {s: i for i, s in enumerate(all_subjects)}
    
    results = []
    
    # Tính Z-score user
    x_user = []
    observed_indices = []
    for s in all_subjects:
        val = user_grades.get(s, np.nan)
        if pd.isna(val):
            x_user.append(np.nan)
        else:
            x_user.append((float(val) - means[s]) / stds[s])
            observed_indices.append(idx_map[s])

    cov = np.asarray(ggm["cov"])

    for target in target_subjects:
        if target not in idx_map:
            results.append({"subject": target, "predicted_letter": "Unknown", "predicted_score": 0.0})
            continue

        target_idx = idx_map[target]
        current_O = np.array([o for o in observed_indices if o != target_idx])
        
        if current_O.size == 0:
            results.append({"subject": target, "predicted_letter": "N/A", "predicted_score": 0.0})
            continue

        # Math logic
        S_TO = cov[target_idx, current_O].reshape(1, -1)
        S_OO = cov[np.ix_(current_O, current_O)]
        x_O = np.array([x_user[o] for o in current_O])

        try:
            inv_S_OO = np.linalg.inv(S_OO)
        except np.linalg.LinAlgError:
            inv_S_OO = np.linalg.pinv(S_OO)

        y_z = (S_TO @ inv_S_OO @ (x_O)).item()
        y_raw = y_z * stds[target] + means[target]
        letter = numeric_to_letter(y_raw)
        
        results.append({
            "subject": target,
            "predicted_letter": letter,
            "predicted_score": convert_letter_to_score(letter.split()[0])
        })
    return results

# ================= LOGIC 2: DỰ ĐOÁN CPA (CŨ) =================
def _build_vector(gpa_list, tc_list):
    features = []
    for gpa, tc in zip(gpa_list, tc_list):
        features.append(gpa)
        features.append(tc)
    return np.array(features).reshape(1, -1)

def predict_cpa_general(student_type: str, current_semester: int, gpa_list: list, tc_list: list):
    type_key = "cunhan" if student_type == "Cử nhân" else "kysu"
    input_vector = _build_vector(gpa_list, tc_list)
    
    res = {"cpa_grad_predict": None, "next_gpa_predict": None}
    
    # 1. Predict CPA
    cpa_models = loaded_resources["general_models"].get(f"{type_key}_cpa")
    if cpa_models:
        key = f"GPA_TC_1_{current_semester}" if current_semester > 1 else "GPA_TC_1"
        if key in cpa_models:
            pred = cpa_models[key].predict(input_vector)[0]
            res["cpa_grad_predict"] = round(float(pred), 2)

    # 2. Predict Next GPA
    next_models = loaded_resources["general_models"].get(f"{type_key}_next")
    max_sem = 6 if student_type == "Cử nhân" else 8
    
    if next_models and current_semester < max_sem:
        key = f"GPA_{current_semester + 1}"
        if key in next_models:
            pred = next_models[key].predict(input_vector)[0]
            res["next_gpa_predict"] = round(float(pred), 2)
            
    return res