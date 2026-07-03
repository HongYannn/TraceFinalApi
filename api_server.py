import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from scipy.stats import pearsonr

# 引入 FastAPI 相關套件
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 引入純後端繪圖物件，完全取代 plt
from matplotlib.figure import Figure

app = FastAPI(title="Side-Channel Analysis API", version="1.0")

# 1. 允許前端跨網域 (CORS) 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 基礎配置路徑 ---
DATASET_PATHS = {
    "fixed": r"C:/Users/schoo/OneDrive/Desktop/model/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5",  
    "rand": r"C:\Users\schoo\OneDrive\Desktop\final\data\ascad-variable.h5",
    "ches": r"C:\Users\schoo\OneDrive\Desktop\final\data\ches_ctf.h5"
}
MODEL_FOLDER = r"C:\Users\schoo\OneDrive\Desktop\final\model"
PIC_OUTPUT_FOLDER = r"C:\Users\schoo\OneDrive\Desktop\final\pic"
os.makedirs(PIC_OUTPUT_FOLDER, exist_ok=True)

# 2. 掛載靜態資料夾，讓前端可以直接透過網址讀取圖片
app.mount("/static", StaticFiles(directory=PIC_OUTPUT_FOLDER), name="static")

# --- 工具常數 ---
AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0x0D, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])
HW = np.array([bin(n).count("1") for n in range(256)])

FILENAME_MAPPING = {
    ("cnn", "id", "fixed"): "fixed_cnn_bo_id_model.h5",
    ("cnn", "hw", "fixed"): "fixed_cnn_bo_hw_model.h5",
    ("mlp", "id", "fixed"): "fixed_mlp_bo_id_model.h5",
    ("mlp", "hw", "fixed"): "fixed_mlp_bo_hw_model.h5",
    ("cnn", "id", "rand"): "rand_cnn_bo_id_model.h5",
    ("cnn", "hw", "rand"): "rand_cnn_bo_hw_model.h5",
    ("mlp", "id", "rand"): "rand_mlp_bo_id_model.h5",
    ("mlp", "hw", "rand"): "rand_mlp_bo_hw_model.h5",
    ("cnn", "id", "ches"): "ches_cnn_bo_id_model.h5",
    ("cnn", "hw", "ches"): "ches_cnn_bo_hw_model.h5",
    ("mlp", "id", "ches"): "ches_mlp_bo_id_model.h5",
    ("mlp", "hw", "ches"): "ches_mlp_bo_hw_model.h5",
}

# 全域狀態，用來記錄當前背景運算狀態
task_status = {"status": "idle", "message": "無執行中的任務"}

# 3. 定義前端傳入的參數結構
class EvaluationRequest(BaseModel):
    model_type: str    # 'CNN' 或 'MLP'
    leakage_model: str # 'ID' 或 'HW'
    dataset_type: str  # 'FIXED', 'RAND', 或 'CHES'

def load_specific_test_dataset(dataset_type):
    db_path = DATASET_PATHS[dataset_type]
    f = h5py.File(db_path, 'r')
    if dataset_type in ["fixed", "rand"]:
        X_prof_raw = np.array(f['Profiling_traces/traces'], dtype=np.float32)
        X_test_raw = np.array(f['Attack_traces/traces'], dtype=np.float32)
        P_test = np.array(f['Attack_traces/metadata']['plaintext'])[:, 2] 
        target_key = f['Attack_traces/metadata']['key'][0][2]
    elif dataset_type == "ches":
        X_prof_raw = np.array(f['profiling_traces'], dtype=np.float32)
        X_test_raw = np.array(f['attacking_traces'], dtype=np.float32)
        att_metadata = np.array(f['attacking_data'])
        P_test = att_metadata[:, 0]      
        target_key = att_metadata[0, 32] 
        
    scaler = StandardScaler()
    scaler.fit(X_prof_raw) 
    X_test_scaled = scaler.transform(X_test_raw)
    return X_test_scaled, X_test_raw, P_test, target_key

# 核心計算邏輯（純背景執行，無任何 GUI）
def background_evaluation_flow(model_type, leakage_model, dataset_type, model_filename):
    global task_status
    task_status["status"] = "processing"
    task_status["message"] = f"正在計算組合: {model_filename}..."
    
    try:
        X_test, X_test_raw, P_test, target_key = load_specific_test_dataset(dataset_type)
        
        if model_type == "cnn":
            X_test_model_input = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        else:
            X_test_model_input = X_test 

        model_path = os.path.join(MODEL_FOLDER, model_filename)
        model = load_model(model_path)
        predictions = model.predict(X_test_model_input)
        
        max_eval_traces = len(X_test)
        key_log_probs = np.zeros(256)
        ge_one_dim_results = []
        
        for i in range(max_eval_traces):
            for k in range(256):
                v = AES_Sbox[P_test[i] ^ k]
                prob_val = predictions[i][v] if leakage_model == "id" else predictions[i][HW[v]]
                key_log_probs[k] += np.log(prob_val + 1e-10)
                
            sorted_keys = np.argsort(key_log_probs)[::-1]
            rank = np.where(sorted_keys == target_key)[0][0]
            ge_one_dim_results.append(int(rank)) 

        num_points = X_test_raw.shape[1]
        pearson_one_dim_results = np.zeros(num_points)
        hypothesized_hw = np.array([HW[AES_Sbox[P_test[i] ^ target_key]] for i in range(max_eval_traces)])
        
        for c_point in range(num_points):
            trace_point_data = X_test_raw[:max_eval_traces, c_point]
            corr, _ = pearsonr(trace_point_data, hypothesized_hw)
            pearson_one_dim_results[c_point] = 0.0 if np.isnan(corr) else float(corr)

        h5_filename = os.path.splitext(os.path.basename(model_filename))[0]

        # --- 純後台建立與儲存 Guessing Entropy (GE) 圖片 ---
        fig_ge = Figure(figsize=(10, 5), dpi=150)
        ax_ge = fig_ge.add_subplot(111)
        ax_ge.plot(ge_one_dim_results, color='tab:blue', linewidth=1.8, label="Guessing Entropy")
        ax_ge.axhline(y=0, color='red', linestyle='--', linewidth=1.2, label="Correct Key (Rank 0)")
        ax_ge.set_title(f"Guessing Entropy: {h5_filename}", fontsize=12)
        ax_ge.set_xlabel("Number of Attack Traces", fontsize=10)
        ax_ge.set_ylabel("Key Rank (0-255)", fontsize=10)
        ax_ge.set_xlim(0, max_eval_traces)
        ax_ge.set_ylim(-5, 260)  
        ax_ge.grid(True, linestyle=':', alpha=0.6)
        ax_ge.legend(loc="upper right")
        
        ge_save_path = os.path.join(PIC_OUTPUT_FOLDER, f"{h5_filename}_ge.png")
        fig_ge.savefig(ge_save_path, bbox_inches='tight')

        # --- 純後台建立與儲存 Pearson Correlation 圖片 ---
        fig_pearson = Figure(figsize=(10, 5), dpi=150)
        ax_pearson = fig_pearson.add_subplot(111)
        ax_pearson.plot(pearson_one_dim_results, color='tab:orange', linewidth=1.2, label="Pearson Correlation")
        ax_pearson.set_title(f"Pearson Correlation: {h5_filename}", fontsize=12)
        ax_pearson.set_xlabel("Time Samples", fontsize=10)
        ax_pearson.set_ylabel("Correlation Coefficient", fontsize=10)
        ax_pearson.grid(True, linestyle=':', alpha=0.6)
        ax_pearson.legend(loc="upper right")
        
        pearson_save_path = os.path.join(PIC_OUTPUT_FOLDER, f"{h5_filename}_pearson.png")
        fig_pearson.savefig(pearson_save_path, bbox_inches='tight')

        task_status["status"] = "success"
        task_status["message"] = "計算完成"
        
    except Exception as e:
        task_status["status"] = "failed"
        task_status["message"] = f"執行出錯: {str(e)}"

# 4. API 路由：觸發評估
@app.post("/api/evaluate")
def evaluate_model(req: EvaluationRequest, background_tasks: BackgroundTasks):
    global task_status
    if task_status["status"] == "processing":
        raise HTTPException(status_code=400, detail="目前有其他任務正在計算中，請稍後再試。")

    m_type = req.model_type.strip().lower()
    l_model = req.leakage_model.strip().lower()
    d_type = req.dataset_type.strip().lower()

    target_file = FILENAME_MAPPING.get((m_type, l_model, d_type))
    if not target_file:
        raise HTTPException(status_code=400, detail="不合法的組合參數，找不到對應權重。")

    model_path = os.path.join(MODEL_FOLDER, target_file)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"伺服器找不到模型權重檔案: {target_file}")

    h5_filename = os.path.splitext(target_file)[0]
    base_url = "http://127.0.0.1:8000/static"
    
    background_tasks.add_task(background_evaluation_flow, m_type, l_model, d_type, target_file)
    
    return {
        "status": "started",
        "message": "分析任務已在背景啟動",
        "expected_urls": {
            "ge_url": f"{base_url}/{h5_filename}_ge.png",
            "pearson_url": f"{base_url}/{h5_filename}_pearson.png"
        }
    }

# 5. API 路由：查詢目前狀態
@app.get("/api/status")
def get_status():
    return task_status