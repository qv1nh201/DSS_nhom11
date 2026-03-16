from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import pyodbc
import json
import os

app = Flask(__name__)

# ==========================================
# 0. CẤU HÌNH KẾT NỐI CSDL DÙNG CHUNG (FIX LỖI NOT DEFINED)
# ==========================================
# Thay đổi 'localhost' thành tên Server của bạn nếu cần (VD: r'localhost\SQLEXPRESS')
CONNECTION_STRING = (
    r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=localhost;' 
    r'DATABASE=DSS_Startup;'
    r'Trusted_Connection=yes;'
)

CONFIG_FILE = 'ahp_config.json'

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH AI & CẤU HÌNH AHP
# ==========================================
try:
    model = pickle.load(open('loan_model.pkl', 'rb'))
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file loan_model.pkl.")

# Trọng số mức độ đánh giá (Absolute AHP)
INTENSITY_SCORES = {
    "Tốt": 0.633,
    "Khá": 0.260,
    "Kém": 0.106
}

def load_ahp_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "weights": [0.5378, 0.2745, 0.1285, 0.0591],
        "labels": ["Sức khỏe tài chính", "Uy tín tín dụng", "Tính khả thi dự án", "Năng lực Founder"]
    }

# Khởi tạo trọng số hệ thống
ahp_data = load_ahp_config()
CRITERIA_WEIGHTS = np.array(ahp_data["weights"])

def calculate_ahp(matrix):
    column_sums = np.sum(matrix, axis=0)
    norm_matrix = matrix / column_sums
    weights = np.mean(norm_matrix, axis=1)
    n = len(matrix)
    lamda_max = np.sum(column_sums * weights)
    ci = (lamda_max - n) / (n - 1)
    ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12} 
    cr = ci / ri_dict.get(n, 1.0)
    return weights, cr

# ==========================================
# 2. CÁC ROUTE GIAO DIỆN
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    return render_template('assessment.html')

@app.route('/history')
def history():
    return render_template('history.html')

# ==========================================
# 3. CÁC API XỬ LÝ DỮ LIỆU
# ==========================================

@app.route('/api/ahp_weights', methods=['GET', 'POST'])
def api_ahp_weights():
    global CRITERIA_WEIGHTS
    if request.method == 'GET':
        return jsonify(load_ahp_config())
    
    try:
        data = request.json
        matrix = np.array(data['ahp_matrix'], dtype=float)
        weights, cr = calculate_ahp(matrix)
        
        if cr > 0.1:
            return jsonify({"error": f"CR={round(cr,4)} > 0.1"}), 400

        CRITERIA_WEIGHTS = weights
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"weights": weights.tolist()}, f)

        return jsonify({
            "weights": [round(w * 100, 2) for w in weights],
            "cr": round(cr, 4),
            "labels": ahp_data["labels"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def api_dashboard():
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Loans")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM Loans WHERE Recommendation = N'Duyệt vay'")
        approved = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM Loans WHERE Recommendation LIKE N'Từ chối%'")
        rejected = cursor.fetchone()[0]
        
        cursor.execute("SELECT TOP 5 ID, AHP_Score, AI_Status, Recommendation FROM Loans ORDER BY ID DESC")
        recent = [{"startup": f"Startup #{r.ID}", "score": r.AHP_Score, "ai_status": r.AI_Status, "recommendation": r.Recommendation} for r in cursor.fetchall()]
        conn.close()
        return jsonify({"total": total, "approved": approved, "rejected": rejected, "recent": recent})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Loans ORDER BY ID DESC")
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete_loan/<int:id>', methods=['DELETE'])
def delete_loan(id):
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Loans WHERE ID = ?", (id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        web_data = data['financial_data'] 
        qualitative_data = data.get('qualitative_data', ["Khá", "Khá", "Khá"]) 

        # --- 1. AI Predict ---
        feature_names = model.feature_names_in_
        input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        input_df.at[0, 'Term'] = web_data[0]
        input_df.at[0, 'NoEmp'] = web_data[1]
        input_df.at[0, 'GrAppv'] = web_data[2]
        input_df.at[0, 'DisbursementGross'] = web_data[3]
        input_df.at[0, 'RealEstate'] = web_data[4]
        
        prediction = 1 if (web_data[2] < 5000 or web_data[3] < 2000) else model.predict(input_df)[0]
        ai_status = "An toàn" if prediction == 0 else "Rủi ro"

        # --- 2. AHP Calculate ---
        financial_score = INTENSITY_SCORES["Tốt"] if prediction == 0 else INTENSITY_SCORES["Kém"]
        expert_scores = np.array([
            financial_score, 
            INTENSITY_SCORES[qualitative_data[0]], 
            INTENSITY_SCORES[qualitative_data[1]], 
            INTENSITY_SCORES[qualitative_data[2]]
        ])
        final_score = np.dot(CRITERIA_WEIGHTS, expert_scores) * 100 
        
        # Luồng ra quyết định
        if ai_status == "Rủi ro" and final_score >= 65: 
            recommendation = "Xem xét đặc biệt (Giám đốc duyệt)"
        elif final_score >= 65 and ai_status == "An toàn": 
            recommendation = "Duyệt vay"
        elif ai_status == "Rủi ro": 
            recommendation = "Từ chối (Rủi ro tài chính cao)"
        else: 
            recommendation = "Từ chối (Điểm tiềm năng AHP quá thấp)"

        # --- 3. TẠO VĂN BẢN GIẢI TRÌNH (XAI) ---
        reasons = []
        # Giải trình về AI
        if ai_status == "Rủi ro":
            reasons.append(f"- Sức khỏe tài chính: Hệ thống AI đánh giá có rủi ro cao (Trọng số ảnh hưởng: {round(CRITERIA_WEIGHTS[0]*100)}%).")
        else:
            reasons.append("- Sức khỏe tài chính: Chỉ số tài chính đạt ngưỡng an toàn.")

        # Giải trình về các tiêu chí AHP
        labels = ["Uy tín", "Khả thi", "Founder"]
        for i in range(1, 4):
            if expert_scores[i] <= 0.260: # Nếu chọn mức Khá hoặc Kém
                reasons.append(f"- {labels[i-1]}: Đánh giá mức '{qualitative_data[i-1]}' làm giảm điểm tiềm năng.")
        
        explanation = " ".join(reasons)

        # --- 4. Lưu vào SQL Server (Thêm cột Explanation) ---
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        
        # Đảm bảo bảng Loans đã có cột Explanation (NVARCHAR(MAX))
        insert_query = """
            INSERT INTO Loans (Term, NoEmp, GrAppv, Disbursement, RealEstate, AHP_Score, AI_Status, Recommendation, Explanation) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_query, (
            web_data[0], web_data[1], web_data[2], web_data[3], web_data[4], 
            float(final_score), ai_status, recommendation, explanation
        ))
        conn.commit()
        conn.close()

        return jsonify({
            "score": round(final_score, 2), 
            "ai_status": ai_status, 
            "recommendation": recommendation,
            "explanation": explanation # Trả về luôn để UI hiện ngay lập tức
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)