from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import pyodbc

app = Flask(__name__)

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH AI & CẤU HÌNH TRỌNG SỐ AHP
# ==========================================
try:
    model = pickle.load(open('loan_model.pkl', 'rb'))
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file loan_model.pkl. Hãy chạy file train_model.py trước.")

# Lấy trực tiếp kết quả CR=0.027 từ file Excel "MA TRẬN SO SÁNH CẶP.csv" của nhóm
# Thứ tự: [Tài chính, Uy tín, Khả thi, Founder]
CRITERIA_WEIGHTS = np.array([0.5378, 0.2745, 0.1285, 0.0591])

# Quy ước điểm số cho các mức đánh giá định tính (AHP Tuyệt đối)
INTENSITY_SCORES = {
    "Tốt": 1.0,
    "Khá": 0.5,
    "Kém": 0.1
}

def calculate_ahp(matrix):
    """Hàm này giữ lại chỉ để dùng cho API /api/ahp_weights (Vẽ biểu đồ Admin)"""
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
# 2. CÁC API RENDER GIAO DIỆN & DASHBOARD (Giữ nguyên)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    return render_template('assessment.html')

CRITERIA_WEIGHTS = np.array([0.5378, 0.2745, 0.1285, 0.0591])

@app.route('/api/ahp_weights', methods=['POST'])
def api_ahp_weights():
    # Thêm dòng này để báo cho Python biết là mình muốn sửa biến hệ thống
    global CRITERIA_WEIGHTS 
    
    try:
        data = request.json
        matrix = np.array(data['ahp_matrix'], dtype=float)
        
        if not np.all(matrix > 0) or np.isnan(matrix).any() or np.isinf(matrix).any():
            return jsonify({"error": "Dữ liệu ma trận có chứa số 0."}), 400
            
        weights, cr = calculate_ahp(matrix)
        
        if cr > 0.1:
            return jsonify({"error": f"CR={round(cr,4)} > 0.1"}), 400

        # ---> ĐIỂM ĂN TIỀN LÀ ĐOẠN NÀY <---
        # Nếu CR hợp lệ (<= 0.1), hệ thống sẽ cập nhật bộ trọng số mới
        CRITERIA_WEIGHTS = weights
        print("Đã cập nhật trọng số hệ thống mới:", CRITERIA_WEIGHTS)

        return jsonify({
            "weights": [round(w * 100, 2) for w in weights],
            "cr": round(cr, 4),
            "labels": ["Sức khỏe tài chính", "Uy tín tín dụng", "Tính khả thi dự án", "Năng lực Founder"] 
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def api_dashboard():
    # LƯU Ý: Sửa lại SERVER=... nếu máy bạn dùng SQLEXPRESS như đã bàn ở trên nhé
    conn_str = (
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        r'SERVER=localhost;' # Thêm \SQLEXPRESS nếu cần
        r'DATABASE=DSS_Startup;'
        r'Trusted_Connection=yes;'
    )
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Đếm tổng hồ sơ
        cursor.execute("SELECT COUNT(*) FROM Loans")
        total = cursor.fetchone()[0]
        
        # Đếm hồ sơ duyệt
        cursor.execute("SELECT COUNT(*) FROM Loans WHERE Recommendation = N'Duyệt vay'")
        approved = cursor.fetchone()[0]
        
        # Đếm hồ sơ từ chối
        cursor.execute("SELECT COUNT(*) FROM Loans WHERE Recommendation LIKE N'Từ chối%'")
        rejected = cursor.fetchone()[0]
        
        pending = total - approved - rejected 
        
        # Lấy 5 hồ sơ mới nhất để hiển thị ra bảng
        cursor.execute("SELECT TOP 5 ID, AHP_Score, AI_Status, Recommendation FROM Loans ORDER BY ID DESC")
        rows = cursor.fetchall()
        
        recent_loans = []
        for row in rows:
            recent_loans.append({
                "startup": f"Hồ sơ Startup #{row.ID}",
                "score": row.AHP_Score,
                "ai_status": row.AI_Status,
                "recommendation": row.Recommendation
            })
            
        conn.close()
        
        # Flask yêu cầu bắt buộc phải return jsonify hoặc string
        return jsonify({
            "total": total, "approved": approved, "rejected": rejected,
            "pending": pending, "recent": recent_loans
        })
    except Exception as e:
        print("Lỗi Dashboard API:", e)
        # Trả về lỗi định dạng JSON để JS không bị sập
        return jsonify({"error": str(e)}), 500

# ==========================================
# 3. API CỐT LÕI: THẨM ĐỊNH TỔNG HỢP (ĐÃ SỬA LUỒNG)
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # UI chỉ cần gửi 2 mảng: Dữ liệu tài chính (cho AI) và Dữ liệu định tính (cho AHP)
        web_data = data['financial_data'] # [Term, NoEmp, GrAppv, Disbursement, RealEstate]
        
        # Mảng 3 giá trị định tính (Uy tín, Khả thi, Founder) được nhập từ form UI (Dạng: "Tốt", "Khá", "Kém")
        qualitative_data = data.get('qualitative_data', ["Khá", "Khá", "Khá"]) 

        # --- BƯỚC 1: AI DỰ BÁO SỨC KHỎE TÀI CHÍNH ---
        feature_names = model.feature_names_in_
        input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        
        input_df.at[0, 'Term'] = web_data[0]
        input_df.at[0, 'NoEmp'] = web_data[1]
        input_df.at[0, 'GrAppv'] = web_data[2]
        input_df.at[0, 'DisbursementGross'] = web_data[3]
        input_df.at[0, 'RealEstate'] = web_data[4]
        
        # [Hack Demo]
        if web_data[2] < 5000 or web_data[3] < 2000:
            prediction = 1 
        else:
            prediction = model.predict(input_df)[0]
            
        ai_status = "An toàn" if prediction == 0 else "Rủi ro"

        # --- BƯỚC 2: CHUYỂN ĐỔI KẾT QUẢ AI VÀO MÔ HÌNH AHP ---
        # Nếu AI báo an toàn -> Tài chính Mức Tốt (1.0). Nếu Rủi ro -> Mức Kém (0.1)
        financial_score = INTENSITY_SCORES["Tốt"] if prediction == 0 else INTENSITY_SCORES["Kém"]

        # Gom điểm 4 tiêu chí: [Tài chính (từ AI), Uy tín (từ UI), Khả thi (từ UI), Founder (từ UI)]
        expert_scores = np.array([
            financial_score,
            INTENSITY_SCORES[qualitative_data[0]],
            INTENSITY_SCORES[qualitative_data[1]],
            INTENSITY_SCORES[qualitative_data[2]]
        ])

        # --- BƯỚC 3: TÍNH TỔNG ĐIỂM AHP (AHP Tuyệt đối) ---
        # Nhân ma trận trọng số (weights) với mức điểm cường độ (scores)
        # Điểm thang 100 cho dễ nhìn
        final_score = np.dot(CRITERIA_WEIGHTS, expert_scores) * 100 
        
        # Luồng Logic Đa Cấp
        if ai_status == "Rủi ro" and final_score >= 65: # Điểm vớt cao do chuyên gia đánh giá team Founder tốt
            recommendation = "Xem xét đặc biệt (Giám đốc duyệt)"
        elif final_score >= 65 and ai_status == "An toàn":
            recommendation = "Duyệt vay"
        elif ai_status == "Rủi ro":
            recommendation = "Từ chối (Rủi ro tài chính cao)"
        else:
            recommendation = "Từ chối (Điểm tiềm năng AHP quá thấp)"

        # --- BƯỚC 4: LƯU VÀO SQL SERVER ---
        # --- BƯỚC 4: LƯU VÀO SQL SERVER ---
        # LƯU Ý: Server Name hiện tại là DESKTOP-QEOF2P9\SQLEXPRESS
        conn_str = (
            r'DRIVER={ODBC Driver 17 for SQL Server};'
            r'SERVER=localhost;' 
            r'DATABASE=DSS_Startup;'
            r'Trusted_Connection=yes;'
        )
        
        try:
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            
            insert_query = '''
                INSERT INTO Loans (Term, NoEmp, GrAppv, Disbursement, RealEstate, AHP_Score, AI_Status, Recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            cursor.execute(insert_query, (
                web_data[0], web_data[1], web_data[2], web_data[3], web_data[4], 
                float(final_score), ai_status, recommendation
            ))
            conn.commit()
            conn.close()
            print("Đã lưu hồ sơ thành công vào SQL Server!")
            
        except Exception as db_err:
            print("Lỗi khi lưu vào Database:", db_err)
            # Hệ thống vẫn chạy tiếp dù lỗi DB, không làm sập Web

        # --- BƯỚC 5: TRẢ KẾT QUẢ CHO GIAO DIỆN ---
        return jsonify({
            "score": round(final_score, 2),
            "ai_status": ai_status,
            "recommendation": recommendation,
            "ahp_breakdown": {
                "tai_chinh": financial_score * CRITERIA_WEIGHTS[0] * 100,
                "uy_tin": expert_scores[1] * CRITERIA_WEIGHTS[1] * 100,
                "kha_thi": expert_scores[2] * CRITERIA_WEIGHTS[2] * 100,
                "founder": expert_scores[3] * CRITERIA_WEIGHTS[3] * 100
            }
        })

    except Exception as e:
        import traceback
        return jsonify({"error": f"Lỗi hệ thống lõi: {str(e)}", "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)