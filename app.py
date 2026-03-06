from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import pyodbc

app = Flask(__name__)

# 1. Load mô hình AI đã huấn luyện
try:
    model = pickle.load(open('loan_model.pkl', 'rb'))
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file loan_model.pkl. Hãy chạy file train_model.py trước.")

# 2. Định nghĩa hàm tính toán AHP (Máy lọc định tính)
def calculate_ahp(matrix):
    # Tính Vector trọng số (Eigenvector)
    column_sums = np.sum(matrix, axis=0)
    norm_matrix = matrix / column_sums
    weights = np.mean(norm_matrix, axis=1)
    
    # Tính chỉ số nhất quán CR
    n = len(matrix)
    lamda_max = np.sum(column_sums * weights)
    ci = (lamda_max - n) / (n - 1)
    # Bảng số ngẫu nhiên Saaty (RI)
    ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12} 
    cr = ci / ri_dict.get(n, 1.0)
    return weights, cr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    return render_template('assessment.html')
@app.route('/api/ahp_weights', methods=['POST'])
def api_ahp_weights():
    try:
        data = request.json
        matrix = np.array(data['ahp_matrix'], dtype=float)
        
        # Chặn lỗi nhập sai (có số 0)
        if not np.all(matrix > 0) or np.isnan(matrix).any() or np.isinf(matrix).any():
            return jsonify({"error": "Dữ liệu ma trận AHP có chứa số 0. Vui lòng nhập 1-9!"}), 400

        weights, cr = calculate_ahp(matrix)
        
        # Báo lỗi nếu CR > 10%
        if cr > 0.1:
            return jsonify({"error": f"Độ nhất quán AHP không đạt (CR={round(cr,4)}). Vui lòng đánh giá lại ma trận!"}), 400

        # Quy đổi ra phần trăm để vẽ biểu đồ tròn
        weights_percent = [round(w * 100, 2) for w in weights]
        
        return jsonify({
            "weights": weights_percent,
            "cr": round(cr, 4),
            "labels": ["Năng lực Founder", "Ý tưởng SP", "Thị trường"] 
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/dashboard', methods=['GET'])
def api_dashboard():
    # LƯU Ý: Vẫn dùng đúng chuỗi kết nối của máy bạn
    conn_str = (
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        r'SERVER=localhost;' 
        r'DATABASE=DSS_Startup;'
        r'Trusted_Connection=yes;'
    )
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # 1. Đếm các trạng thái hồ sơ
        cursor.execute("SELECT COUNT(*) FROM Loans")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Loans WHERE Recommendation = N'Duyệt vay'")
        approved = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Loans WHERE Recommendation LIKE N'Từ chối%'")
        rejected = cursor.fetchone()[0]
        
        pending = total - approved - rejected # Số còn lại là đang xem xét
        
        # 2. Lấy 5 hồ sơ mới nhất vừa chấm xong
        cursor.execute("SELECT TOP 5 ID, AHP_Score, AI_Status, Recommendation FROM Loans ORDER BY ID DESC")
        rows = cursor.fetchall()
        
        recent_loans = []
        for row in rows:
            recent_loans.append({
                "startup": f"Hồ sơ Startup #{row.ID}", # Vì CSDL mình chưa có cột tên, nên tạm gọi theo ID
                "score": row.AHP_Score,
                "ai_status": row.AI_Status,
                "recommendation": row.Recommendation
            })
            
        conn.close()
        
        # Trả dữ liệu về cho trang Web
        return jsonify({
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
            "recent": recent_loans
        })
    except Exception as e:
        print("Lỗi Dashboard API:", e)
        return jsonify({"error": str(e)}), 500
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # --- BƯỚC 1: XỬ LÝ AHP ---
        matrix = np.array(data['ahp_matrix'], dtype=float)
        
        # CHẶN LỖI: Kiểm tra xem có số 0 hoặc giá trị rỗng (NaN/Inf) trong ma trận không
        if not np.all(matrix > 0) or np.isnan(matrix).any() or np.isinf(matrix).any():
            return jsonify({
                "error": "Dữ liệu ma trận AHP không hợp lệ (có chứa số 0 hoặc để trống). Vui lòng nhập thang điểm 1-9!"
            }), 400

        weights, cr = calculate_ahp(matrix)
        
        # Kiểm tra độ nhất quán CR < 10%
        if cr > 0.1:
            return jsonify({"error": f"Độ nhất quán AHP không đạt (CR={round(cr,4)}). Vui lòng đánh giá lại!"}), 400

        # --- BƯỚC 2: XỬ LÝ AI (DECISION TREE) ---
        # 1. Lấy mảng 5 con số từ UI gửi lên (Term, NoEmp, GrAppv, Disbursement, RealEstate)
        web_data = data['financial_data']
        
        # 2. Trích xuất danh sách tên cột gốc mà AI đã học từ file CSV
        feature_names = model.feature_names_in_
        
        # 3. Tạo một bảng dữ liệu (DataFrame) có 1 dòng, chứa đủ các cột (mặc định là 0)
        input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        
        # 4. Lắp 5 con số từ Web vào đúng tên cột tương ứng
        input_df.at[0, 'Term'] = web_data[0]
        input_df.at[0, 'NoEmp'] = web_data[1]
        input_df.at[0, 'GrAppv'] = web_data[2]
        input_df.at[0, 'DisbursementGross'] = web_data[3]
        input_df.at[0, 'RealEstate'] = web_data[4]
        
        # 5. Đưa bảng dữ liệu đã chuẩn hóa vào AI dự báo
        prediction = model.predict(input_df)[0]
        ai_status = "An toàn" if prediction == 0 else "Rủi ro"

        # --- BƯỚC 3: TỔNG HỢP QUYẾT ĐỊNH ---
        expert_scores = np.array(data['expert_scores']) 
        final_score = np.dot(weights, expert_scores) * 10 
        
        if final_score >= 60 and ai_status == "An toàn":
            recommendation = "Duyệt vay"
        elif ai_status == "Rủi ro":
            recommendation = "Từ chối (Rủi ro tài chính cao)"
        else:
            recommendation = "Cần thẩm định thêm"

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

        # --- BƯỚC 5: TRẢ KẾT QUẢ VỀ WEB ---
        return jsonify({
            "score": round(final_score, 2),
            "ai_status": ai_status,
            "recommendation": recommendation,
            "cr": round(cr, 4)
        })

    except Exception as e:
        # Nếu có lỗi lớn ở Bước 1, 2, 3 thì mới báo sập hệ thống (500)
        return jsonify({"error": f"Lỗi hệ thống: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)