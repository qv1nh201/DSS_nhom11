from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# 1. Load mô hình AI đã huấn luyện [cite: 238]
try:
    model = pickle.load(open('loan_model.pkl', 'rb'))
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file loan_model.pkl. Hãy chạy file train_model.py trước.")

# 2. Định nghĩa hàm tính toán AHP (Máy lọc định tính) [cite: 158, 247, 277]
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # --- BƯỚC 1: XỬ LÝ AHP ---
        # Nhận ma trận so sánh cặp từ UI [cite: 245]
        matrix = np.array(data['ahp_matrix']) 
        weights, cr = calculate_ahp(matrix)
        
        # Kiểm tra độ nhất quán CR < 10% [cite: 165, 197, 249]
        if cr > 0.1:
            return jsonify({"error": f"Độ nhất quán AHP không đạt (CR={round(cr,4)}). Vui lòng đánh giá lại!"}), 400

        # --- BƯỚC 2: XỬ LÝ AI (DECISION TREE) ---
        # Lấy dữ liệu tài chính (phải đủ 19 trường như lúc train) [cite: 231, 239]
        financial_input = np.array(data['financial_data']).reshape(1, -1)
        
        # Dự báo nhãn rủi ro [cite: 241, 263]
        prediction = model.predict(financial_input)[0]
        ai_status = "An toàn" if prediction == 0 else "Rủi ro"

        # --- BƯỚC 3: TỔNG HỢP QUYẾT ĐỊNH [cite: 200, 265] ---
        # Chấm điểm tiềm năng dựa trên trọng số AHP
        expert_scores = np.array(data['expert_scores']) 
        final_score = np.dot(weights, expert_scores) * 10 
        
        # Logic đưa ra khuyến nghị cuối cùng [cite: 203, 256]
        if final_score >= 60 and ai_status == "An toàn":
            recommendation = "Duyệt vay"
        elif ai_status == "Rủi ro":
            recommendation = "Từ chối (Rủi ro tài chính cao)"
        else:
            recommendation = "Cần thẩm định thêm"

        return jsonify({
            "score": round(final_score, 2),
            "ai_status": ai_status,
            "recommendation": recommendation,
            "cr": round(cr, 4)
        })
    except Exception as e:
        return jsonify({"error": f"Lỗi hệ thống: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)