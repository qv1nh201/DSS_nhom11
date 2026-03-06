import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# 1. Đọc dữ liệu [cite: 269]
df = pd.read_csv('data.csv')

# 2. Tiền xử lý dữ liệu [cite: 79, 270]
# Loại bỏ cột gây nhiễu và cột chữ không cần thiết [cite: 270]
cols_to_drop = ['Selected', 'LoanNr_ChkDgt', 'Name', 'City', 'State', 'Zip', 
                'Bank', 'BankState', 'ApprovalDate', 'ChgOffDate', 'DisbursementDate', 'MIS_Status']
X = df.drop(columns=cols_to_drop)

# Chuyển đổi cột kết quả 'Default' (đã là số 0/1 trong file của bạn) làm nhãn [cite: 281]
y = df['Default'] 

# Xử lý các cột chữ còn lại sang số (Mã hóa) 
X['RevLineCr'] = X['RevLineCr'].map({'Y': 1, 'N': 0}).fillna(0)
X['LowDoc'] = X['LowDoc'].map({'Y': 1, 'N': 0}).fillna(0)

# Xử lý giá trị trống (Missing values) [cite: 271]
X = X.fillna(X.median(numeric_only=True))

# 3. Chia dữ liệu 80/20 [cite: 280]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Huấn luyện mô hình [cite: 81, 281]
# Sử dụng tiêu chí Entropy như trong đề cương nhóm [cite: 173, 282]
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Kiểm tra độ chính xác [cite: 83, 283]
accuracy = model.score(X_test, y_test)
print(f"--- Huấn luyện thành công! ---")
print(f"Độ chính xác mô hình: {round(accuracy * 100, 2)}%")

# 6. Xuất file model [cite: 302]
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Đã tạo file loan_model.pkl")