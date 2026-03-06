-- Tạo bảng lưu hồ sơ vay [cite: 189, 285]
CREATE TABLE Loans (
    LoanID INT PRIMARY KEY IDENTITY(1,1),
    BusinessName NVARCHAR(255),
    Revenue FLOAT,               -- Doanh thu [cite: 188, 232]
    Employees INT,               -- Số lượng nhân sự [cite: 188, 232]
    AHP_Score FLOAT,             -- Điểm từ AHP [cite: 201]
    AI_Status NVARCHAR(50),      -- Kết quả từ Decision Tree [cite: 202]
    FinalDecision NVARCHAR(50),  -- Quyết định cuối cùng [cite: 203]
    CreatedAt DATETIME DEFAULT GETDATE()
);

-- Tạo bảng quản lý người dùng [cite: 181, 212]
CREATE TABLE Users (
    UserID INT PRIMARY KEY IDENTITY(1,1),
    Username VARCHAR(50) UNIQUE,
    PasswordHash VARCHAR(255),
    Role NVARCHAR(20) -- Admin hoặc Nhân viên tín dụng [cite: 184]
);