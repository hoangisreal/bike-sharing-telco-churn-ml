## 1. Cấu trúc Dự án

* bike_sharing_telco_churn_ml_pipeline.ipynb: Tệp Jupyter Notebook chính chứa toàn bộ mã nguồn xử lý dữ liệu, trực quan hóa, huấn luyện mô hình, đánh giá hệ thống và phân tích báo cáo chuyên sâu.
* README.md: Tệp tài liệu hướng dẫn vận hành dự án hiện tại.

## 2. Các Bài toán và Mô hình Triển khai

### Bài toán 1: Dự báo Nhu cầu Sử dụng Phương tiện (Bike Sharing Demand Forecasting)
* Dạng bài toán: Hồi quy phi tuyến tính (Regression).
* Biến mục tiêu: cnt (Tổng lượng xe được thuê trong một giờ cụ thể).
* Ý nghĩa kinh doanh: Hỗ trợ doanh nghiệp vận tải quản lý đội xe tối ưu, điều phối phương tiện thông minh giữa các trạm để giảm thiểu chi phí vận hành và tăng trải nghiệm khách hàng.
* Mô hình chiến thắng: HistGradientBoostingRegressor (kết hợp TransformedTargetRegressor để áp dụng log-transform trên biến mục tiêu).

### Bài toán 2: Dự báo Tỷ lệ Khách hàng Rời bỏ Dịch vụ (Telco Customer Churn Prediction)
* Dạng bài toán: Phân loại nhị phân mất cân bằng nhãn (Imbalanced Binary Classification).
* Biến mục tiêu: Churn (Khách hàng rời bỏ dịch vụ hay ở lại - Yes/No).
* Ý nghĩa kinh doanh: Cho phép bộ phận CRM chủ động nhận diện nhóm khách hàng có rủi ro rời mạng cao để đưa ra các chương trình ưu đãi cá nhân hóa, tối đa hóa Giá trị Trọn đời Khách hàng (CLV) và tối thiểu hóa Chi phí Giữ chân Khách hàng (CRC).
* Mô hình chiến thắng: HistGradientBoostingClassifier.

## 3. Các Giải pháp Kỹ thuật và Phương pháp luận nâng cao

### Ngăn ngừa Rò rỉ Dữ liệu (Data Leakage Prevention)
* Trong bài toán Bike Sharing, loại bỏ hoàn toàn hai biến casual và registered trước khi đưa dữ liệu vào huấn luyện vì tổng của chúng bằng đúng biến mục tiêu cnt.
* Toàn bộ quá trình điền khuyết dữ liệu thiếu (Imputation) và co giãn đặc trưng (Scaling) được đóng gói chặt chẽ trong ColumnTransformer và Scikit-Learn Pipeline. Điều này đảm bảo các thông số thống kê chỉ được tính toán trên tập huấn luyện (Train set) và áp dụng tĩnh lên tập kiểm thử (Test set), ngăn chặn hoàn toàn lỗi rò rỉ thông tin thống kê.
* Sử dụng TimeSeriesSplit (5 folds) cho chuỗi thời gian Bike Sharing nhằm tránh lỗi nhìn trước tương lai (look-ahead bias).
* Sử dụng StratifiedKFold (5 folds) cho dữ liệu phân loại mất cân bằng Telco Churn để bảo toàn tỷ lệ nhãn trong từng fold kiểm thử chéo.

### Kỹ nghệ Đặc trưng Temporal nâng cao (Advanced Feature Engineering)
* Trích xuất các biến ngữ cảnh thời gian: ngày cuối tuần (is_weekend), giờ cao điểm giao thông (rush_hour).
* Mã hóa Chu kỳ lượng giác (Cyclical Encoding): Ánh xạ đặc trưng giờ (0 đến 23) và tháng (1 đến 12) lên đường tròn lượng giác hai chiều thông qua phép biến đổi sin/cos. Điều này giúp bảo toàn khoảng cách thực tế giữa các mốc thời gian tuần hoàn (ví dụ: giờ 23 và giờ 0 cách nhau đúng 1 đơn vị thời gian thay vì 23 đơn vị tuyến tính).

### Học máy trên target Biến đổi (Target Transformation)
* Áp dụng log-transform y' = log(y + 1) trên biến mục tiêu cnt có phân phối lệch phải mạnh thông qua TransformedTargetRegressor. Việc này giúp ổn định phương sai sai số, đưa phân phối target về gần dạng chuẩn, giúp các mô hình học tốt hơn và đưa ra dự báo chính xác hơn. Hàm nghịch đảo sử dụng np.maximum(0, expm1) để loại bỏ rủi ro dự đoán ra số lượng xe âm.

### Tinh chỉnh Ngưỡng Quyết định Thực nghiệm (Decision Threshold Tuning)
* Thay vì sử dụng ngưỡng mặc định 0.5, hệ thống thu thập dự đoán xác suất Out-Of-Fold (OOF) trên tập huấn luyện để vẽ đường cong Precision-Recall và tìm ngưỡng tối ưu hóa F1-Score (khoảng ~0.334). Cách tiếp cận này giúp mô hình tăng đáng kể chỉ số Recall (khả năng phát hiện khách hàng thực sự rời mạng), tối ưu hóa bài toán kinh tế cho doanh nghiệp.

### Giải thích Mô hình phi tuyến tính (Model-Agnostic Explainability)
* Sử dụng thuật toán Permutation Feature Importance để đo lường mức độ ảnh hưởng thực tế của từng đặc trưng dựa trên mức độ suy giảm hiệu năng của mô hình trên tập kiểm thử chưa thấy khi đặc trưng đó bị xáo trộn.

## 4. Thiết lập Môi trường và Hướng dẫn Vận hành

### Các thư viện phụ thuộc (Dependencies)
Dự án yêu cầu cài đặt các thư viện sau:
* python >= 3.8
* kagglehub (Tải dữ liệu tự động qua API)
* pandas (Xử lý dữ liệu bảng)
* numpy (Tính toán số học)
* matplotlib, seaborn (Trực quan hóa dữ liệu và biểu đồ sai số)
* scikit-learn (Xây dựng pipeline, tiền xử lý, mô hình hóa, đánh giá)
* scipy (Hàm phân phối thống kê)
* joblib (Đóng gói mô hình nhị phân)

### Hướng dẫn chạy thử nghiệm
1. cài đặt trực tiếp thông qua cell cài đặt thư viện ở đầu notebook
2. Chạy bằng Kaggle hoặc Google Colab.
3. Mở tệp bike_sharing_telco_churn_ml_pipeline.ipynb.
4. Chọn Run All để hệ thống tự động tải dữ liệu từ Kaggle, thực hiện xử lý dữ liệu, vẽ các biểu đồ phân tích và hiển thị kết quả đánh giá mô hình.
