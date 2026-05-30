# Viva Guide cho notebook `bike_sharing_telco_churn_ml_pipeline.ipynb`

Tài liệu này giúp bạn nắm được toàn bộ luồng làm trong notebook và chuẩn bị trả lời vấn đáp. Cách học tốt nhất là đọc theo 2 vòng:

1. Đọc phần **quy trình từng bước** để hiểu notebook đang làm gì.
2. Đọc phần **kiến thức trọng tâm và câu hỏi vấn đáp** để biết vì sao làm như vậy.

Notebook gồm 2 bài toán:

- **Bike Sharing Demand**: hồi quy, dự báo `cnt`, tức tổng lượt thuê xe theo từng giờ.
- **Telco Customer Churn**: phân loại nhị phân, dự đoán khách hàng có churn hay không.

---

## Phần 1: Chi tiết các bước trong toàn bộ notebook

## 0. Thiết lập môi trường

Notebook bắt đầu bằng cell:

```python
%pip install -q kagglehub pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

Ý nghĩa:

- Cài thư viện cần thiết trong Colab.
- `kagglehub` dùng để tải dataset từ Kaggle.
- `pandas`, `numpy` dùng xử lý dữ liệu.
- `matplotlib`, `seaborn` dùng vẽ biểu đồ.
- `scikit-learn` dùng preprocessing, model, cross-validation, metrics.
- `scipy` dùng `loguniform` cho random search hyperparameter.

Sau đó notebook import thư viện và cấu hình:

- `RANDOM_STATE = 42`: cố định random để kết quả có thể tái lập.
- `np.random.seed(RANDOM_STATE)`: cố định seed cho NumPy.
- `pd.set_option(...)`: hiển thị bảng dễ đọc hơn.
- `sns.set_theme(...)`: thống nhất style biểu đồ.

Khi vấn đáp, bạn có thể nói: bước setup đảm bảo môi trường Colab có đủ dependency và kết quả random có tính tái lập.

## 0.1. Utility functions

Notebook định nghĩa các hàm dùng lại:

- `download_dataset(slug)`: tải dataset Kaggle, in đường dẫn và file CSV có trong dataset.
- `find_csv(root, preferred_name)`: chọn đúng file CSV cần dùng.
- `make_ohe()`: tạo `OneHotEncoder` tương thích nhiều phiên bản scikit-learn.
- `show_missing(df)`: thống kê số lượng và tỷ lệ missing theo từng cột.
- `regression_metrics(y_true, y_pred)`: tính MAE, RMSE, R2 cho bài toán Bike.
- `positive_expm1(x)`: inverse transform an toàn cho log-target, không cho dự báo âm.
- `classification_metrics(y_true, y_score, threshold)`: tính accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
- `estimator_supports_parameter(...)`: kiểm tra model có hỗ trợ tham số nào đó không.
- `plot_perm_importance(...)`: đo permutation importance để giải thích mô hình.

Điểm quan trọng: các helper làm code ngắn hơn, tránh lặp, và giúp metric/plot được tính nhất quán giữa các phần.

---

# 1. Bike Sharing Demand Forecasting

## 1.1. Tải dữ liệu Bike

Notebook tải dataset:

```python
bike_root = download_dataset('lakshmi25npathi/bike-sharing-dataset')
bike_csv = find_csv(bike_root, 'hour.csv')
bike_df = pd.read_csv(bike_csv)
bike_df['dteday'] = pd.to_datetime(bike_df['dteday'])
```

Ý nghĩa:

- Dùng file `hour.csv` vì bài toán là dự báo theo từng giờ.
- Chuyển `dteday` thành datetime để sort theo thời gian và vẽ trend.
- In `shape`, `head()`, `info()` để kiểm tra dữ liệu ban đầu.

Output chính:

- Dataset có `17,379` dòng và `17` cột.
- Không có missing trong dữ liệu Bike gốc.
- Target là `cnt`.
- `casual` và `registered` cộng lại thành `cnt`, nên không được đưa vào feature khi train.

Câu trả lời vấn đáp ngắn: em tải dữ liệu hourly Bike, kiểm tra shape/dtype/missing, xác định target `cnt`, đồng thời nhận diện `casual` và `registered` là leakage vì chúng tạo thành target.

## 1.2. EDA Bike

Mục tiêu EDA Bike là hiểu demand theo giờ thay đổi vì những yếu tố nào.

Notebook kiểm tra:

- Shape và duplicate.
- Thống kê mô tả bằng `describe`.
- Missing bằng `show_missing`.
- Phân phối `cnt` và `log1p(cnt)`.
- Top 1% peak demand.
- Trend theo ngày và rolling mean 14 ngày.
- Demand theo mùa, giờ, ngày làm việc, thời tiết.
- Heatmap `weekday x hour`.
- Correlation numeric.
- Scatter `temp` vs `cnt`.

## Ý nghĩa các bảng EDA Bike

### Bảng shape, duplicate, missing

Output:

- `Rows, columns: (17379, 17)`.
- `Duplicate rows: 0`.
- Missing count của tất cả cột đều bằng 0.

Kết luận:

- Dữ liệu sạch ở mức cơ bản.
- Không cần xử lý duplicate/missing lớn.
- Trọng tâm chuyển sang feature engineering và chống leakage.

### Bảng target stats

Output:

- `cnt` mean khoảng `189.46`, median `142`, max `977`.
- Skew của `cnt` khoảng `1.2774`.
- Skew của `log1p(cnt)` khoảng `-0.8182`.

Ý nghĩa:

- `cnt` lệch phải: nhiều giờ demand thấp/trung bình, một số giờ demand rất cao.
- RMSE sẽ nhạy với lỗi ở giờ peak.
- Có cơ sở thử mô hình log-target, vì log làm phân phối bớt lệch.

### Bảng top 1% demand

Output:

- Ngưỡng top 1% demand khoảng `782.22`.
- Các giờ cao nhất chủ yếu rơi vào giờ `17-18`, ngày làm việc, tháng 9-10, thời tiết tốt.

Ý nghĩa:

- Peak demand không ngẫu nhiên.
- Cần feature về giờ cao điểm, ngày làm việc, mùa, thời tiết.
- Nếu vận hành thật, dự báo peak quan trọng vì thiếu xe giờ cao điểm gây ảnh hưởng lớn.

## Ý nghĩa từng biểu đồ EDA Bike

### 1. Histogram `cnt`

Biểu đồ cho thấy phân phối target gốc.

Cách đọc:

- Nếu cột cao tập trung ở vùng nhỏ và đuôi kéo dài sang phải, target lệch phải.
- Demand có nhiều giờ bình thường và ít giờ cực cao.

Kết luận:

- Cần model xử lý phi tuyến tốt.
- Nên báo cáo cả MAE và RMSE.
- RMSE sẽ phản ánh lỗi ở peak demand mạnh hơn.

### 2. Histogram `log1p(cnt)`

Biểu đồ cho thấy target sau biến đổi log.

Cách đọc:

- Nếu phân phối bớt lệch hơn, log-target có thể giúp model học ổn định hơn.

Kết luận:

- Notebook thử `LogTarget_HistGradientBoostingRegressor`.
- Tuy nhiên log-target chỉ là ứng viên; model cuối vẫn chọn bằng CV RMSE.

### 3. Daily demand và rolling mean 14 ngày

Đường daily demand cho thấy tổng lượt thuê từng ngày. Rolling mean 14 ngày làm mượt trend.

Cách đọc:

- Daily line dao động mạnh là biến động ngắn hạn.
- Rolling mean tăng/giảm theo giai đoạn là dấu hiệu mùa vụ/trend.

Kết luận:

- Dữ liệu có yếu tố thời gian rõ.
- Không nên random split cho Bike.
- Cần time-based split và `TimeSeriesSplit`.

### 4. Nhu cầu trung bình theo mùa

Barplot theo `season` so sánh demand trung bình giữa các mùa.

Cách đọc:

- Mùa có demand cao/thấp khác nhau thì `season` là feature quan trọng.

Kết luận:

- Giữ `season`, `mnth`.
- Tạo thêm `mnth_sin`, `mnth_cos` để biểu diễn chu kỳ năm.

### 5. Boxplot `cnt` theo giờ

Boxplot cho thấy median, độ phân tán và outlier của demand theo từng giờ.

Cách đọc:

- Giờ khuya thường demand thấp.
- Giờ đi làm/tan làm thường có median và outlier cao.

Kết luận:

- `hr` là feature rất quan trọng.
- Tạo `rush_hour`.
- Vì giờ có tính chu kỳ, tạo `hr_sin`, `hr_cos`.

### 6. Pointplot theo giờ và `workingday`

Biểu đồ so sánh pattern giữa ngày làm việc và không làm việc.

Cách đọc:

- Ngày làm việc thường có hai đỉnh sáng/chiều.
- Ngày không làm việc thường dàn đều hơn giữa ngày.

Kết luận:

- Nên tạo interaction `rush_hour_workingday`.
- Demand phụ thuộc đồng thời vào giờ và loại ngày.

### 7. Heatmap `weekday x hour`

Heatmap thể hiện demand trung bình theo ngày trong tuần và giờ trong ngày.

Cách đọc:

- Ô màu đậm là tổ hợp ngày-giờ demand cao.
- Nếu có vùng đậm theo hàng/cột, dữ liệu có pattern lịch rõ.

Kết luận:

- Giữ `weekday`, `hr`.
- Tạo `weekday_sin`, `weekday_cos`, `hr_sin`, `hr_cos`.
- Đây là bằng chứng trực quan cho feature chu kỳ.

### 8. Nhu cầu trung bình theo thời tiết

Barplot theo `weathersit` kiểm tra demand dưới các điều kiện thời tiết.

Cách đọc:

- Thời tiết tốt thường demand cao hơn thời tiết xấu.

Kết luận:

- Giữ `weathersit`, `temp`, `atemp`, `hum`, `windspeed`.
- Tạo interaction `temp_hum`, `temp_windspeed`.

### 9. Correlation heatmap numeric

Heatmap tương quan cho biết quan hệ tuyến tính giữa các biến số.

Cách đọc:

- `casual` và `registered` tương quan rất mạnh với `cnt`.
- `temp`/`atemp` có quan hệ với demand.
- `hum` có thể có quan hệ ngược chiều.

Kết luận quan trọng:

- Không dùng `casual`, `registered` khi train vì `cnt = casual + registered`.
- Correlation dùng cho EDA, không đồng nghĩa được đưa hết vào model.

### 10. Scatter `temp` vs `cnt`, tô màu top 1%

Scatter kiểm tra peak demand có tập trung ở vùng nhiệt độ nào không.

Cách đọc:

- Peak thường xuất hiện trong vùng nhiệt độ dễ chịu, nhưng không chỉ do nhiệt độ.

Kết luận:

- `temp` có tín hiệu, nhưng peak còn phụ thuộc giờ/ngày/mùa.
- Model cần kết hợp nhiều feature, không thể chỉ dùng thời tiết.

## 1.3. Feature engineering Bike

Notebook sort dữ liệu theo `dteday`, `hr`, rồi tạo feature:

### Nhóm lịch/thời gian

- `is_weekend`: cuối tuần hay không.
- `rush_hour`: giờ cao điểm.
- `hr_sin`, `hr_cos`: mã hóa chu kỳ 24 giờ.
- `weekday_sin`, `weekday_cos`: mã hóa chu kỳ 7 ngày.
- `mnth_sin`, `mnth_cos`: mã hóa chu kỳ 12 tháng.

Lý do:

- Thời gian có tính chu kỳ.
- Số giờ 23 và 0 gần nhau, nhưng nếu dùng số thô thì khoảng cách là 23.
- Sin/cos giúp biểu diễn tính vòng tròn.

### Nhóm weather interaction

- `temp_hum = temp * hum`.
- `temp_windspeed = temp * windspeed`.

Lý do:

- Tác động của nhiệt độ có thể thay đổi theo độ ẩm/gió.
- Interaction giúp model học tình huống thời tiết kết hợp.

### Nhóm commute interaction

- `rush_hour_workingday`.

Lý do:

- Giờ cao điểm ngày làm việc khác cuối tuần.
- EDA pointplot đã cho thấy pattern này.

### Nhóm lag/rolling

- `cnt_lag_1`: demand giờ trước.
- `cnt_lag_24`: cùng giờ hôm trước.
- `cnt_lag_168`: cùng giờ tuần trước.
- `cnt_roll_mean_3`: trung bình 3 giờ trước.
- `cnt_roll_mean_24`: trung bình 24 giờ trước.
- `cnt_roll_std_24`: độ biến động 24 giờ trước.

Lý do:

- Demand theo giờ có tự tương quan.
- Chỉ dùng giá trị quá khứ bằng `shift`, tránh nhìn vào target hiện tại.

## 1.4. Chống leakage Bike

Notebook loại:

- `instant`: ID dòng.
- `dteday`: datetime raw.
- `casual`, `registered`: thành phần tạo ra target.
- `cnt`: target.

Câu trả lời vấn đáp:

`casual` và `registered` phải loại vì nếu đưa vào model thì model gần như biết đáp án, do `cnt = casual + registered`. Đây là target leakage.

## 1.5. Split và preprocessing Bike

Notebook chia:

- 80% đầu làm train.
- 20% cuối làm test.
- Không shuffle.

Output:

- Train: `(13,903, 29)`.
- Test: `(3,476, 29)`.
- Train date: `2011-01-01` đến `2012-08-07`.
- Test date: `2012-08-07` đến `2012-12-31`.

Lý do:

- Bike là dữ liệu thời gian.
- Random split có thể cho model học pattern từ tương lai, làm metric quá đẹp.

Preprocessing:

- Numeric: median imputation + `StandardScaler`.
- Categorical: most-frequent imputation + `OneHotEncoder`.
- Dùng `ColumnTransformer`.

Lý do:

- Numeric/categorical cần xử lý khác nhau.
- Đặt preprocessing trong pipeline để tránh leakage trong cross-validation.

## 1.6. Model comparison Bike

Notebook benchmark:

- `DummyRegressor`: baseline dự báo trung bình.
- `RandomForestRegressor`.
- `HistGradientBoostingRegressor`.
- `LogTarget_HistGradientBoostingRegressor`.

Validation:

- `TimeSeriesSplit(n_splits=5)`.
- `RandomizedSearchCV`.
- Scoring: `neg_root_mean_squared_error`.

Lý do:

- `DummyRegressor` là mốc tối thiểu.
- Tree/boosting phù hợp dữ liệu tabular phi tuyến.
- `TimeSeriesSplit` giữ đúng logic train quá khứ, validate tương lai.
- Chọn model bằng CV RMSE, không chọn bằng test.

Output chính:

- Best model: `HistGradientBoostingRegressor`.
- CV RMSE khoảng `51.4660`.
- Holdout: MAE `29.7488`, RMSE `46.1534`, R2 `0.9562`.
- Baseline RMSE khoảng `232.6084`, R2 âm.

Kết luận:

- Model thật vượt baseline rất xa.
- HGB học tốt quan hệ phi tuyến giữa giờ, mùa, thời tiết và lag demand.

## 1.7. Đánh giá cuối Bike

Notebook đánh giá:

- Bảng MAE/RMSE/R2.
- Scatter actual vs predicted.
- Residual theo thời gian.
- Histogram residual.
- Residual vs predicted.
- Permutation importance.

### Actual vs Predicted

Cách đọc:

- Điểm càng gần đường chéo càng tốt.
- Nếu điểm lệch nhiều ở vùng demand cao, model có thể dự báo peak chưa tốt.

### Residual theo thời gian

Residual = actual - predicted.

Cách đọc:

- Residual quanh 0 là tốt.
- Residual dương liên tục: model under-predict.
- Residual âm liên tục: model over-predict.

### Histogram residual

Cách đọc:

- Nếu phân phối residual tập trung quanh 0, sai số tương đối cân bằng.
- Đuôi dài cho biết có một số lỗi lớn.

### Residual vs predicted

Cách đọc:

- Nếu residual tăng theo predicted, model sai nhiều ở demand cao.
- Nếu không có pattern rõ, model ổn định hơn.

### Permutation importance Bike

Top feature:

- `cnt_lag_1`.
- `hr_cos`, `hr_sin`, `hr`.
- `rush_hour_workingday`.
- `cnt_lag_24`, `cnt_lag_168`.
- `weathersit`, `workingday`, `temp`, `atemp`, `hum`.

Kết luận:

- Demand hiện tại phụ thuộc rất mạnh vào demand giờ trước.
- Chu kỳ giờ/ngày và giờ cao điểm là tín hiệu quan trọng.
- Thời tiết có ích nhưng yếu hơn lag/time features.

---

# 2. Telco Customer Churn Prediction

## 2.1. Tải dữ liệu Telco

Notebook tải dataset:

```python
telco_root = download_dataset('blastchar/telco-customer-churn')
telco_csv = find_csv(telco_root, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
telco_df = pd.read_csv(telco_csv)
telco_df['TotalCharges'] = pd.to_numeric(telco_df['TotalCharges'], errors='coerce')
```

Ý nghĩa:

- `TotalCharges` trong dữ liệu gốc là object vì có giá trị rỗng.
- `pd.to_numeric(..., errors='coerce')` chuyển giá trị không hợp lệ thành `NaN`.

Output:

- Dataset có `7,043` khách hàng, `21` cột.
- `TotalCharges` có `7,032` non-null, tức thiếu `11` dòng.
- Target là `Churn`.

Kết luận:

- Cần xử lý missing `TotalCharges`.
- `customerID` là định danh, không dùng làm feature.

## 2.2. EDA Telco

Mục tiêu EDA Telco là hiểu nhóm khách hàng nào churn cao và vì sao accuracy không đủ.

Notebook kiểm tra:

- Shape, duplicate.
- `describe(include='all')`.
- Missing values.
- Churn distribution.
- Missing `TotalCharges`.
- Churn rate theo `Contract`, `PaymentMethod`, `InternetService`, `tenure_group`.
- Churn rate theo service add-ons.
- Countplot churn.
- Boxplot tenure/monthly charges theo churn.
- KDE TotalCharges theo churn.
- Barplot churn rate theo nhóm.
- Correlation heatmap numeric.

## Ý nghĩa các bảng EDA Telco

### Churn distribution

Output:

- `No`: khoảng `73.46%`.
- `Yes`: khoảng `26.54%`.

Kết luận:

- Dữ liệu mất cân bằng.
- Accuracy không đủ.
- Cần precision, recall, F1, ROC-AUC, PR-AUC.

### Missing TotalCharges

Output:

- 11 dòng missing.
- Tất cả có `tenure = 0`.
- Tất cả `Churn = No`.

Kết luận:

- Đây có thể là khách hàng mới chưa phát sinh tổng phí.
- Không cần drop cột.
- Impute trong pipeline là hợp lý.

### Churn rate theo Contract

Output:

- `Month-to-month`: khoảng `42.71%`.
- `One year`: khoảng `11.27%`.
- `Two year`: khoảng `2.83%`.

Kết luận:

- `Contract` là tín hiệu churn rất mạnh.
- Hợp đồng ngắn hạn có rủi ro cao.

### Churn rate theo PaymentMethod

Output:

- `Electronic check`: khoảng `45.29%`, cao nhất.

Kết luận:

- Payment method có khả năng phân biệt churn.
- Có thể liên quan hành vi thanh toán hoặc nhóm khách hàng.

### Churn rate theo InternetService

Output:

- `Fiber optic`: khoảng `41.89%`, cao hơn DSL và No internet service.

Kết luận:

- Fiber optic là nhóm cần phân tích sâu.
- Không kết luận nhân quả, chỉ kết luận đây là tín hiệu dự báo.

### Churn rate theo tenure_group

Output:

- Nhóm `1-6` tháng churn khoảng `53.33%`.
- Nhóm tenure cao hơn churn thấp hơn.

Kết luận:

- Khách hàng mới là nhóm rủi ro cao.
- Onboarding/chăm sóc sớm rất quan trọng.

### Service add-on churn

Output:

- Nhóm không có `OnlineSecurity`, `TechSupport`, `OnlineBackup`, `DeviceProtection` có churn rate cao.

Kết luận:

- Add-on service có thể thể hiện mức độ gắn kết.
- Nên giữ các biến dịch vụ trong mô hình.

## Ý nghĩa từng biểu đồ EDA Telco

### 1. Countplot `Churn`

Cho thấy số khách churn ít hơn nhiều so với no-churn.

Kết luận:

- Không thể chỉ dùng accuracy.
- Baseline đoán `No churn` có accuracy cao nhưng vô dụng cho retention.

### 2. Boxplot `tenure` theo `Churn`

Cho thấy nhóm churn thường tenure thấp hơn.

Kết luận:

- `tenure` là feature mạnh.
- Khách hàng mới cần được ưu tiên chăm sóc.

### 3. Boxplot `MonthlyCharges` theo `Churn`

Cho thấy nhóm churn có xu hướng monthly charges cao hơn.

Kết luận:

- `MonthlyCharges` có tín hiệu.
- Giá hoặc cảm nhận giá trị dịch vụ có thể liên quan churn.

### 4. KDE `TotalCharges` theo `Churn`

Cho thấy phân phối tổng phí tích lũy của churn/no-churn.

Kết luận:

- `TotalCharges` có tín hiệu nhưng liên quan mạnh tới `tenure`.
- Không nên diễn giải độc lập một cách nhân quả.

### 5. Churn rate theo `Contract`

Barplot cho thấy month-to-month churn cao nhất.

Kết luận:

- `Contract` nên là feature quan trọng.
- Kinh doanh có thể ưu tiên nhóm month-to-month.

### 6. Churn rate theo `PaymentMethod`

Electronic check có churn rate cao.

Kết luận:

- `PaymentMethod` nên được one-hot encode.
- Có thể dùng insight này để phân nhóm chiến dịch.

### 7. Churn rate theo `InternetService`

Fiber optic có churn cao.

Kết luận:

- Cần giữ `InternetService`.
- Cần đọc cùng `MonthlyCharges`, `Contract`, add-ons.

### 8. Churn rate theo `tenure_group`

Churn giảm khi tenure tăng.

Kết luận:

- Early lifecycle là giai đoạn rủi ro.
- Mô hình có thể giúp phát hiện khách hàng mới rủi ro cao.

### 9. Correlation heatmap numeric

Kiểm tra quan hệ giữa `tenure`, `MonthlyCharges`, `TotalCharges`.

Kết luận:

- `TotalCharges` thường liên quan tới `tenure`.
- Khi giải thích, tránh nói quan hệ nhân quả đơn giản.

## 2.3. Preprocessing Telco

Notebook tạo target:

```python
y_telco = (telco_df['Churn'] == 'Yes').astype(int)
```

Ý nghĩa:

- `1` là churn.
- `0` là no churn.

Feature:

- Drop `Churn`.
- Drop `customerID`.
- Map `SeniorCitizen`: `0/1` thành `No/Yes`.

Lý do:

- `customerID` là định danh, không tổng quát hóa.
- `SeniorCitizen` là category nhị phân, không phải biến liên tục.

Split:

- `train_test_split(..., stratify=y_telco, test_size=0.2)`.

Output:

- Train: `(5,634, 19)`.
- Test: `(1,409, 19)`.
- Positive rate train: `0.26535`.
- Positive rate test: `0.26544`.

Kết luận:

- Stratified split giữ tỷ lệ churn gần như giống nhau ở train/test.
- Test set đại diện tốt hơn cho phân phối target.

Preprocessing:

- Numeric: median imputation + scaling.
- Categorical: most-frequent imputation + one-hot encoding.
- Dùng `ColumnTransformer`.

## 2.4. Model comparison Telco

Notebook benchmark:

- `DummyClassifier`: baseline đoán lớp phổ biến.
- `LogisticRegression`.
- `RandomForestClassifier`.
- `HistGradientBoostingClassifier`.

Validation:

- `StratifiedKFold(n_splits=5, shuffle=True)`.
- `RandomizedSearchCV`.
- Scoring: `roc_auc`.

Lý do chọn ROC-AUC:

- Churn là bài toán ranking rủi ro.
- ROC-AUC không phụ thuộc threshold 0.5.
- Accuracy không đủ vì dữ liệu mất cân bằng.

Class weight:

- Notebook kiểm tra HGB có hỗ trợ `class_weight` không.
- Nếu có thì thêm `class_weight=[None, 'balanced']` vào search space.

Output chính:

- Best model: `HistGradientBoostingClassifier`.
- CV ROC-AUC khoảng `0.8479`.
- Holdout ROC-AUC khoảng `0.8469`.
- PR-AUC khoảng `0.6571`.
- Ở threshold 0.5: accuracy `0.8020`, precision `0.6589`, recall `0.5267`, F1 `0.5854`.

Kết luận:

- HGB có ranking rủi ro tốt nhất theo CV ROC-AUC.
- Logistic có F1 threshold 0.5 cao hơn trong bảng, nhưng notebook chọn HGB vì tiêu chí selection là CV ROC-AUC.

## 2.5. Threshold tuning Telco

Notebook không dùng threshold 0.5 một cách máy móc.

Cách làm:

1. Dự báo probability trên test bằng best model.
2. Dùng `cross_val_predict` trên train để tạo out-of-fold probability.
3. Tính precision-recall curve trên OOF train.
4. Tính F1 theo threshold.
5. Chọn threshold tối ưu F1 trên OOF train.
6. Áp threshold đó lên test.

Vì sao dùng OOF train?

- Mỗi điểm train được dự báo bởi model không fit trên chính điểm đó.
- Tránh chọn threshold trực tiếp trên test.
- Test vẫn là đánh giá cuối.

Output:

- Threshold tối ưu F1: khoảng `0.3274`.
- Threshold 0.5: recall `0.5267`, F1 `0.5854`.
- Threshold tuned: recall `0.7433`, F1 `0.6391`.
- Precision giảm từ `0.6589` xuống `0.5605`.
- Accuracy giảm từ `0.8020` xuống `0.7771`.

Kết luận:

- Threshold thấp hơn giúp bắt nhiều churn hơn.
- Đổi lại liên hệ nhầm nhiều khách hơn.
- Đây là trade-off kinh doanh, không phải lỗi mô hình.

## 2.6. Evaluation plots Telco

### Confusion matrix

Cho biết số:

- True negative: đoán no churn đúng.
- False positive: đoán churn nhưng thực tế no churn.
- False negative: đoán no churn nhưng thực tế churn.
- True positive: đoán churn đúng.

Trong churn, false negative thường nguy hiểm vì bỏ sót khách rời dịch vụ.

### ROC curve

ROC curve thể hiện trade-off TPR/FPR ở mọi threshold.

ROC-AUC:

- `0.5`: như random.
- Gần `1`: ranking rất tốt.
- Notebook đạt khoảng `0.8469`, khá tốt cho churn tabular.

### Precision-Recall curve

PR curve tập trung vào lớp positive là churn.

PR-AUC hữu ích khi dữ liệu mất cân bằng.

### Permutation importance Telco

Top feature:

- `Contract`.
- `tenure`.
- `InternetService`.
- `TotalCharges`.
- `MonthlyCharges`.
- `OnlineSecurity`.
- `TechSupport`.

Kết luận:

- Mô hình học đúng pattern EDA.
- Hợp đồng, thời gian gắn bó, loại internet và dịch vụ hỗ trợ là nhóm tín hiệu chính.

---

# Phần 2: Kiến thức cơ bản và trọng tâm để vấn đáp

## 1. Supervised learning là gì?

Supervised learning là học có nhãn. Dữ liệu train gồm feature `X` và target `y`. Mô hình học quan hệ giữa `X` và `y`, sau đó dự đoán `y` cho dữ liệu mới.

Trong notebook:

- Bike: `X` là thời gian/thời tiết/lag, `y` là `cnt`.
- Telco: `X` là thông tin khách hàng/dịch vụ/chi phí, `y` là churn `0/1`.

## 2. Regression và classification khác gì?

Regression dự đoán giá trị số liên tục.

- Ví dụ: dự báo `cnt`.
- Metric: MAE, RMSE, R2.

Classification dự đoán lớp.

- Ví dụ: churn hay không churn.
- Metric: accuracy, precision, recall, F1, ROC-AUC, PR-AUC.

## 3. EDA là gì?

EDA là Exploratory Data Analysis, phân tích khám phá dữ liệu.

Mục tiêu:

- Kiểm tra chất lượng dữ liệu.
- Hiểu phân phối target.
- Phát hiện missing/outlier/imbalance.
- Tìm pattern để tạo feature.
- Phát hiện nguy cơ leakage.

Trong notebook, EDA không chỉ để vẽ biểu đồ mà dẫn tới quyết định modeling.

## 4. Data leakage là gì?

Data leakage là khi model trong lúc train nhìn thấy thông tin không có sẵn ở thời điểm dự đoán thật.

Ví dụ Bike:

- `casual` + `registered` = `cnt`.
- Nếu dùng `casual`, `registered` để dự đoán `cnt`, model gần như biết đáp án.

Ví dụ Telco:

- Fit imputer/encoder trên toàn bộ dataset trước train/test split cũng là leakage nhẹ.
- Vì vậy preprocessing được đặt trong pipeline.

## 5. Vì sao dùng pipeline?

Pipeline giúp gộp preprocessing và model thành một quy trình thống nhất.

Lợi ích:

- Tránh leakage trong cross-validation.
- Mỗi fold fit imputer/scaler/encoder chỉ trên train fold.
- Code gọn, tái lập, dễ deploy.

## 6. One-hot encoding là gì?

One-hot encoding chuyển categorical thành nhiều cột 0/1.

Ví dụ:

- `Contract = Month-to-month`, `One year`, `Two year`.
- Sau one-hot thành các cột binary.

Lý do:

- Nhiều model sklearn không xử lý string trực tiếp.
- Không tạo thứ tự giả giữa các category.

## 7. Imputation là gì?

Imputation là thay missing value bằng giá trị hợp lý.

Notebook dùng:

- Numeric: median.
- Categorical: most frequent.

Vì sao median?

- Median ít nhạy với outlier hơn mean.

## 8. Scaling là gì?

Scaling chuẩn hóa numeric feature về thang đo tương đương.

Trong notebook dùng `StandardScaler`.

Quan trọng với:

- Logistic Regression.
- Một số mô hình tối ưu bằng gradient.

Tree-based model không quá cần scaling, nhưng pipeline dùng chung cho nhiều model nên vẫn scale numeric.

## 9. Train/test split là gì?

Train set dùng để học model. Test set dùng để đánh giá cuối trên dữ liệu chưa thấy.

Nguyên tắc:

- Không dùng test để chọn model/hyperparameter/threshold.
- Test chỉ dùng để báo cáo cuối.

## 10. Cross-validation là gì?

Cross-validation chia train thành nhiều fold để đánh giá model ổn định hơn.

Trong notebook:

- Bike dùng `TimeSeriesSplit`.
- Telco dùng `StratifiedKFold`.

## 11. Vì sao Bike dùng TimeSeriesSplit?

Bike là dữ liệu thời gian. Nếu dùng KFold random, train có thể chứa dữ liệu tương lai so với validation.

`TimeSeriesSplit` giúp:

- Train trên quá khứ.
- Validate trên tương lai gần.
- Đánh giá thực tế hơn.

## 12. Vì sao Telco dùng StratifiedKFold?

Telco có target mất cân bằng. `StratifiedKFold` giữ tỷ lệ churn/no-churn gần giống nhau trong từng fold.

Lợi ích:

- Metric ổn định hơn.
- Mỗi fold đại diện tốt hơn cho phân phối target.

## 13. Baseline là gì?

Baseline là mô hình đơn giản làm mốc so sánh.

Trong notebook:

- Bike: `DummyRegressor` dự báo trung bình.
- Telco: `DummyClassifier` đoán lớp phổ biến.

Nếu model phức tạp không vượt baseline, model đó không có giá trị.

## 14. RandomizedSearchCV là gì?

Đây là phương pháp tìm hyperparameter.

Thay vì thử toàn bộ tổ hợp như GridSearch, RandomizedSearch thử ngẫu nhiên `n_iter` tổ hợp.

Lợi ích:

- Tiết kiệm thời gian.
- Phù hợp Colab CPU.
- Vẫn tìm được vùng tham số tốt.

## 15. Random Forest là gì?

Random Forest là ensemble của nhiều decision trees.

Đặc điểm:

- Học quan hệ phi tuyến.
- Giảm overfitting so với một cây đơn.
- Mạnh với dữ liệu tabular.

## 16. HistGradientBoosting là gì?

HistGradientBoosting là mô hình boosting trong scikit-learn.

Ý tưởng:

- Xây cây tuần tự.
- Cây sau học phần lỗi còn lại của cây trước.
- Dùng histogram để tăng tốc.

Vì sao phù hợp?

- Dữ liệu tabular.
- Quan hệ phi tuyến.
- Có interaction giữa feature.

## 17. Logistic Regression là gì?

Logistic Regression là mô hình phân loại tuyến tính.

Nó dự đoán xác suất class 1 bằng logistic function.

Ưu điểm:

- Nhanh.
- Dễ diễn giải.
- Baseline mạnh cho dữ liệu one-hot.

Nhược điểm:

- Khó học quan hệ phi tuyến nếu không tạo feature interaction.

## 18. MAE là gì?

MAE = trung bình sai số tuyệt đối.

Ý nghĩa:

- Dễ hiểu theo đơn vị target.
- Bike MAE `29.75` nghĩa là trung bình lệch khoảng 30 lượt thuê mỗi giờ.

## 19. RMSE là gì?

RMSE là căn bậc hai của trung bình bình phương sai số.

Đặc điểm:

- Phạt lỗi lớn mạnh hơn MAE.
- Hữu ích khi peak demand quan trọng.

Bike RMSE `46.15` cho thấy có một số giờ sai số lớn hơn trung bình.

## 20. R2 là gì?

R2 đo tỷ lệ biến thiên target được mô hình giải thích.

- R2 gần 1: tốt.
- R2 bằng 0: tương đương dự báo trung bình.
- R2 âm: tệ hơn dự báo trung bình.

Bike R2 `0.9562` là rất tốt trong holdout.

## 21. Accuracy là gì?

Accuracy = tỷ lệ dự báo đúng.

Nhược điểm:

- Dễ gây hiểu nhầm khi dữ liệu mất cân bằng.

Telco baseline đoán `No churn` vẫn đạt accuracy khoảng 73%, nhưng recall churn bằng 0.

## 22. Precision là gì?

Precision trả lời: trong các khách được dự báo churn, bao nhiêu người thật sự churn?

Precision cao nghĩa là ít liên hệ nhầm.

## 23. Recall là gì?

Recall trả lời: trong các khách thật sự churn, model bắt được bao nhiêu?

Recall cao nghĩa là ít bỏ sót churn.

Trong retention, recall thường rất quan trọng.

## 24. F1 là gì?

F1 là trung bình điều hòa của precision và recall.

Dùng khi cần cân bằng precision/recall.

Notebook chọn threshold tối ưu F1 trên OOF train.

## 25. ROC-AUC là gì?

ROC-AUC đo khả năng xếp hạng positive cao hơn negative.

Ý nghĩa:

- `0.5`: như random.
- `1.0`: hoàn hảo.
- Không phụ thuộc threshold cụ thể.

Telco ROC-AUC khoảng `0.8469`, nghĩa là ranking churn khá tốt.

## 26. PR-AUC là gì?

PR-AUC là diện tích dưới Precision-Recall curve.

Quan trọng khi:

- Lớp positive ít.
- Bài toán mất cân bằng.

Với Telco, PR-AUC hữu ích vì churn chỉ khoảng 26.54%.

## 27. Threshold tuning là gì?

Model classification thường trả về xác suất. Threshold quyết định xác suất bao nhiêu thì gán nhãn churn.

Default thường là 0.5, nhưng không phải lúc nào cũng tốt.

Trong notebook:

- Threshold 0.5 recall churn `0.5267`.
- Threshold tuned 0.3274 recall churn `0.7433`.

Trade-off:

- Threshold thấp: tăng recall, giảm precision.
- Threshold cao: tăng precision, giảm recall.

## 28. False positive và false negative trong churn

False positive:

- Model dự báo churn, thực tế không churn.
- Tốn chi phí chăm sóc nhầm.

False negative:

- Model dự báo không churn, thực tế churn.
- Bỏ lỡ cơ hội giữ chân khách hàng.

Nếu mất khách đắt hơn chăm sóc nhầm, nên ưu tiên recall.

## 29. Permutation importance là gì?

Permutation importance đo tầm quan trọng feature bằng cách:

1. Hoán vị ngẫu nhiên một feature.
2. Dự báo lại.
3. Đo metric xấu đi bao nhiêu.

Nếu metric giảm mạnh, feature đó quan trọng.

Ưu điểm:

- Model-agnostic.
- Dùng được cho pipeline sklearn.

Nhược điểm:

- Feature tương quan cao có thể làm importance bị chia sẻ.
- Tốn thời gian hơn feature importance nội bộ.

---

# Phần 3: Bộ câu hỏi vấn đáp thường gặp

## Câu 1: Notebook này giải quyết những bài toán gì?

Notebook giải quyết 2 bài toán supervised learning. Bike là regression dự báo `cnt`, tổng lượt thuê xe theo giờ. Telco là binary classification dự đoán khách hàng churn hay không.

## Câu 2: Vì sao chọn `cnt` làm target Bike?

Vì `cnt` là tổng nhu cầu thuê xe trong một giờ. Đây là đại lượng cần dự báo để phục vụ điều phối xe và chuẩn bị nguồn lực vận hành.

## Câu 3: Vì sao không dùng `casual` và `registered` làm feature?

Vì `casual + registered = cnt`. Nếu dùng hai cột này, model sẽ biết trực tiếp target. Đó là target leakage.

## Câu 4: Vì sao Bike không random split?

Vì Bike là dữ liệu thời gian. Random split có thể đưa dữ liệu tương lai vào train và quá khứ vào test, làm đánh giá quá lạc quan. Time-based split mô phỏng đúng tình huống dùng quá khứ dự báo tương lai.

## Câu 5: Vì sao tạo `hr_sin` và `hr_cos`?

Vì giờ trong ngày là biến chu kỳ. 23h và 0h gần nhau, nhưng nếu dùng số thô thì model thấy khoảng cách lớn. Sin/cos biểu diễn tính vòng tròn của thời gian.

## Câu 6: Vì sao tạo lag features?

Demand hiện tại thường liên quan demand quá khứ gần. `cnt_lag_1`, `cnt_lag_24`, `cnt_lag_168` giúp mô hình học tính liên tục theo giờ, ngày và tuần.

## Câu 7: Lag feature có bị leakage không?

Không, nếu được tạo bằng `shift` và chỉ dùng giá trị quá khứ. Notebook dùng `shift(1)`, `shift(24)`, `shift(168)`, nên không lấy `cnt` hiện tại làm feature.

## Câu 8: Vì sao chọn HistGradientBoostingRegressor cho Bike?

Vì nó có `cv_RMSE` thấp nhất trong `TimeSeriesSplit`. Nó cũng đạt holdout MAE khoảng `29.75`, RMSE khoảng `46.15`, R2 khoảng `0.9562`, vượt baseline rất xa.

## Câu 9: Bike R2 0.9562 có nghĩa là gì?

Nghĩa là mô hình giải thích được khoảng 95.62% biến thiên của `cnt` trên holdout theo định nghĩa R2. Đây là kết quả rất tốt, nhưng vẫn cần đọc residual để xem lỗi ở peak demand.

## Câu 10: Feature quan trọng nhất Bike là gì?

`cnt_lag_1` là quan trọng nhất. Điều này hợp lý vì nhu cầu giờ hiện tại phụ thuộc mạnh vào nhu cầu ngay giờ trước.

## Câu 11: Vì sao Telco phải chuyển `TotalCharges` sang numeric?

Vì cột này trong CSV gốc chứa chuỗi và một số giá trị rỗng. Nếu không chuyển numeric, pipeline sẽ xử lý sai kiểu dữ liệu và mất tín hiệu tài chính quan trọng.

## Câu 12: Missing `TotalCharges` nói lên điều gì?

Có 11 dòng thiếu `TotalCharges`, tất cả có `tenure = 0`. Có thể đây là khách hàng mới chưa phát sinh tổng phí. Notebook xử lý bằng median imputation trong pipeline.

## Câu 13: Vì sao drop `customerID`?

`customerID` là định danh duy nhất, không có khả năng tổng quát hóa cho khách hàng mới. Nếu one-hot ID, model có thể học thuộc ID thay vì học pattern churn.

## Câu 14: Vì sao Telco dùng stratified split?

Vì churn chỉ chiếm khoảng 26.54%. Stratified split giữ tỷ lệ churn/no-churn gần bằng nhau giữa train và test, giúp metric ổn định hơn.

## Câu 15: Vì sao accuracy không đủ cho Telco?

Dữ liệu mất cân bằng. Nếu đoán tất cả là `No churn`, accuracy vẫn khoảng 73.46%, nhưng không bắt được khách churn nào. Vì vậy cần recall, precision, F1, ROC-AUC, PR-AUC.

## Câu 16: Vì sao chọn ROC-AUC để chọn model Telco?

Vì mục tiêu chính là xếp hạng rủi ro churn. ROC-AUC đo khả năng ranking và không phụ thuộc threshold 0.5.

## Câu 17: Vì sao Logistic có F1 cao hơn nhưng không được chọn?

Trong notebook, tiêu chí chọn model là `cv_roc_auc`, không phải F1 ở threshold 0.5. HGB có CV ROC-AUC cao nhất nên được chọn. Threshold được tune riêng ở bước sau.

## Câu 18: Threshold 0.3274 có ý nghĩa gì?

Nếu xác suất churn >= 0.3274 thì gán nhãn churn. Threshold này được chọn từ OOF train để tối ưu F1, không chọn trực tiếp trên test.

## Câu 19: Vì sao threshold thấp hơn 0.5?

Vì muốn tăng recall, tức bắt nhiều khách churn hơn. Trong retention, bỏ sót khách churn có thể tốn kém hơn liên hệ nhầm.

## Câu 20: Khi threshold giảm, điều gì xảy ra?

Recall tăng, precision giảm. Notebook thấy recall tăng từ `0.5267` lên `0.7433`, nhưng precision giảm từ `0.6589` xuống `0.5605`.

## Câu 21: ROC-AUC và PR-AUC có đổi khi đổi threshold không?

Không. ROC-AUC và PR-AUC đánh giá ranking của score trên mọi threshold. Đổi threshold chỉ làm thay đổi nhãn cứng và các metric như accuracy, precision, recall, F1.

## Câu 22: Feature quan trọng nhất Telco là gì?

`Contract` là feature quan trọng nhất theo permutation importance, tiếp theo là `tenure`, `InternetService`, `TotalCharges`, `MonthlyCharges`, `OnlineSecurity`, `TechSupport`.

## Câu 23: EDA Telco kết luận nhóm nào churn cao?

Nhóm month-to-month, tenure thấp, electronic check, fiber optic, thiếu online security hoặc tech support có churn cao hơn.

## Câu 24: Có thể kết luận `Fiber optic` gây churn không?

Không. EDA và model cho thấy `Fiber optic` liên quan tới churn, nhưng không chứng minh quan hệ nhân quả. Có thể nó đi kèm monthly charges cao hoặc nhóm khách hàng khác.

## Câu 25: Pipeline này có hạn chế gì?

Bike thiếu dữ liệu trạm, dự báo thời tiết tương lai, sự kiện, giao thông. Telco thiếu CLV, chi phí retention, ngân sách chiến dịch và dữ liệu hành vi theo thời gian.

## Câu 26: Nếu triển khai thật Bike cần thêm gì?

Cần dữ liệu theo trạm, weather forecast, sự kiện, lịch vận hành, traffic, backtesting nhiều giai đoạn, monitoring drift và retraining định kỳ.

## Câu 27: Nếu triển khai thật Telco cần thêm gì?

Cần CLV, retention cost, save rate, budget, capacity đội chăm sóc, calibration xác suất, lift/gains/profit analysis và monitoring sau triển khai.

## Câu 28: Vì sao dùng permutation importance thay vì SHAP?

Permutation importance có sẵn trong scikit-learn, nhẹ, không thêm dependency. SHAP mạnh hơn nhưng thêm dependency và phức tạp hơn, không cần thiết trong notebook sklearn-only này.

## Câu 29: Vì sao dùng RandomizedSearchCV thay vì GridSearchCV?

RandomizedSearchCV nhanh hơn vì chỉ thử một số tổ hợp ngẫu nhiên. Với Colab CPU và nhiều model, random search là lựa chọn thực tế hơn.

## Câu 30: Một câu tóm tắt toàn notebook?

Notebook xây hai pipeline sklearn-only: Bike dùng time-aware regression để dự báo nhu cầu theo giờ, Telco dùng classification và threshold tuning để xếp hạng/ràng nhãn rủi ro churn; cả hai đều có EDA, chống leakage, preprocessing pipeline, benchmark model, cross-validation, holdout evaluation và permutation importance.

---

# Phần 4: Kịch bản trình bày ngắn khi vấn đáp

## Trình bày 1 phút

Notebook của em gồm hai bài toán supervised learning. Bài Bike là hồi quy dự báo tổng lượt thuê xe `cnt` theo giờ, còn bài Telco là phân loại nhị phân dự đoán khách hàng churn. Em bắt đầu bằng EDA để hiểu dữ liệu, sau đó tạo feature phù hợp, chống leakage, chia train/test đúng cách, dùng pipeline preprocessing, benchmark nhiều model sklearn bằng cross-validation, chọn model theo metric CV, cuối cùng đánh giá trên holdout và giải thích mô hình bằng permutation importance.

## Trình bày Bike 1 phút

Ở Bike, EDA cho thấy demand có chu kỳ rõ theo giờ, ngày làm việc, mùa và thời tiết; target lệch phải và có peak demand. Vì đây là dữ liệu thời gian, em sort theo ngày-giờ, tạo feature chu kỳ như `hr_sin/hr_cos`, feature giờ cao điểm, weather interaction, và lag/rolling như `cnt_lag_1`, `cnt_lag_24`, `cnt_lag_168`. Em loại `casual`, `registered`, `cnt` để tránh leakage, chia train/test theo thời gian và dùng `TimeSeriesSplit`. Model tốt nhất là `HistGradientBoostingRegressor`, đạt RMSE khoảng `46.15` và R2 khoảng `0.9562` trên holdout.

## Trình bày Telco 1 phút

Ở Telco, EDA cho thấy churn rate khoảng `26.54%`, nên accuracy không đủ. Churn cao ở nhóm month-to-month, tenure thấp, electronic check, fiber optic và thiếu một số dịch vụ hỗ trợ. Em chuyển `TotalCharges` sang numeric, drop `customerID`, map `SeniorCitizen` thành category, split stratified và dùng pipeline one-hot/imputation/scaling. Model được chọn bằng CV ROC-AUC là `HistGradientBoostingClassifier`, ROC-AUC holdout khoảng `0.8469`. Sau đó em tune threshold từ OOF train xuống khoảng `0.3274`, giúp recall churn tăng từ `0.5267` lên `0.7433`.

---

# Phần 5: Các điểm dễ bị hỏi xoáy

## 1. Tại sao không dùng test set để chọn threshold?

Vì test set phải giữ vai trò đánh giá cuối. Nếu chọn threshold trên test, ta đã tối ưu theo test và metric sẽ lạc quan. Notebook dùng OOF train prediction để chọn threshold, rồi áp lên test.

## 2. Tại sao Bike dùng lag của target mà không bị leakage?

Vì lag được tạo bằng `shift`, chỉ dùng giá trị quá khứ. Trong dự báo thực tế, demand giờ trước hoặc hôm trước có thể đã biết tại thời điểm dự báo.

## 3. Có vấn đề gì với lag khi dự báo tương lai nhiều bước?

Có. Nếu dự báo nhiều giờ tương lai liên tiếp, lag gần như `cnt_lag_1` có thể chưa biết và cần dự báo đệ quy hoặc chiến lược multi-step. Notebook phù hợp hơn với thiết lập dự báo khi dữ liệu quá khứ gần đã có.

## 4. Vì sao HGB thắng nhưng không chắc chắn production-ready?

Vì notebook dùng dữ liệu Kaggle tĩnh. Production cần kiểm tra drift, retraining, dữ liệu tương lai hợp lệ, latency, monitoring và business metric.

## 5. EDA có chứng minh nhân quả không?

Không. EDA chỉ cho thấy association/pattern. Ví dụ month-to-month churn cao không chứng minh hợp đồng tháng gây churn; có thể do nhóm khách hàng khác nhau.

## 6. Nếu business muốn giảm chi phí retention thì threshold nên thế nào?

Nếu muốn giảm liên hệ nhầm, tăng precision, nên tăng threshold. Nếu muốn bắt nhiều churn hơn, tăng recall, nên giảm threshold.

## 7. Nếu hỏi model nào dễ giải thích hơn?

Logistic Regression dễ giải thích hệ số hơn. HGB mạnh hơn về performance nhưng cần công cụ giải thích như permutation importance.

## 8. Nếu hỏi vì sao không dùng XGBoost/CatBoost/SHAP?

Notebook giữ sklearn-only để nhẹ, dễ chạy Colab, ít dependency. HGB và permutation importance đủ tốt cho phạm vi bài này.

## 9. Nếu hỏi tại sao dùng `class_weight`?

Vì churn là lớp thiểu số. `class_weight='balanced'` tăng trọng số lỗi của lớp thiểu số, giúp model chú ý hơn tới churn. Tuy nhiên notebook để random search tự chọn `None` hoặc `balanced`.

## 10. Nếu hỏi metric nào quan trọng nhất?

- Bike: RMSE quan trọng vì peak demand cần được phạt lỗi lớn; MAE giúp diễn giải sai số trung bình; R2 cho biết mức giải thích biến thiên.
- Telco: ROC-AUC/PR-AUC để đánh giá ranking, recall/F1 để đánh giá bắt churn sau threshold.

---

# Phần 6: Checklist ôn trước vấn đáp

Bạn nên nhớ chắc các con số sau:

## Bike

- Dataset: `17,379` dòng, `17` cột.
- Target: `cnt`.
- Không missing, không duplicate.
- `cnt` mean khoảng `189.46`, median `142`, max `977`.
- Top 1% demand threshold khoảng `782.22`.
- Train/test: `(13,903, 29)` và `(3,476, 29)`.
- Best model: `HistGradientBoostingRegressor`.
- Holdout: MAE `29.75`, RMSE `46.15`, R2 `0.9562`.
- Feature quan trọng nhất: `cnt_lag_1`.

## Telco

- Dataset: `7,043` khách hàng, `21` cột.
- Target: `Churn`.
- `TotalCharges` thiếu `11` dòng.
- Churn rate khoảng `26.54%`.
- Train/test: `(5,634, 19)` và `(1,409, 19)`.
- Best model: `HistGradientBoostingClassifier`.
- CV ROC-AUC khoảng `0.8479`.
- Holdout ROC-AUC khoảng `0.8469`.
- Threshold tuned khoảng `0.3274`.
- Recall tăng từ `0.5267` lên `0.7433`.
- Feature quan trọng nhất: `Contract`.

---

# Phần 7: Cách trả lời khi bị hỏi "kết luận chính là gì?"

## Kết luận Bike

Nhu cầu thuê xe theo giờ có pattern rất mạnh theo thời gian: giờ cao điểm, ngày làm việc, mùa và demand quá khứ gần. Model HGB học tốt các pattern này và đạt R2 cao trên holdout. Feature quan trọng nhất là demand giờ trước, chứng minh tính tự tương quan của bài toán.

## Kết luận Telco

Churn tập trung ở khách hàng month-to-month, tenure thấp, electronic check, fiber optic và thiếu dịch vụ hỗ trợ. Vì dữ liệu mất cân bằng, cần đánh giá bằng ROC-AUC/PR-AUC và tune threshold. Threshold tuned giúp bắt được nhiều churn hơn, phù hợp với mục tiêu giữ chân khách hàng.

## Kết luận chung

Notebook thể hiện quy trình Data Science đúng: EDA dẫn tới feature engineering, validation phù hợp với bài toán, chống leakage, model selection bằng cross-validation, đánh giá cuối bằng holdout và giải thích bằng permutation importance.
