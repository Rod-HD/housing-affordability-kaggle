# Predicting Housing Affordability in the U.S. (Kaggle)

Dự đoán `AffordabilityPercentageTrue`, tức % hộ gia đình chi ≤ 30% thu nhập cho nhà ở, theo từng ZIP code ở Mỹ.
Cuộc thi: https://www.kaggle.com/competitions/predicting-housing-affordability-in-the-u-s

Báo cáo đầy đủ mọi thí nghiệm, kể cả các phương pháp cho kết quả thấp: [docs/BAO_CAO_THI_NGHIEM.md](docs/BAO_CAO_THI_NGHIEM.md)

## Bài toán
- Train gồm tháng 5 và 6/2026 của 6.830 ZIP. Test là tháng 7/2026, gồm 6.830 ZIP đã có trong train (**seen**) và 1.704 ZIP chưa từng xuất hiện (**new**, được chọn ngẫu nhiên).
- Điểm = 0,2 × RMSE(seen) + 0,8 × RMSE(new). Trọng số được suy ra từ các lần nộp; đề bài chỉ ghi "ưu tiên ZIP mới".

## Kết quả public leaderboard
| Phiên bản | Ý tưởng chính | Public |
|---|---|---|
| sub01 | Seen: giữ nhãn tháng 6; New: ensemble GBM + target encoding + KNN theo mã ZIP | 9,958 |
| sub02 | + tọa độ ZIP (GeoNames), KNN theo khoảng cách thật | 9,577 |
| sub06 | + láng giềng "gần và giống nhau", ensemble tối ưu RMSE (NNLS) | 9,435 |
| **sub07** | **+ giá nhà Zillow ZHVI, phân phối thu nhập IRS, CDC SDOH; seen chỉnh theo thay đổi giá nhà** | **4,473** |
| sub08 | + chi phí theo năm mua, loss Huber, trọng số √SampleSize (CV ZIP mới 5,070 → 5,016) | _chưa nộp_ |

## Pipeline (sub07)
1. **Seen ZIP**: `y_jul = y_jun + β · Δlog(ZHVI) · y_jun(100 − y_jun)/100`, với β = −2,43 khớp trên bước tháng 5→6 (không dùng intercept).
2. **New ZIP**:
   - Khử drift các cột hộ gia đình tháng 7 về thang tháng 6.
   - Khoảng 150 đặc trưng:
     - Tỷ lệ giá nhà/thu nhập theo từng phân khúc Zillow.
     - Chi phí trả góp ước tính.
     - Tỷ lệ hộ có thu nhập vượt ngưỡng chi phí (từ IRS).
     - Target encoding vùng và KNN địa lý (tính out-of-fold).
     - Hồi quy cục bộ GWR.
   - Mô hình: LightGBM ×2 + CatBoost, trọng số NNLS, 3 seed.
3. **Kiểm định "July-proxy"**: lấy ZIP đã biết làm ZIP mới giả, dùng đặc trưng tháng 7 của chúng và so với nhãn tháng 6. KFold 5 phần theo ZIP. Kết quả này dự đoán đúng điểm leaderboard trong khoảng ±0,15.

## Chạy lại
```powershell
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
# đặt train.csv, test.csv, sample_submission.csv của cuộc thi vào thư mục data/
.venv\Scripts\python.exe src\download_external.py     # Zillow, IRS, CDC, GeoNames -> external/
copy .env.example .env                                # (tuỳ chọn) điền CENSUS_API_KEY
.venv\Scripts\python.exe src\download_census.py       # (tuỳ chọn) ACS B25106 theo ZCTA
cd src
..\.venv\Scripts\python.exe exp_zillow.py ZI lgb,cat,lgb_slow   # kiểm định (đặt biến môi trường IRS=1)
..\.venv\Scripts\python.exe ens_rmse.py ..\outputs\rmse_oof_ZI_s0.csv
..\.venv\Scripts\python.exe make_submission_v3.py               # -> submissions/sub07_zillow_irs_v3.csv
```

## Cấu trúc thư mục
| Đường dẫn | Nội dung |
|---|---|
| `src/common.py` | Nạp dữ liệu, feature engineering, đặc trưng láng giềng, Zillow, IRS, GWR |
| `src/pipeline.py` | Dựng ma trận đặc trưng, danh sách mô hình, khử drift |
| `src/explore*.py`, `ana_oof.py` | Phân tích dữ liệu, adversarial validation, phân tích phần dư |
| `src/exp_known.py`, `exp_seen_z.py` | Thí nghiệm cho seen ZIP |
| `src/exp_newzip.py`, `exp_geo.py`, `exp_julyproxy.py`, `exp_rmse.py`, `exp_zillow.py` | Thí nghiệm cho new ZIP |
| `src/tune.py`, `tune_v2.py`, `ens_rmse.py` | Optuna, tối ưu trọng số ensemble theo RMSE |
| `src/structural_model.py` | Mô hình cấu trúc tham số (thu nhập log-normal), kết quả âm |
| `src/make_submission*.py` | Tạo file nộp |
| `outputs/figures/` | Hình cho báo cáo |

## Dữ liệu ngoài (công khai, không chứa nhãn test)
Zillow Research (ZHVI/ZORI), IRS SOI ZIP Code Data, CDC PLACES SDOH (ZCTA), GeoNames.
Dữ liệu của cuộc thi **không** được đưa lên repo, theo luật cuộc thi.
