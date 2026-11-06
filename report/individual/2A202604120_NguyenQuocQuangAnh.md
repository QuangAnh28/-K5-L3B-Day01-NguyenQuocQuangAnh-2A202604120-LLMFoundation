# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.000 | 0.000 | 0.168 | 0 |
| 2. BoW + MLP | 0.000 | 0.000 | 0.439 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.880 | 0.974 | 50.535 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.903 | 0.919 | 36.779 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.914 | 0.919 | 30.867 | 0 |
| 4a. LLM nhỏ zero-shot | 0.878 | 0.878 | 2390.262 | 0 (chưa có giá) |
| 4b. LLM lớn zero-shot | 0.900 | 0.900 | 2392.400 | 0 (chưa có giá) |
| 4c. LLM nhỏ few-shot | 0.850 | 0.850 | 4230.083 | 0 (chưa có giá) |

## 3. Dự đoán và thực tế
### 3.1. Dự đoán F1 của Logistic Regression (Phần 1)

- **Dự đoán:** F1 khoảng 0.80–0.85, Precision có thể cao hơn Recall.
- **Thực tế:** F1 = 0.000, Precision = 0.000, Recall = 0.000, mặc dù Accuracy Test đạt 88.1%.
- **Giải thích:** Kết quả thấp hơn nhiều so với dự đoán vì mô hình không phát hiện đúng tin Spam nào ở ngưỡng 0.5. Dữ liệu mất cân bằng với số lượng Ham lớn hơn Spam, kết hợp với đặc trưng Bag-of-Words, có thể khiến mô hình thiên về dự đoán Ham. Điều này cho thấy Accuracy cao chưa chắc mô hình phân loại Spam tốt.

### 3.2. Dự đoán MLP và hiện tượng Overfitting (Phần 2)

- **Dự đoán:** Validation Loss có thể tăng trở lại khi mô hình overfit. MLP có thể đạt F1 cao hơn Logistic Regression nhờ khả năng học quan hệ phi tuyến.
- **Thực tế:** Train Loss và Validation Loss đều giảm rồi ổn định quanh 0.38, chưa xuất hiện dấu hiệu overfitting rõ ràng. MLP đạt F1 Test = 0.000, bằng Logistic Regression.
- **Giải thích:** MLP có 128.129 tham số nhưng không cải thiện F1 so với Logistic Regression. Nguyên nhân có thể do dữ liệu mất cân bằng và đầu vào Bag-of-Words không biểu diễn đầy đủ ngữ nghĩa. Điều này cho thấy tăng độ phức tạp của mô hình không đảm bảo cải thiện chất lượng phân loại.

### 3.3. Dự đoán hiệu quả Few-shot của LLM (Phần 4)

- **Dự đoán:** Few-shot sẽ giúp F1 tăng nhẹ hoặc giữ nguyên vì 4 ví dụ mẫu giúp LLM hiểu cách phân loại Spam/Ham tốt hơn.
- **Thực tế:** Model nhỏ Zero-shot đạt F1 = 0.878, trong khi Few-shot đạt F1 = 0.850 trên tập SUB 40 tin nhắn, giảm 0.028.
- **Giải thích:** Việc bổ sung ví dụ không đảm bảo cải thiện kết quả vì chất lượng và cách lựa chọn ví dụ có thể ảnh hưởng đến dự đoán của LLM. Ngoài ra, tập SUB chỉ có 40 tin nhắn nên chênh lệch F1 có thể chưa ổn định. Few-shot cũng làm prompt dài hơn và tăng lượng token đầu vào.
## 4. Phân tích 50 case khó
| Mô hình | F1 SUB 40 | F1 50 case khó | Thay đổi |
|---|---:|---:|---:|
| BoW + LogReg | 0.000 | 0.000 | 0.000 |
| BoW + MLP | 0.000 | 0.000 | 0.000 |
| MiniLM + kNN | 0.974 | 0.737 | -0.237 |
| MiniLM + LogReg | 0.919 | 0.667 | -0.252 |
| MiniLM + MLP head | 0.919 | 0.654 | **-0.265** |
| LLM nhỏ Zero-shot | 0.878 | 0.923 | +0.045 |

**MiniLM + MLP head giảm F1 nhiều nhất (0.265)** vì các case khó có cách diễn đạt khác dữ liệu huấn luyện.

### 4.2. Hai trường hợp khó

**Case 1 — Spam giống tin bình thường:**
"Hey it's Jess from the gym... 3 months for the price of 1."

- Nhãn thật: Spam.
- BoW dễ bỏ sót vì tin nhắn giống cuộc trò chuyện thông thường, ít từ khóa Spam quen thuộc.

**Case 2 — Lừa đảo giả mạo ngân hàng:**
"Your bank account has been temporarily suspended..."

- Nhãn thật: Spam.
- BoW khó phát hiện vì nội dung dùng từ ngữ thông thường nhưng mang ý nghĩa lừa đảo.

### 4.3. Kết luận

BoW gặp khó khăn với Spam không có từ khóa đặc trưng. Embedding hiểu ngữ nghĩa tốt hơn nhưng vẫn giảm F1 trên dữ liệu khó. LLM nhỏ với prompt phòng thủ đạt F1 = 0.935 và Accuracy = 92%, cao nhất trên bộ 50 case này.
## 5. Thử thách cá nhân
**SEED:** 7485  
**Đề số 4:** Chạy KNN với k = 1, 5, 25; so sánh F1 và giải thích.

**Cách làm:** Sử dụng `knn_predict()` trên embedding MiniLM, thay đổi k và tính F1 trên tập Test.

| k | Accuracy | F1 |
|---|---:|---:|
| 1 | 0.970 | 0.880 |
| 5 | 0.970 | 0.880 |
| 25 | 0.961 | 0.845 |

**Giải thích:** k = 1 và k = 5 đạt F1 cao nhất (0.880). k quá nhỏ dễ bị ảnh hưởng bởi nhiễu; k quá lớn dễ bị nhóm Ham chiếm đa số làm sai lệch dự đoán. Trong thí nghiệm này, k = 5 là lựa chọn hợp lý vì ít phụ thuộc vào một hàng xóm duy nhất.

## 6. Phương án cho sản phẩm thật
Em chọn **Encoder (MiniLM) + MLP head** vì đạt:

- F1 Test = 0.914, F1 SUB = 0.919.
- p95 = 34.074 ms/tin, đáp ứng SLA dưới 50 ms.
- Tốc độ khoảng 50.395 tin/giây.

Hệ thống cần xử lý khoảng 116 tin/giây để đạt 10 triệu SMS/ngày. Vì vậy, cần tối ưu và chạy song song nhiều luồng xử lý, đồng thời kiểm thử tải thực tế.

**Đánh đổi:** Mô hình cần tài nguyên để tính embedding và thời gian huấn luyện. Tuy nhiên, mô hình không cần gọi LLM API cho từng tin nên tránh được chi phí API và độ trễ lớn của LLM.

Có thể kết hợp LLM để xử lý riêng những tin nhắn có độ tin cậy thấp, nhưng phải kiểm soát chi phí và độ trễ.


## 7. AI log và điều đã học
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| 1 | Gợi ý code vectorize, sigmoid, predict_proba | Đưa code Bag-of-Words và hàm Sigmoid | Tự viết câu trả lời, đối chiếu |
| 2 | Gợi ý code build_mlp và 5 dòng train_mlp | Viết nn.Sequential, zero_grad, backward, step | Code, chạy test, kiểm tra tham số và kết quả huấn luyện |
| 2 | Giải thích overfitting, ReLU và zero_grad | Giải thích cơ chế học và ảnh hưởng khi bỏ ReLU | Chạy 30 epochs, quan sát train/val loss và thử bỏ ReLU |
| 3 | Gợi ý code cosine_similarity, top_k_similar, knn_predict | Viết code chuẩn hóa vector, tính cosine và KNN | Sửa lỗi cú pháp, chạy test và đối chiếu kết quả |
| 4 | Hướng dẫn lấy Gemini API key và cấu hình Colab | Hướng dẫn sử dụng Secrets và OpenAI-compatible API | Tự thêm key, chạy thử Gemini rồi chuyển sang Groq |
| 4 | Nhờ viết TODO 4.1–4.4 | Đưa code gọi LLM, tạo prompt, parse nhãn, đo chi phí | Sửa lỗi import và chạy ô kiểm tra |
| 4 | Hỏi cách xử lý lỗi API 429, 400 và đổi provider | Giải thích rate limit, reasoning_effort và Groq | Điều chỉnh cấu hình, đổi sang Groq, chạy lại thí nghiệm |
| 5 | Nhờ phân tích bảng so sánh và biểu đồ Pareto | Giải thích đánh đổi giữa F1, p95 và chi phí | Kiểm tra bảng kết quả, chọn Encoder + MLP head |

### Điều em học được

Qua bài Lab, em hiểu rằng Accuracy cao chưa chắc mô hình phân loại Spam tốt, đặc biệt khi dữ liệu mất cân bằng. Em cũng nhận thấy Embedding giúp mô hình hiểu ngữ nghĩa tốt hơn Bag-of-Words. Ngoài ra, em học được cách đánh giá mô hình dựa trên F1, độ trễ và chi phí để lựa chọn phương án phù hợp khi triển khai thực tế.
