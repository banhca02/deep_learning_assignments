# CO5085 – Deep Learning Assignments (E1–E3)

Repository này chứa ba notebook thực nghiệm phân loại chữ số viết tay **MNIST** cho môn **CO5085 – Deep Learning and Its Applications in Computer Vision**. Các bài sử dụng cùng cách chia dữ liệu để so sánh các kiến trúc học sâu.

## 1. Nội dung

| Bài | Notebook | Mô hình / thí nghiệm |
|---|---|---|
| E1 | `co5085-e1-mnist-classification.ipynb` | Softmax Classifier, MLP, CNN |
| E2 | `co5085-e2-mnist-classification.ipynb` | Transformer: Patch / Row / CNN Embedding với MSA tự viết; Patch Embedding với PyTorch MSA |
| E3 | `co5085-e3-mnist-classification.ipynb` | LSTM, GRU với ảnh biểu diễn thành chuỗi 28 hàng |

> Nếu repository đặt notebook trong các thư mục `E1/`, `E2/`, `E3/`, hãy mở notebook tương ứng trong từng thư mục. Tên tệp trong bảng là tên dự kiến sau khi bỏ hậu tố phiên bản như `(1)` hoặc `(2)`.

## 2. Cấu hình thực nghiệm

- **Dataset:** MNIST, ảnh xám 28 × 28, 10 lớp (0–9).
- **Chia dữ liệu:** 50.000 train, 10.000 validation, 10.000 test.
- **Tiền xử lý:** `transforms.ToTensor()`; dữ liệu được tải tự động bằng `torchvision.datasets.MNIST`.
- **Seed:** `42` cho Python, NumPy, PyTorch và thao tác chia tập dữ liệu.
- **Batch size:** `128`.
- **Epochs:** `10`.
- **Optimizer:** Adam, learning rate `0.001`.
- **Loss:** `CrossEntropyLoss` (đầu ra của mô hình là logits).
- **Môi trường đã dùng:** Kaggle Notebook, PyTorch, GPU NVIDIA Tesla T4.

## 3. Cách chạy trên Kaggle (khuyến nghị)

1. Đăng nhập [Kaggle](https://www.kaggle.com/) và tạo một Notebook mới (hoặc tải notebook `.ipynb` trong repository lên Kaggle).
2. Mở phần **Settings / Session options** và chọn **Accelerator: GPU** (nếu có). Cho phép truy cập Internet khi cần tải MNIST lần đầu.
3. Mở lần lượt notebook E1, E2 hoặc E3. Mỗi notebook có thể chạy **độc lập**, không cần chạy bài trước.
4. Chọn **Run All** và chạy các cell theo thứ tự từ đầu đến cuối; không bỏ qua cell tạo dữ liệu, định nghĩa mô hình và hàm huấn luyện.
5. Xem kết quả sau khi huấn luyện: accuracy trên tập test, tổng số tham số, thời gian huấn luyện, biểu đồ validation accuracy, confusion matrix và ví dụ dự đoán sai.

Notebook tải MNIST vào `/kaggle/working/data` và lưu trọng số mô hình tốt nhất (theo validation loss) tại `/kaggle/working/<model_name>_best.pth`. Các tệp này nằm trong thư mục làm việc của phiên Kaggle, không tự động được commit lên GitHub.

## 4. Cách chạy trên máy cá nhân (tùy chọn)

Yêu cầu Python 3.10 trở lên, Jupyter Notebook/JupyterLab và các thư viện sau:

```bash
python -m pip install torch torchvision numpy matplotlib scikit-learn tqdm notebook
```

Tải repository:

```bash
git clone https://github.com/banhca02/deep_learning_assignments.git
cd deep_learning_assignments
jupyter notebook
```

Mở một notebook và chọn **Run All**.

**Quan trọng:** Trong cả ba notebook, đường dẫn tải dữ liệu và lưu checkpoint đang được viết cho Kaggle (`/kaggle/working/...`). Trước khi chạy trên máy cá nhân, hãy sửa:

- `root="/kaggle/working/data"` thành `root="./data"` ở cả tập train và test.
- Đường dẫn lưu trong `torch.save(...)` từ `/kaggle/working/...` thành đường dẫn hợp lệ trên máy, chẳng hạn `./<model_name>_best.pth`.
- Giữ lựa chọn `device = "cuda" if torch.cuda.is_available() else "cpu"` để chạy trên GPU nếu khả dụng, hoặc CPU nếu không có GPU (thời gian có thể lâu hơn).

Không cần tải MNIST thủ công nếu máy có Internet và `download=True` vẫn được bật.

## 5. Kết quả và khả năng tái lập

Mỗi notebook tự triển khai vòng lặp huấn luyện, theo dõi train/validation loss và accuracy, lưu checkpoint có validation loss thấp nhất, sau đó đánh giá trên tập test. Hãy sử dụng cùng phiên bản môi trường, GPU và seed nếu muốn so sánh thời gian huấn luyện; thời gian và accuracy có thể chênh lệch giữa các lần chạy hoặc các thiết bị khác nhau.

Các notebook sử dụng **chung dữ liệu MNIST** nhưng chạy độc lập. E3 so sánh thêm với kết quả MLP/CNN từ E1 trong phần báo cáo, không yêu cầu tải checkpoint E1 để huấn luyện E3.

## 6. Mã nguồn và báo cáo

- Repository: https://github.com/banhca02/deep_learning_assignments.git
- Báo cáo tổng hợp: xem tệp PDF của dự án tại reports/exercises.pdf.

