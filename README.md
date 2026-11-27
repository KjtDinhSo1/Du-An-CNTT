# Phân loại chủ đề và cảm xúc lời bài hát tiếng Việt

Dự án Công nghệ thông tin, Nguyễn Anh Kiệt (MSSV 52300123), Khoa Công nghệ thông tin, Trường Đại học Tôn Đức Thắng.
Giảng viên hướng dẫn: ThS. Nguyễn Quốc Bình.

Từ lời của 61.525 bài hát tiếng Việt, hệ thống gán cho mỗi bài:

- **một chủ đề**: Tình yêu; Tình yêu quê hương, đất nước; Tuổi trẻ, khát vọng và sự dấn thân; Tình cảm gia đình và cội nguồn; Nhạc thiếu nhi; hoặc Khác;
- **một cảm xúc chủ đạo**: Vui tươi, Bình yên, Buồn hoặc Mạnh mẽ;
- kèm **từ khóa làm bằng chứng**, **từ ngữ ẩn dụ / lóng** và **từ gợi cảm xúc** để người xem biết vì sao bài được xếp như vậy.

Kết quả được xuất ra CSV và một trang web `demo.html` chạy độc lập (mở bằng trình duyệt, không cần máy chủ) để tra cứu bài hát hoặc dán lời mới.

## Thư mục

```
52300123/
├── report.pdf, report.docx       báo cáo
├── README.md
└── source_code/
    ├── source.ipynb              toàn bộ quy trình: làm sạch → mô hình → đánh giá → xuất CSV + demo.html
    ├── phobert_embed.py          tính vector PhoBERT cho cả kho (lưu tiến độ, chạy tiếp được)
    ├── demo.html                 giao diện tra cứu (do notebook sinh ra, ~118 MB)
    ├── requirements.txt
    ├── tests/test_core.py        kiểm tra tự động
    └── data/
        ├── final_dataset_v3.csv                    dữ liệu gốc
        ├── final_dataset_with_topics_meaning.csv   dữ liệu notebook đọc vào và ghi kết quả ra
        ├── phobert_embeddings.npz                  vector PhoBERT của 61.525 bài
        ├── labels_gold.csv, labels_bo_sung.csv, labels_ngau_nhien.csv   nhãn chủ đề gán tay
        └── labels_cam_xuc.csv                      nhãn cảm xúc gán tay
```

## Cài đặt

```
git lfs install          # 4 file lớn (demo.html, 2 file CSV, file vector) nằm trong Git LFS
git clone https://github.com/KjtDinhSo1/Du-An-CNTT.git
cd Du-An-CNTT/source_code
pip install -r requirements.txt
```

## Chạy

1. Mở `source_code/source.ipynb` (Jupyter hoặc VS Code), chọn kernel Python đã cài thư viện, chạy toàn bộ (*Run All*). Trên máy 8 GB RAM mất khoảng 8–10 phút.
2. Kết quả: `data/final_dataset_with_topics_meaning.csv` được ghi đè và `demo.html` được tạo lại. Mở `demo.html` bằng trình duyệt.
3. Không cần chạy `phobert_embed.py`: vector đã có sẵn trong `data/phobert_embeddings.npz`. Chỉ chạy lại khi thêm bài mới hoặc sửa lời (cần cài `torch`, `transformers` và có mạng để tải mô hình `vinai/phobert-base` lần đầu; khoảng 10–12 giờ trên CPU, Ctrl+C dừng được, chạy lại là tính tiếp).

Kiểm tra tự động (vài giây, không cần dữ liệu lớn):

```
cd source_code
python -m unittest discover -s tests -v
```

## Kết quả chính

| | Độ chính xác (đo ngoài mẫu) |
|---|---|
| Chủ đề, 292 bài nhãn chuẩn | 82,2% |
| Chủ đề, 144 bài lấy ngẫu nhiên (ước lượng cho cả kho) | 91,0% ± 4,7% |
| Cảm xúc, 499 bài nhãn tay (PhoBERT) | 76,2% (đoán toàn "Buồn": 57,6%) |
| Cảm xúc, bộ đoán nhẹ dùng trên web | 74,7% |

**Lưu ý về nhãn:** toàn bộ nhãn tay dùng để huấn luyện và đánh giá được gán với sự hỗ trợ của công cụ AI (Claude), dựa trên tên bài và khoảng 300 ký tự đầu của lời, chưa được nhiều người gán độc lập. Các con số trên vì thế là ước lượng, có thể hơi lạc quan.

Lời bài hát trong dữ liệu thuộc bản quyền của tác giả; dữ liệu chỉ dùng cho mục đích học tập.
