# My Learning — Deep Learning 8 tuần

Thư mục bài làm cá nhân, tách riêng khỏi `phases/` để không xung đột khi đồng bộ fork với repo gốc.

Mục tiêu: hiểu 30–40% kiến thức lõi deep learning — giải thích được và chạy được (xem 6 tiêu chí trong `PROGRESS.md`).

## Cấu trúc

```
my-learning/
├── PROGRESS.md          # nhật ký tiến độ — cập nhật sau mỗi bài
├── TEMPLATE-notes.md    # mẫu ghi chú, copy cho mỗi bài
├── week-01/ … week-08/  # mỗi tuần: README liệt kê bài + ghi chú + code của bạn
└── mini-project/        # tuần 8: MLP phân loại Fashion-MNIST
```

## Quy ước đặt tên

- Ghi chú: `week-XX/<mã bài>-notes.md` — ví dụ `week-01/P1-01-notes.md`
- Code: `week-XX/<mã bài>-<tên ngắn>.py` — ví dụ `week-01/P1-01-vectors.py`
- Mã bài: `P<phase>-<số bài>`, ví dụ `P3-03` = Phase 3, bài 03 (Backpropagation)

## Mỗi bài, làm 5 bước

1. Đọc `phases/.../docs/en.md` của bài.
2. Tự gõ lại code (không copy) vào `week-XX/`, chạy, đổi 1 tham số và dự đoán trước kết quả.
3. Viết ghi chú theo `TEMPLATE-notes.md`.
4. Thêm 1 dòng vào `PROGRESS.md`.
5. Commit và push, rồi nhắn Claude: "Chấm bài P1-01".
