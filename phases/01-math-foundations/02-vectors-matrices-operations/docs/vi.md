# Vector, Ma trận và các Phép toán (Vectors, Matrices & Operations)

> Mọi mạng nơ-ron chỉ là phép nhân ma trận cộng thêm vài bước.

> Bản dịch tiếng Việt của `docs/en.md` để tự học. Thuật ngữ chuyên ngành giữ tiếng Anh trong ngoặc khi cần; code giữ nguyên.

**Loại bài:** Build
**Ngôn ngữ:** Python, Julia
**Yêu cầu trước:** Phase 1, Bài 01 (Linear Algebra Intuition)
**Thời gian:** ~60 phút

## Mục tiêu học tập

- Xây dựng lớp Matrix có phép toán từng phần tử (element-wise), nhân ma trận, chuyển vị (transpose), định thức (determinant) và nghịch đảo (inverse)
- Phân biệt nhân từng phần tử với nhân ma trận và giải thích khi nào dùng cái nào
- Cài đặt một lớp dense của mạng nơ-ron (`relu(W @ x + b)`) chỉ bằng lớp Matrix tự viết
- Giải thích quy tắc broadcasting và cách cộng bias hoạt động trong các framework mạng nơ-ron

## Vấn đề

Bạn muốn xây một mạng nơ-ron. Bạn đọc code và thấy dòng này:

```text
output = activation(weights @ input + bias)
```

Dấu `@` đó là phép nhân ma trận. `weights` là một ma trận. `input` là một vector. Nếu bạn không biết các phép toán đó làm gì, dòng này là phép màu. Nếu biết, đó là toàn bộ bước forward của một lớp gói gọn trong ba phép toán.

Mỗi ảnh mà mô hình xử lý là một ma trận giá trị pixel. Mỗi word embedding là một vector. Mỗi lớp của mọi mạng nơ-ron là một phép biến đổi ma trận. Bạn không thể xây hệ thống AI nếu không thành thạo phép toán ma trận, cũng như không thể viết code mà không hiểu biến.

Bài này xây dựng sự thành thạo đó từ đầu.

## Khái niệm

### Vector: danh sách số có thứ tự

Vector là một danh sách số có hướng và độ lớn. Trong AI, vector biểu diễn điểm dữ liệu, đặc trưng hoặc tham số.

```text
v = [3, 4]        -- vector 2D
w = [1, 0, -2]    -- vector 3D
```

Vector 2D `[3, 4]` chỉ tới tọa độ (3, 4) trên mặt phẳng. Độ dài (magnitude) của nó là 5 (tam giác 3-4-5).

### Ma trận: lưới số

Ma trận là một lưới 2 chiều gồm hàng và cột. Ma trận m x n có m hàng và n cột.

```text
A = | 1  2  3 |     -- ma trận 2x3 (2 hàng, 3 cột)
    | 4  5  6 |
```

Trong mạng nơ-ron, ma trận trọng số biến vector input thành vector output. Một lớp có 784 input và 128 output dùng ma trận trọng số 128x784.

### Vì sao shape quan trọng

Phép nhân ma trận có một quy tắc chặt: `(m x n) @ (n x p) = (m x p)`. Hai chiều bên trong phải khớp nhau.

```text
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Chiều bên trong: 784 = 784  -- hợp lệ
```

Nếu bạn gặp lỗi shape mismatch trong PyTorch, đây là lý do.

### Bản đồ các phép toán

| Phép toán | Làm gì | Dùng trong mạng nơ-ron |
|-----------|-------------|-------------------|
| Cộng (Addition) | Kết hợp từng phần tử | Cộng bias vào output |
| Nhân vô hướng (Scalar multiply) | Co giãn mọi phần tử | Learning rate * gradient |
| Nhân ma trận (Matrix multiply) | Biến đổi vector | Bước forward của một lớp |
| Chuyển vị (Transpose) | Đổi hàng thành cột | Backpropagation |
| Định thức (Determinant) | Một con số tóm tắt | Kiểm tra tính khả nghịch |
| Nghịch đảo (Inverse) | Hoàn tác một phép biến đổi | Giải hệ phương trình tuyến tính |
| Đơn vị (Identity) | Ma trận "không làm gì" | Khởi tạo, residual connection |

### Nhân từng phần tử và nhân ma trận

Sự phân biệt này làm người mới vấp liên tục.

Nhân từng phần tử (element-wise): nhân các vị trí tương ứng. Hai ma trận phải cùng shape.

```text
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Nhân ma trận: tích vô hướng giữa hàng và cột. Hai chiều bên trong phải khớp.

```text
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Phép toán khác nhau, kết quả khác nhau, quy tắc khác nhau.

### Broadcasting

Khi bạn cộng một vector bias vào ma trận output, shape không khớp. Broadcasting kéo giãn mảng nhỏ hơn cho vừa.

```text
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting kéo giãn vector theo các hàng:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Mọi framework hiện đại đều làm việc này tự động. Hiểu nó giúp bạn không bối rối khi shape trông có vẻ sai nhưng code vẫn chạy.

```figure
vector-projection
```

## Tự xây dựng (Build It)

### Bước 1: Lớp Vector

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Bước 2: Lớp Matrix với các phép toán lõi

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Bước 3: Chạy thử

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Bước 4: Nối với mạng nơ-ron

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Đây là một lớp dense: `output = relu(W @ x + b)`. Mọi lớp dense trong mọi mạng nơ-ron đều làm đúng việc này.

## Dùng thư viện (Use It)

NumPy làm mọi thứ ở trên với ít dòng hơn và nhanh hơn hàng trăm lần.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Toán tử `@` trong Python gọi `__matmul__`. NumPy cài đặt nó bằng các routine BLAS tối ưu viết bằng C và Fortran. Cùng một phép toán, nhanh hơn 100 lần.

Broadcasting trong NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy tự động broadcast vector bias 1D qua cả hai hàng. Đây là cách cộng bias hoạt động trong mọi framework mạng nơ-ron.

## Sản phẩm (Ship It)

Bài này tạo ra một prompt để dạy phép toán ma trận qua trực giác hình học. Xem `outputs/prompt-matrix-operations.md`.

Lớp Matrix xây ở đây là nền móng cho mini framework mạng nơ-ron chúng ta xây ở Phase 3, Bài 10.

## Bài tập

1. **Kiểm chứng nghịch đảo.** Nhân `A @ A.inverse_2x2()` và xác nhận bạn nhận được ma trận đơn vị. Thử với ba ma trận 2x2 khác nhau. Chuyện gì xảy ra khi định thức bằng 0?

2. **Cài đặt nghịch đảo 3x3.** Mở rộng lớp Matrix để tính nghịch đảo ma trận 3x3 bằng phương pháp ma trận phụ hợp (adjugate). Kiểm tra so với `np.linalg.inv` của NumPy.

3. **Xây mạng hai lớp.** Chỉ dùng lớp Matrix của bạn (không NumPy), tạo một mạng nơ-ron hai lớp: input (3) -> hidden (4) -> output (2). Khởi tạo trọng số ngẫu nhiên, chạy một bước forward và kiểm chứng mọi shape đều đúng.

## Thuật ngữ chính

| Thuật ngữ | Người ta hay nói | Thực chất nghĩa là |
|------|----------------|----------------------|
| Vector | "Một mũi tên" | Một danh sách số có thứ tự. Trong AI: một điểm trong không gian nhiều chiều. |
| Matrix (ma trận) | "Một bảng số" | Một phép biến đổi tuyến tính, ánh xạ vector từ không gian này sang không gian khác. |
| Matrix multiply (nhân ma trận) | "Cứ nhân các số với nhau" | Tích vô hướng giữa từng hàng của ma trận thứ nhất và từng cột của ma trận thứ hai. Thứ tự quan trọng. |
| Transpose (chuyển vị) | "Lật nó lại" | Đổi hàng thành cột. Biến ma trận m x n thành n x m. Rất quan trọng trong backpropagation. |
| Determinant (định thức) | "Một con số nào đó từ ma trận" | Đo ma trận co giãn diện tích (2D) hoặc thể tích (3D) bao nhiêu. Bằng 0 nghĩa là phép biến đổi ép dẹt mất một chiều. |
| Inverse (nghịch đảo) | "Hoàn tác ma trận" | Ma trận đảo ngược phép biến đổi. Chỉ tồn tại khi định thức khác 0. |
| Identity matrix (ma trận đơn vị) | "Ma trận nhàm chán" | Tương đương với nhân cho 1. Dùng trong residual connection (ResNet). |
| Broadcasting | "Phép màu sửa shape" | Kéo giãn mảng nhỏ cho khớp mảng lớn bằng cách lặp lại theo các chiều bị thiếu. |
| Element-wise (từng phần tử) | "Phép nhân bình thường" | Nhân các vị trí tương ứng. Hai mảng phải cùng shape (hoặc broadcast được). |

## Đọc thêm

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - trực giác hình ảnh cho mọi phép toán trong bài
- [Tài liệu NumPy về broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - các quy tắc chính xác NumPy tuân theo
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - tài liệu tham khảo ngắn gọn về đại số tuyến tính cho ML
