"""
Exercise 3: SVM (Support Vector Machine) Classifier
- cv2.ml.SVM 사용
- Linear, Polynomial, RBF 커널 비교
- 마진 & 서포트 벡터 시각화
- 소프트 마진 (C 파라미터 효과)
"""
import cv2
import numpy as np
import os

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 400
GRID_RES = 200

# ── 1. 선형 분리 가능 데이터 ─────────────────────────────────────────────────
rng = np.random.default_rng(0)

n = 40
X1_lin = rng.normal(loc=[1.5, 2.5], scale=0.5, size=(n, 2)).astype(np.float32)
X2_lin = rng.normal(loc=[3.0, 1.0], scale=0.5, size=(n, 2)).astype(np.float32)
X_lin = np.vstack([X1_lin, X2_lin])
y_lin = np.array([1]*n + [-1]*n, dtype=np.int32)

# ── 2. 비선형 데이터 (동심원) ─────────────────────────────────────────────────
angles = rng.uniform(0, 2*np.pi, n)
r1 = rng.uniform(0, 1.0, n)
r2 = rng.uniform(1.5, 2.5, n)

X1_nl = np.column_stack([r1 * np.cos(angles) + 2.5, r1 * np.sin(angles) + 2.5]).astype(np.float32)
X2_nl = np.column_stack([r2 * np.cos(angles) + 2.5, r2 * np.sin(angles) + 2.5]).astype(np.float32)
X_nl = np.vstack([X1_nl, X2_nl])
y_nl = np.array([1]*n + [-1]*n, dtype=np.int32)

def data_to_pixel(pts, x_min, x_max, y_min, y_max, size):
    px = ((pts[:, 0] - x_min) / (x_max - x_min) * (size - 1)).astype(int)
    py = ((1 - (pts[:, 1] - y_min) / (y_max - y_min)) * (size - 1)).astype(int)
    return px, py

def draw_svm(X, y, kernel_type, kernel_name, C=1.0, degree=3, gamma=0.5):
    svm = cv2.ml.SVM_create()
    svm.setKernel(kernel_type)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(C)
    if kernel_type == cv2.ml.SVM_POLY:
        svm.setDegree(degree)
    if kernel_type == cv2.ml.SVM_RBF:
        svm.setGamma(gamma)

    svm.train(X, cv2.ml.ROW_SAMPLE, y)

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5

    xs = np.linspace(x_min, x_max, GRID_RES)
    ys = np.linspace(y_min, y_max, GRID_RES)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    pred = svm.predict(grid)[1].ravel().reshape(GRID_RES, GRID_RES)

    bg = np.zeros((GRID_RES, GRID_RES, 3), dtype=np.uint8)
    bg[pred == 1]  = [220, 180, 180]
    bg[pred == -1] = [180, 180, 220]
    img = cv2.resize(bg, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # 데이터 포인트 그리기
    px, py = data_to_pixel(X, x_min, x_max, y_min, y_max, IMG_SIZE)
    for i, (x_p, y_p) in enumerate(zip(px, py)):
        color = (0, 0, 200) if y[i] == 1 else (200, 0, 0)
        cv2.circle(img, (x_p, y_p), 5, color, -1)
        cv2.circle(img, (x_p, y_p), 5, (0, 0, 0), 1)

    # 서포트 벡터 강조
    sv = svm.getSupportVectors()
    if sv is not None:
        spx, spy = data_to_pixel(sv, x_min, x_max, y_min, y_max, IMG_SIZE)
        for x_p, y_p in zip(spx, spy):
            cv2.circle(img, (x_p, y_p), 9, (0, 255, 0), 2)

    cv2.putText(img, kernel_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
    n_sv = svm.getSupportVectors().shape[0] if svm.getSupportVectors() is not None else 0
    cv2.putText(img, f"SVs: {n_sv}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img

# ── 3. 선형 데이터: 세 커널 비교 ─────────────────────────────────────────────
lin_panels = [
    draw_svm(X_lin, y_lin, cv2.ml.SVM_LINEAR, "Linear",   C=1.0),
    draw_svm(X_lin, y_lin, cv2.ml.SVM_POLY,   "Poly(d=3)", C=1.0, degree=3),
    draw_svm(X_lin, y_lin, cv2.ml.SVM_RBF,    "RBF",       C=1.0, gamma=1.0),
]

# ── 4. 비선형 데이터: 세 커널 비교 ───────────────────────────────────────────
nl_panels = [
    draw_svm(X_nl, y_nl, cv2.ml.SVM_LINEAR, "Linear",    C=1.0),
    draw_svm(X_nl, y_nl, cv2.ml.SVM_POLY,   "Poly(d=3)", C=10.0, degree=3),
    draw_svm(X_nl, y_nl, cv2.ml.SVM_RBF,    "RBF",       C=10.0, gamma=0.5),
]

# ── 5. 행 레이블 추가 ─────────────────────────────────────────────────────────
def row_label(text, h):
    label = np.ones((h, 120, 3), dtype=np.uint8) * 245
    cv2.putText(label, text, (5, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return label

row1 = np.hstack([row_label("Linear", IMG_SIZE)] + lin_panels)
row2 = np.hstack([row_label("Non-linear", IMG_SIZE)] + nl_panels)

# 범례
legend = np.ones((50, row1.shape[1], 3), dtype=np.uint8) * 245
cv2.circle(legend, (50, 25), 6, (0, 0, 200), -1)
cv2.putText(legend, "Class +1", (65, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.circle(legend, (200, 25), 6, (200, 0, 0), -1)
cv2.putText(legend, "Class -1", (215, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.circle(legend, (380, 25), 9, (0, 255, 0), 2)
cv2.putText(legend, "Support Vector", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

result = np.vstack([row1, row2, legend])
out_path = os.path.join(OUTPUT_DIR, "ex3_svm_classifier.png")
cv2.imwrite(out_path, result)
print(f"저장: {out_path}")
print(f"결과 이미지 크기: {result.shape}")
