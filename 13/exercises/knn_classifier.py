"""
Exercise 2: K-Nearest Neighbor (KNN) Classifier
- cv2.ml.KNearest 사용
- k=1, 3, 7 별 결정 경계 시각화
- Voronoi 분할 개념 확인 (k=1)
- 유클리드 거리 기반 분류
"""
import cv2
import numpy as np
import os

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 합성 학습 데이터 생성 (2클래스, 2D) ───────────────────────────────────
rng = np.random.default_rng(42)

n = 60
# 클래스 1: 두 군집 (붉은)
c1_a = rng.normal(loc=[1.5, 2.0], scale=0.6, size=(n // 2, 2))
c1_b = rng.normal(loc=[3.5, 0.5], scale=0.6, size=(n // 2, 2))
X1 = np.vstack([c1_a, c1_b]).astype(np.float32)

# 클래스 2: 두 군집 (파랑)
c2_a = rng.normal(loc=[0.5, 0.5], scale=0.6, size=(n // 2, 2))
c2_b = rng.normal(loc=[2.5, 1.5], scale=0.6, size=(n // 2, 2))
X2 = np.vstack([c2_a, c2_b]).astype(np.float32)

X_train = np.vstack([X1, X2])
y_train = np.array([1]*n + [2]*n, dtype=np.float32)

# ── 2. 결정 경계 그리기 함수 ─────────────────────────────────────────────────
GRID_RES = 200
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

xs = np.linspace(x_min, x_max, GRID_RES)
ys = np.linspace(y_min, y_max, GRID_RES)
xx, yy = np.meshgrid(xs, ys)
grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

IMG_SIZE = 400

def data_to_pixel(pts, x_min, x_max, y_min, y_max, size):
    px = ((pts[:, 0] - x_min) / (x_max - x_min) * (size - 1)).astype(int)
    py = ((1 - (pts[:, 1] - y_min) / (y_max - y_min)) * (size - 1)).astype(int)
    return px, py

def draw_knn(k):
    knn = cv2.ml.KNearest_create()
    knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

    _, pred, _, _ = knn.findNearest(grid, k)
    Z = pred.ravel().reshape(GRID_RES, GRID_RES)

    # 결정 경계 이미지 생성
    bg = np.zeros((GRID_RES, GRID_RES, 3), dtype=np.uint8)
    bg[Z == 1] = [220, 180, 180]   # 연한 빨강
    bg[Z == 2] = [180, 180, 220]   # 연한 파랑

    img = cv2.resize(bg, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # 학습 데이터 점 표시
    px, py = data_to_pixel(X_train, x_min, x_max, y_min, y_max, IMG_SIZE)
    for i, (x_p, y_p) in enumerate(zip(px, py)):
        color = (0, 0, 200) if y_train[i] == 1 else (200, 0, 0)
        cv2.circle(img, (x_p, y_p), 5, color, -1)
        cv2.circle(img, (x_p, y_p), 5, (0, 0, 0), 1)

    cv2.putText(img, f"k = {k}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
    return img

# ── 3. k=1, 3, 7 비교 ─────────────────────────────────────────────────────────
panels = [draw_knn(k) for k in [1, 3, 7]]

# 범례 추가 패널
legend = np.ones((IMG_SIZE, 120, 3), dtype=np.uint8) * 245
cv2.circle(legend, (20, 60), 8, (0, 0, 200), -1)
cv2.putText(legend, "Class 1", (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.circle(legend, (20, 100), 8, (200, 0, 0), -1)
cv2.putText(legend, "Class 2", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(legend, "KNN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

result = np.hstack(panels + [legend])
out_path = os.path.join(OUTPUT_DIR, "ex2_knn_classifier.png")
cv2.imwrite(out_path, result)
print(f"저장: {out_path}")

# ── 4. 정확도 출력 ────────────────────────────────────────────────────────────
for k in [1, 3, 7]:
    knn = cv2.ml.KNearest_create()
    knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    _, pred, _, _ = knn.findNearest(X_train, k)
    acc = np.mean(pred.ravel() == y_train) * 100
    print(f"k={k} 훈련 정확도: {acc:.1f}%")
