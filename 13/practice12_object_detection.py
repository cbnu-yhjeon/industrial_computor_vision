"""
Practice 12: Object Detection
  1. HoG + SVM 보행자 검출 (people.jpg)
  2. Haar Cascade 얼굴 검출 (face.jpeg)
  3. KNN + SVM 숫자 인식 (OpenCV 합성 데이터)
  4. ArUco 마커 생성 및 검출
"""
import cv2
import numpy as np
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = "../Data"

# ── 1. HoG + SVM 보행자 검출 ─────────────────────────────────────────────────
print("=" * 50)
print("1. HoG + SVM 보행자 검출")

img_people = cv2.imread(os.path.join(DATA_DIR, "people.jpg"))
assert img_people is not None, "people.jpg 없음"

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 멀티스케일 슬라이딩 윈도우 검출
rects, weights = hog.detectMultiScale(
    img_people,
    winStride=(8, 8),
    padding=(4, 4),
    scale=1.05
)

# NMS (Non-Maximum Suppression) — 겹치는 박스 제거
def nms(rects, weights, overlap_thresh=0.65):
    if len(rects) == 0:
        return []
    rects = np.array(rects, dtype=np.float32)
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 0] + rects[:, 2]
    y2 = rects[:, 1] + rects[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = weights.ravel().argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w_inter = np.maximum(0, xx2 - xx1)
        h_inter = np.maximum(0, yy2 - yy1)
        inter = w_inter * h_inter
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        idx = np.where(iou <= overlap_thresh)[0]
        order = order[idx + 1]
    return keep

result_hog = img_people.copy()
if len(rects) > 0:
    keep = nms(rects, weights)
    for i in keep:
        x, y, rw, rh = rects[i].astype(int)
        cv2.rectangle(result_hog, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
        cv2.putText(result_hog, f"{weights.ravel()[i]:.2f}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    print(f"  검출된 보행자: {len(keep)}명 (NMS 전: {len(rects)})")
else:
    print("  보행자 검출 없음")

cv2.putText(result_hog, "HoG + SVM Pedestrian Detection", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice12_1_hog_pedestrian.png"), result_hog)
print(f"  저장: output/practice12_1_hog_pedestrian.png")

# ── 2. Haar Cascade 얼굴 검출 ─────────────────────────────────────────────────
print("\n2. Haar Cascade 얼굴 검출")

img_face = cv2.imread(os.path.join(DATA_DIR, "face.jpeg"))
assert img_face is not None, "face.jpeg 없음"
gray_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

faces = face_cascade.detectMultiScale(
    gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

result_haar = img_face.copy()
for (fx, fy, fw, fh) in faces:
    cv2.rectangle(result_haar, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
    cv2.putText(result_haar, "Face", (fx, fy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 얼굴 내부에서 눈 검출
    roi_gray = gray_face[fy:fy+fh, fx:fx+fw]
    roi_color = result_haar[fy:fy+fh, fx:fx+fw]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)

print(f"  검출된 얼굴: {len(faces)}개")
cv2.putText(result_haar, "Haar Cascade Face Detection", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice12_2_haar_face.png"), result_haar)
print(f"  저장: output/practice12_2_haar_face.png")

# ── 3. KNN & SVM 숫자 인식 ───────────────────────────────────────────────────
print("\n3. KNN & SVM 숫자 인식 (합성 데이터)")

# HOG 기반 합성 숫자 특징 생성 (간단 시뮬레이션 — 두 클래스)
# 실제 OCR은 MNIST 등 데이터셋 필요; 여기서는 간단한 2D HOG 특징 모방
rng = np.random.default_rng(7)

# 클래스별 HOG 특징 분포 (2D → 시각화 가능)
n_per_class = 100
classes = [0, 1, 2, 3, 4]
X_list, y_list = [], []
centers = np.array([[0, 0], [3, 0], [1.5, 2.5], [-1.5, 2.5], [0, 5]], dtype=np.float32)

for label, center in enumerate(centers):
    pts = rng.normal(loc=center, scale=0.7, size=(n_per_class, 2)).astype(np.float32)
    X_list.append(pts)
    y_list.extend([label] * n_per_class)

X_all = np.vstack(X_list)
y_all = np.array(y_list, dtype=np.int32)

# 학습/테스트 분할 (80/20)
idx = rng.permutation(len(y_all))
split = int(0.8 * len(idx))
X_tr, X_te = X_all[idx[:split]], X_all[idx[split:]]
y_tr, y_te = y_all[idx[:split]], y_all[idx[split:]]

# ── KNN ──────────────────────────────────────────────────────────────────────
knn = cv2.ml.KNearest_create()
knn.train(X_tr, cv2.ml.ROW_SAMPLE, y_tr.astype(np.float32))
_, knn_pred, _, _ = knn.findNearest(X_te, 5)
knn_acc = np.mean(knn_pred.ravel() == y_te) * 100
print(f"  KNN(k=5) 정확도: {knn_acc:.1f}%")

# ── SVM ──────────────────────────────────────────────────────────────────────
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(10.0)
svm.setGamma(1.0)
svm.train(X_tr, cv2.ml.ROW_SAMPLE, y_tr)
svm_pred = svm.predict(X_te)[1].ravel().astype(int)
svm_acc = np.mean(svm_pred == y_te) * 100
print(f"  SVM(RBF)  정확도: {svm_acc:.1f}%")

# 결정 경계 시각화
GRID_RES = 300
IMG_SZ = 400
x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
xs = np.linspace(x_min, x_max, GRID_RES)
ys = np.linspace(y_min, y_max, GRID_RES)
xx, yy = np.meshgrid(xs, ys)
grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

colors_map = [(180, 180, 240), (180, 240, 180), (240, 180, 180),
              (240, 240, 180), (220, 180, 240)]
point_colors = [(0, 0, 200), (0, 180, 0), (200, 0, 0),
                (180, 160, 0), (150, 0, 200)]

def make_cls_vis(pred_grid, X, y, title):
    Z = pred_grid.reshape(GRID_RES, GRID_RES)
    bg = np.zeros((GRID_RES, GRID_RES, 3), dtype=np.uint8)
    for cls_i, c in enumerate(colors_map):
        bg[Z == cls_i] = c
    img = cv2.resize(bg, (IMG_SZ, IMG_SZ), interpolation=cv2.INTER_NEAREST)

    def to_px(pts):
        px = ((pts[:, 0] - x_min) / (x_max - x_min) * (IMG_SZ - 1)).astype(int)
        py = ((1 - (pts[:, 1] - y_min) / (y_max - y_min)) * (IMG_SZ - 1)).astype(int)
        return px, py

    px, py = to_px(X)
    for i in range(len(X)):
        cv2.circle(img, (px[i], py[i]), 4, point_colors[y[i]], -1)
        cv2.circle(img, (px[i], py[i]), 4, (0, 0, 0), 1)

    cv2.putText(img, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img

_, knn_grid, _, _ = knn.findNearest(grid, 5)
knn_vis = make_cls_vis(knn_grid.ravel().astype(int), X_all, y_all,
                       f"KNN (k=5) acc={knn_acc:.0f}%")

svm_grid = svm.predict(grid)[1].ravel().astype(int)
svm_vis = make_cls_vis(svm_grid, X_all, y_all,
                       f"SVM (RBF) acc={svm_acc:.0f}%")

result_cls = np.hstack([knn_vis, svm_vis])
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice12_3_knn_svm.png"), result_cls)
print(f"  저장: output/practice12_3_knn_svm.png")

# ── 4. ArUco 마커 생성 & 검출 ────────────────────────────────────────────────
print("\n4. ArUco 마커 생성 및 검출")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# 여러 마커를 한 이미지에 배치
MARKER_SIZE = 120
MARGIN = 20
marker_ids = [0, 7, 23, 42]
n_markers = len(marker_ids)

canvas_w = (MARKER_SIZE + MARGIN) * 2 + MARGIN
canvas_h = (MARKER_SIZE + MARGIN) * 2 + MARGIN
canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 200

positions = [
    (MARGIN, MARGIN),
    (MARGIN * 2 + MARKER_SIZE, MARGIN),
    (MARGIN, MARGIN * 2 + MARKER_SIZE),
    (MARGIN * 2 + MARKER_SIZE, MARGIN * 2 + MARKER_SIZE),
]

for mid, (px_off, py_off) in zip(marker_ids, positions):
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, MARKER_SIZE)
    canvas[py_off:py_off+MARKER_SIZE, px_off:px_off+MARKER_SIZE] = marker_img

# 가우시안 노이즈 추가 (실환경 시뮬레이션)
noise = rng.normal(0, 10, canvas.shape).astype(np.int16)
canvas_noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# 저장 후 로드 (JPEG 압축 시뮬레이션)
cv2.imwrite("/tmp/aruco_test.png", canvas_noisy)
canvas_test = cv2.imread("/tmp/aruco_test.png", cv2.IMREAD_GRAYSCALE)

# 검출
corners, ids, rejected = detector.detectMarkers(canvas_test)

result_aruco = cv2.cvtColor(canvas_test, cv2.COLOR_GRAY2BGR)
if ids is not None:
    cv2.aruco.drawDetectedMarkers(result_aruco, corners, ids)
    print(f"  검출된 마커 ID: {ids.ravel().tolist()}")
    for i, corner in enumerate(corners):
        c = corner[0].mean(axis=0).astype(int)
        cv2.putText(result_aruco, f"ID:{ids[i][0]}", (c[0]-15, c[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
else:
    print("  마커 검출 실패")

cv2.putText(result_aruco, "ArUco Marker Detection (DICT_6X6_250)", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice12_4_aruco.png"), result_aruco)
print(f"  저장: output/practice12_4_aruco.png")

# ── 요약 출력 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("실습 결과 요약")
print(f"  1. HoG+SVM  → 보행자 검출 ({len(keep) if len(rects) > 0 else 0}명)")
print(f"  2. Haar     → 얼굴 검출 ({len(faces)}개)")
print(f"  3. KNN/SVM  → 숫자 분류 (KNN {knn_acc:.0f}% / SVM {svm_acc:.0f}%)")
print(f"  4. ArUco    → 마커 검출 ({len(ids) if ids is not None else 0}/{len(marker_ids)}개)")
print("=" * 50)
