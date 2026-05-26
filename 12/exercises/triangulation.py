import cv2
import numpy as np
import os

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 카메라 내부 파라미터 ──────────────────────────────────────────────────────
K = np.array([[800,   0, 320],
              [  0, 800, 240],
              [  0,   0,   1]], dtype=np.float64)

# ── 카메라 1: 원점 ────────────────────────────────────────────────────────────
R1 = np.eye(3)
t1 = np.zeros((3, 1))
P1 = K @ np.hstack([R1, t1])

# ── 카메라 2: x축으로 2m 이동 + y축 5도 회전 (수렴형 스테레오) ──────────────
theta = np.radians(5)
R2 = np.array([[ np.cos(theta), 0, np.sin(theta)],
               [             0, 1,             0],
               [-np.sin(theta), 0, np.cos(theta)]])
t2 = np.array([[-2.0], [0.0], [0.0]])
P2 = K @ np.hstack([R2, t2])

# ── 알려진 3D 점 (지면 진실) ──────────────────────────────────────────────────
pts_3d = np.array([
    [ 1.0,  0.5, 5.0], [-1.0,  0.5, 5.0],
    [ 1.0, -0.5, 5.0], [-1.0, -0.5, 5.0],
    [ 0.5,  1.0, 7.0], [-0.5,  1.0, 7.0],
    [ 0.5, -1.0, 7.0], [-0.5, -1.0, 7.0],
    [ 0.0,  0.0, 6.0],
], dtype=np.float64)


def project(P, X):
    Xh = np.hstack([X, np.ones((len(X), 1))])
    ph = (P @ Xh.T).T
    return ph[:, :2] / ph[:, 2:3]


x1 = project(P1, pts_3d)
x2 = project(P2, pts_3d)

# ── 노이즈 추가 ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
x1_n = x1 + rng.normal(0, 1.0, x1.shape)
x2_n = x2 + rng.normal(0, 1.0, x2.shape)


# ── DLT 삼각측량 (SVD 기반) ───────────────────────────────────────────────────
# 각 점 쌍에 대해 Ax=0 → SVD의 최소 특이값에 해당하는 벡터
def triangulate_dlt(P1, P2, pts1, pts2):
    results = []
    for p1, p2 in zip(pts1, pts2):
        A = np.array([
            p1[0] * P1[2] - P1[0],
            p1[1] * P1[2] - P1[1],
            p2[0] * P2[2] - P2[0],
            p2[1] * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        results.append(X[:3] / X[3])
    return np.array(results)


pts_dlt = triangulate_dlt(P1, P2, x1_n, x2_n)

# ── cv2.triangulatePoints 비교 ────────────────────────────────────────────────
pts_cv_h = cv2.triangulatePoints(
    P1, P2,
    x1_n.T.astype(np.float32),
    x2_n.T.astype(np.float32)
)
pts_cv = (pts_cv_h[:3] / pts_cv_h[3]).T

# ── 오차 계산 및 출력 ─────────────────────────────────────────────────────────
err_dlt = np.linalg.norm(pts_3d - pts_dlt, axis=1)
err_cv  = np.linalg.norm(pts_3d - pts_cv,  axis=1)

print("=== Triangulation 결과 ===")
for i in range(len(pts_3d)):
    print(f"  GT={pts_3d[i]}  DLT err={err_dlt[i]:.4f}  CV err={err_cv[i]:.4f}")
print(f"\nDLT   평균 3D 오차: {err_dlt.mean():.4f} m")
print(f"OpenCV 평균 3D 오차: {err_cv.mean():.4f} m")


# ── 시각화 ────────────────────────────────────────────────────────────────────
H_c, W_c = 480, 640


def draw_view(pts_obs, pts_reproj, title):
    img = np.zeros((H_c, W_c, 3), np.uint8)
    for p_o, p_r in zip(pts_obs, pts_reproj):
        pi_o = tuple(np.clip(p_o.astype(int), [0, 0], [W_c - 1, H_c - 1]))
        pi_r = tuple(np.clip(p_r.astype(int), [0, 0], [W_c - 1, H_c - 1]))
        cv2.circle(img, pi_o, 6, (0, 255, 255), -1)   # 측정점 (노이즈)
        cv2.circle(img, pi_r, 4, (0, 80, 255), 2)     # 삼각측량 재투영
        cv2.line(img, pi_o, pi_r, (80, 80, 80), 1)
    cv2.putText(img, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(img, "Cyan=noisy obs  Orange=triangulated reproj",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return img


img1 = draw_view(x1_n, project(P1, pts_dlt), "Camera 1 | DLT Triangulation")
img2 = draw_view(x2_n, project(P2, pts_dlt), "Camera 2 | DLT Triangulation")

stats = np.zeros((70, W_c * 2, 3), np.uint8)
cv2.putText(stats, f"DLT   mean 3D error : {err_dlt.mean():.4f} m",
            (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
cv2.putText(stats, f"OpenCV mean 3D error: {err_cv.mean():.4f} m",
            (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

out = np.vstack([np.hstack([img1, img2]), stats])
cv2.imwrite(os.path.join(OUTPUT_DIR, "ex1_triangulation.png"), out)
print("Saved: ex1_triangulation.png")

cv2.imshow("Triangulation (DLT / SVD)", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
