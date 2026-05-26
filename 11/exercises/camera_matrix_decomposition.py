import cv2
import numpy as np
import os

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 알려진 파라미터로 P 생성 ──────────────────────────────────────────────────
K_true = np.array([[800,   0, 320],
                   [  0, 800, 240],
                   [  0,   0,   1]], dtype=np.float64)

# 회전 행렬: y축 15도 회전
theta = np.radians(15)
R_true = np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [             0, 1,             0],
                   [-np.sin(theta), 0, np.cos(theta)]])

t_true = np.array([[0.5], [-0.3], [5.0]])

# P = K [R | t]
Rt = np.hstack([R_true, t_true])
P = K_true @ Rt
print("=== 생성된 P (3x4) ===")
print(np.round(P, 3))

# ── Camera Matrix Decomposition ───────────────────────────────────────────────
# M = P[:, :3] = K @ R → RQ 분해로 K, R 복원
M = P[:, :3]

def rq_decomp(A):
    """RQ decomposition: A = R @ Q (R upper-triangular, Q orthogonal)"""
    A_flip = np.flipud(np.fliplr(A))
    Q_flip, R_flip = np.linalg.qr(A_flip.T)
    R = np.flipud(np.fliplr(R_flip.T))
    Q = np.flipud(np.fliplr(Q_flip.T))
    # 대각 원소 양수 보장
    D = np.diag(np.sign(np.diag(R)))
    return R @ D, D @ Q

K_rec, R_rec = rq_decomp(M)

# K 정규화 (K[2,2] = 1)
K_rec = K_rec / K_rec[2, 2]

# t 복원: t = K^-1 @ p4  (p4 = P의 4번째 열)
t_rec = np.linalg.inv(K_rec) @ P[:, 3]

print("\n=== 복원된 K (Intrinsic) ===")
print(np.round(K_rec, 3))
print("\n=== 원본 K ===")
print(np.round(K_true, 3))

print("\n=== 복원된 R (Rotation) ===")
print(np.round(R_rec, 4))
print("\n=== 원본 R ===")
print(np.round(R_true, 4))

print("\n=== 복원된 t (Translation) ===")
print(np.round(t_rec.reshape(-1), 4))
print("\n=== 원본 t ===")
print(np.round(t_true.flatten(), 4))

# ── 카메라 중심 C = -R^T t ────────────────────────────────────────────────────
C = -R_rec.T @ t_rec
print(f"\n카메라 중심 C = {np.round(C, 4)}")

# ── 시각화: P, K, R, t 값 출력 이미지 ────────────────────────────────────────
canvas = np.zeros((480, 760, 3), dtype=np.uint8)
lines = [
    "Camera Matrix Decomposition",
    "",
    f"P[:,0] = [{P[0,0]:.1f}, {P[1,0]:.1f}, {P[2,0]:.4f}]",
    f"P[:,1] = [{P[0,1]:.1f}, {P[1,1]:.1f}, {P[2,1]:.4f}]",
    f"P[:,2] = [{P[0,2]:.1f}, {P[1,2]:.1f}, {P[2,2]:.4f}]",
    f"P[:,3] = [{P[0,3]:.1f}, {P[1,3]:.1f}, {P[2,3]:.4f}]",
    "",
    f"K (recovered): fx={K_rec[0,0]:.1f} fy={K_rec[1,1]:.1f}",
    f"               cx={K_rec[0,2]:.1f} cy={K_rec[1,2]:.1f}",
    f"K (true):      fx={K_true[0,0]:.1f} fy={K_true[1,1]:.1f}",
    f"               cx={K_true[0,2]:.1f} cy={K_true[1,2]:.1f}",
    "",
    f"R[0] rec:  {np.round(R_rec[0],4)}",
    f"R[0] true: {np.round(R_true[0],4)}",
    "",
    f"t rec:  {np.round(t_rec,4)}",
    f"t true: {np.round(t_true.flatten(),4)}",
    "",
    f"Camera center C: {np.round(C,4)}",
]
for i, line in enumerate(lines):
    color = (0, 255, 255) if i == 0 else (200, 200, 200)
    cv2.putText(canvas, line, (15, 28 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2_camera_matrix_decomposition.png"), canvas)
print("\nSaved: ex2_camera_matrix_decomposition.png")

cv2.imshow("Camera Matrix Decomposition (P -> K, R, t)", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
