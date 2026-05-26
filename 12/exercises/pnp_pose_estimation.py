import cv2
import numpy as np
import os

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 카메라 내부 파라미터 ──────────────────────────────────────────────────────
K = np.array([[800,   0, 320],
              [  0, 800, 240],
              [  0,   0,   1]], dtype=np.float64)

# ── 3D 객체 점: 단위 큐브(8개) + 좌표축(4개) ─────────────────────────────────
s = 1.0
obj_pts = np.array([
    [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],  # 앞면
    [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],  # 뒷면
    [ 0,  0,  0], [2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0],     # 원점+축
], dtype=np.float64)

# ── 진실 포즈 (Rodrigues 벡터 + 이동벡터) ────────────────────────────────────
rvec_true = np.array([[0.3], [-0.4], [0.1]])
tvec_true = np.array([[0.5], [-0.3], [6.0]])

# ── 2D 관측점 생성 (투영 + 노이즈) ───────────────────────────────────────────
img_pts_gt, _ = cv2.projectPoints(obj_pts, rvec_true, tvec_true, K, None)
img_pts_gt = img_pts_gt.reshape(-1, 2)

rng = np.random.default_rng(1)
img_pts_noisy = img_pts_gt + rng.normal(0, 1.5, img_pts_gt.shape)

# ── PnP (RANSAC) ──────────────────────────────────────────────────────────────
success, rvec_est, tvec_est, inliers = cv2.solvePnPRansac(
    obj_pts.reshape(-1, 1, 3).astype(np.float32),
    img_pts_noisy.reshape(-1, 1, 2).astype(np.float32),
    K, None
)

# ── 재투영 ────────────────────────────────────────────────────────────────────
img_pts_rep, _ = cv2.projectPoints(obj_pts, rvec_est, tvec_est, K, None)
img_pts_rep = img_pts_rep.reshape(-1, 2)

reproj_err = np.linalg.norm(img_pts_gt - img_pts_rep, axis=1).mean()

# ── 회전 오차 (각도) ──────────────────────────────────────────────────────────
R_true, _ = cv2.Rodrigues(rvec_true)
R_est,  _ = cv2.Rodrigues(rvec_est)
cos_angle = np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1, 1)
rot_err_deg = np.degrees(np.arccos(cos_angle))

print(f"PnP 성공: {success}  (인라이어: {len(inliers)}/{len(obj_pts)})")
print(f"재투영 오차: {reproj_err:.4f} px")
print(f"회전 오차:   {rot_err_deg:.4f} deg")
print(f"rvec 진실: {rvec_true.flatten().round(4)}")
print(f"rvec 추정: {rvec_est.flatten().round(4)}")
print(f"tvec 진실: {tvec_true.flatten().round(4)}")
print(f"tvec 추정: {tvec_est.flatten().round(4)}")


# ── 시각화 ────────────────────────────────────────────────────────────────────
def clip_pt(p, w=640, h=480):
    return tuple(np.clip(p.astype(int), [0, 0], [w - 1, h - 1]))


canvas = np.zeros((480, 640, 3), np.uint8)

# 큐브 모서리 (추정 포즈 기준 재투영)
edges = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]
for a, b in edges:
    cv2.line(canvas, clip_pt(img_pts_rep[a]), clip_pt(img_pts_rep[b]),
             (150, 150, 150), 1)

# 관측점 (노이즈 포함)
for p in img_pts_noisy[:8]:
    cv2.circle(canvas, clip_pt(p), 7, (0, 220, 220), -1)

# 재투영점 (추정 포즈)
for p in img_pts_rep[:8]:
    cv2.circle(canvas, clip_pt(p), 5, (0, 60, 255), 2)

# 좌표축
origin = clip_pt(img_pts_rep[8])
for idx, color, label in [(9, (0,0,255), "X"), (10, (0,255,0), "Y"), (11, (255,100,0), "Z")]:
    pt = clip_pt(img_pts_rep[idx])
    cv2.arrowedLine(canvas, origin, pt, color, 2, tipLength=0.2)
    cv2.putText(canvas, label, clip_pt(img_pts_rep[idx] + [6, 0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

info = [
    "PnP Pose Estimation (RANSAC)",
    f"rvec true: {rvec_true.flatten().round(3)}",
    f"rvec est : {rvec_est.flatten().round(3)}",
    f"tvec true: {tvec_true.flatten().round(3)}",
    f"tvec est : {tvec_est.flatten().round(3)}",
    f"Reproj err: {reproj_err:.4f} px   Rot err: {rot_err_deg:.4f} deg",
    "Cyan=observed(noisy)  Orange=reprojected(est. pose)",
]
for i, text in enumerate(info):
    color = (0, 255, 255) if i == 0 else (200, 200, 200)
    cv2.putText(canvas, text, (10, 305 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2_pnp_pose_estimation.png"), canvas)
print("Saved: ex2_pnp_pose_estimation.png")

cv2.imshow("PnP Pose Estimation", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
