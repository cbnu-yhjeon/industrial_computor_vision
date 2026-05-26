import cv2
import numpy as np
import glob
import os

DATA_DIR   = "../Data/pinhole_calib"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 참값 로드 ──────────────────────────────────────────────────────────────
K_ref    = np.load(os.path.join(DATA_DIR, "camera_mat.npy"))
dist_ref = np.load(os.path.join(DATA_DIR, "dist_coefs.npy"))
print("=== 참값 (camera_mat.npy) ===")
print(f"K_ref :\n{np.round(K_ref, 3)}")
print(f"dist  : {np.round(dist_ref, 5)}")

# ── 2. 체스보드 코너 검출 ─────────────────────────────────────────────────────
BOARD_W, BOARD_H = 10, 7
SQUARE_SIZE = 1.0   # 단위 임의 (비율만 중요)

img_paths = sorted(glob.glob(os.path.join(DATA_DIR, "img_*.png")))
print(f"\n이미지 총 {len(img_paths)}장 처리 중...")

obj_pts_1 = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
obj_pts_1[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE

obj_points, img_points = [], []
detected_views, failed_paths = [], []

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for path in img_paths:
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)
    name = os.path.basename(path)
    if ret:
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(obj_pts_1)
        img_points.append(corners_sub)
        vis = img.copy()
        cv2.drawChessboardCorners(vis, (BOARD_W, BOARD_H), corners_sub, True)
        detected_views.append(vis)
        print(f"  {name}: OK")
    else:
        failed_paths.append(name)
        print(f"  {name}: FAILED")

print(f"\n코너 검출: {len(detected_views)}/{len(img_paths)}장")

# ── 3. 카메라 캘리브레이션 ────────────────────────────────────────────────────
img_h, img_w = cv2.imread(img_paths[0]).shape[:2]
rms, K_est, dist_est, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, (img_w, img_h), None, None)

print(f"\n재투영 오차 (RMS): {rms:.4f} px")
print(f"\n추정된 K:\n{np.round(K_est, 3)}")
print(f"참값   K:\n{np.round(K_ref, 3)}")
print(f"\n추정된 dist: {np.round(dist_est, 5)}")
print(f"참값   dist: {np.round(dist_ref, 5)}")

fx_err = abs(K_est[0, 0] - K_ref[0, 0]) / K_ref[0, 0] * 100
fy_err = abs(K_est[1, 1] - K_ref[1, 1]) / K_ref[1, 1] * 100
print(f"\nfx 오차: {fx_err:.2f}%   fy 오차: {fy_err:.2f}%")

# ── 4. 재투영 오차 히스토그램 계산 ───────────────────────────────────────────
reproj_errors = []
for i, (rvec, tvec, objp, imgp) in enumerate(zip(rvecs, tvecs, obj_points, img_points)):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K_est, dist_est)
    err = np.linalg.norm(imgp.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
    reproj_errors.append(err.mean())

# ── 5. 왜곡 보정 비교 ─────────────────────────────────────────────────────────
sample_img  = cv2.imread(img_paths[0])
undist_est  = cv2.undistort(sample_img, K_est,  dist_est)
undist_ref  = cv2.undistort(sample_img, K_ref,  dist_ref)

cv2.putText(sample_img, "Original (distorted)",      (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,80,255),  2)
cv2.putText(undist_est, "Undistorted (estimated K)",  (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,200,80),  2)
cv2.putText(undist_ref, "Undistorted (reference K)",  (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200,200,0), 2)
compare = np.hstack([sample_img, undist_est, undist_ref])

# ── 6. 정보 패널 ─────────────────────────────────────────────────────────────
info = np.zeros((150, compare.shape[1], 3), np.uint8)
lines = [
    f"Images: {len(detected_views)}/{len(img_paths)} detected    RMS: {rms:.4f} px",
    f"K_est : fx={K_est[0,0]:.2f}  fy={K_est[1,1]:.2f}  cx={K_est[0,2]:.2f}  cy={K_est[1,2]:.2f}",
    f"K_ref : fx={K_ref[0,0]:.2f}  fy={K_ref[1,1]:.2f}  cx={K_ref[0,2]:.2f}  cy={K_ref[1,2]:.2f}",
    f"fx_err={fx_err:.2f}%   fy_err={fy_err:.2f}%",
    f"dist_est: k1={dist_est[0,0]:.4f}  k2={dist_est[0,1]:.4f}  p1={dist_est[0,2]:.5f}  p2={dist_est[0,3]:.5f}  k3={dist_est[0,4]:.4f}",
    f"dist_ref: k1={dist_ref[0,0]:.4f}  k2={dist_ref[0,1]:.4f}  p1={dist_ref[0,2]:.5f}  p2={dist_ref[0,3]:.5f}  k3={dist_ref[0,4]:.4f}",
]
for i, l in enumerate(lines):
    cv2.putText(info, l, (10, 24+i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (0,255,255), 1)

result = np.vstack([compare, info])
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice10_camera_calibration.png"), result)

# ── 7. 코너 검출 그리드 ───────────────────────────────────────────────────────
TH, cols_n = 120, 7
tw = int(img_w * TH / img_h)

def make_grid(imgs):
    rows = []
    for r in range((len(imgs) + cols_n - 1) // cols_n):
        row = []
        for c in range(cols_n):
            idx = r * cols_n + c
            t = cv2.resize(imgs[idx], (tw, TH)) if idx < len(imgs) \
                else np.full((TH, tw, 3), 50, np.uint8)
            row.append(t)
        rows.append(np.hstack(row))
    return np.vstack(rows)

grid = make_grid(detected_views)
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice10_detection_grid.png"), grid)

print("\nSaved: practice10_camera_calibration.png")
print("Saved: practice10_detection_grid.png")

cv2.imshow("Detected Views", grid)
cv2.imshow("Calibration Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
