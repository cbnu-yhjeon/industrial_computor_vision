import cv2
import numpy as np
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. 참 카메라 파라미터 ─────────────────────────────────────────────────────
IMG_W, IMG_H = 800, 600
K_true = np.array([[700.0,    0, IMG_W / 2],
                   [    0, 700.0, IMG_H / 2],
                   [    0,    0,          1]], dtype=np.float64)
dist_true = np.array([-0.30, 0.08, 0.0, 0.0, 0.0], dtype=np.float64)

BOARD_W, BOARD_H = 9, 6
SQUARE_MM = 25.0  # 한 칸 = 25mm

# ── 2. 3D 코너 좌표 (Z=0 평면) ────────────────────────────────────────────────
obj_pts_1 = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
obj_pts_1[:, :2] = (np.mgrid[0:BOARD_W, 0:BOARD_H]
                    .T.reshape(-1, 2) * SQUARE_MM)

# ── 3. 가상 카메라 포즈 생성 (R, t) ──────────────────────────────────────────
def make_pose(ax_deg, ay_deg, az_deg, tx, ty, tz):
    ax, ay, az = np.radians([ax_deg, ay_deg, az_deg])
    Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
    Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
    Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
    R  = Rz @ Ry @ Rx
    t  = np.array([[tx],[ty],[tz]], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t

poses = [
    make_pose(-15,  0,   0,  -100,  -50, 550),
    make_pose( 15,  0,   0,   100,  -50, 550),
    make_pose(  0, -15,  0,  -100,   50, 550),
    make_pose(  0,  15,  0,   100,   50, 550),
    make_pose(-10, -10,  5,   -80,  -80, 500),
    make_pose( 10,  10, -5,    80,   80, 500),
    make_pose(-12,  8,  0,    50,  -60, 520),
    make_pose(  8, -12, 0,   -50,   60, 520),
    make_pose(  0,   0, 10,    0,    0, 480),
    make_pose(  5,   5, -10,  30,  -30, 510),
    make_pose(-5,  -5,  8,  -30,   30, 510),
    make_pose( 12, -5,  3,   60,  -40, 530),
]

# ── 4. 체스보드 렌더링 함수 ───────────────────────────────────────────────────
def draw_board(rvec, tvec, K, dist, img_w=IMG_W, img_h=IMG_H):
    canvas = np.full((img_h, img_w, 3), 160, dtype=np.uint8)
    sq = SQUARE_MM
    bw, bh = BOARD_W + 1, BOARD_H + 1
    # 사각형 채우기
    for row in range(bh):
        for col in range(bw):
            corners_3d = np.float32([
                [col*sq,   row*sq,   0],
                [(col+1)*sq, row*sq,   0],
                [(col+1)*sq, (row+1)*sq, 0],
                [col*sq,   (row+1)*sq, 0],
            ])
            pts, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, dist)
            pts = pts.astype(np.int32).reshape(-1, 1, 2)
            color = (0, 0, 0) if (row + col) % 2 == 1 else (255, 255, 255)
            cv2.fillConvexPoly(canvas, pts, color)
    return canvas

# ── 5. 가상 이미지 생성 + 코너 검출 ──────────────────────────────────────────
obj_points, img_points, detected_views = [], [], []
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for i, (rvec, tvec) in enumerate(poses):
    view = draw_board(rvec, tvec, K_true, dist_true)
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)
    if ret:
        corners_sub = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        obj_points.append(obj_pts_1)
        img_points.append(corners_sub)
        vis = view.copy()
        cv2.drawChessboardCorners(vis, (BOARD_W, BOARD_H), corners_sub, True)
        detected_views.append(vis)
        print(f"  View {i+1:2d}: OK")
    else:
        print(f"  View {i+1:2d}: NOT detected (board partly outside frame)")

print(f"\n{len(detected_views)}/{len(poses)} views used")

# ── 6. 카메라 캘리브레이션 ────────────────────────────────────────────────────
rms, K_est, dist_est, rvecs_est, tvecs_est = cv2.calibrateCamera(
    obj_points, img_points, (IMG_W, IMG_H), None, None)

print(f"\n재투영 오차 (RMS): {rms:.4f} px")
print(f"\n추정된 K:\n{np.round(K_est, 2)}")
print(f"참값   K:\n{K_true}")
print(f"\n추정된 dist: k1={dist_est[0,0]:.4f}  k2={dist_est[0,1]:.4f}")
print(f"참값   dist: k1={dist_true[0]:.4f}  k2={dist_true[1]:.4f}")
fx_err = abs(K_est[0,0] - K_true[0,0]) / K_true[0,0] * 100
print(f"\nfx 오차: {fx_err:.2f}%")

# ── 7. 왜곡 보정 결과 ─────────────────────────────────────────────────────────
sample      = detected_views[0].copy()
undistorted = cv2.undistort(sample, K_est, dist_est)

# ── 8. 결과 저장 ─────────────────────────────────────────────────────────────
TH = 120
tw = int(IMG_W * TH / IMG_H)
cols_n = 4

def make_grid(imgs):
    rows = []
    for r in range((len(imgs) + cols_n - 1) // cols_n):
        row = []
        for c in range(cols_n):
            idx = r * cols_n + c
            t = cv2.resize(imgs[idx], (tw, TH)) if idx < len(imgs) \
                else np.full((TH, tw, 3), 60, np.uint8)
            row.append(t)
        rows.append(np.hstack(row))
    return np.vstack(rows)

grid = make_grid(detected_views)
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice10_detection_grid.png"), grid)

# 왜곡 전후 비교
cv2.putText(sample,      "Distorted",   (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,80,255), 2)
cv2.putText(undistorted, "Undistorted", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,200,80), 2)
compare = np.hstack([sample, undistorted])

info = np.zeros((130, compare.shape[1], 3), np.uint8)
lines = [
    f"RMS reprojection error: {rms:.4f} px   (used {len(detected_views)} views)",
    f"K_est : fx={K_est[0,0]:.1f}  fy={K_est[1,1]:.1f}  cx={K_est[0,2]:.1f}  cy={K_est[1,2]:.1f}",
    f"K_true: fx={K_true[0,0]:.1f}  fy={K_true[1,1]:.1f}  cx={K_true[0,2]:.1f}  cy={K_true[1,2]:.1f}",
    f"dist_est : k1={dist_est[0,0]:.4f}  k2={dist_est[0,1]:.4f}",
    f"dist_true: k1={dist_true[0]:.4f}  k2={dist_true[1]:.4f}    fx_err={fx_err:.2f}%",
]
for i, l in enumerate(lines):
    cv2.putText(info, l, (10, 24+i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

result = np.vstack([compare, info])
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice10_camera_calibration.png"), result)

print("\nSaved: practice10_detection_grid.png")
print("Saved: practice10_camera_calibration.png")

cv2.imshow("Detected Views", grid)
cv2.imshow("Calibration Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
