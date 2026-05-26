import cv2
import numpy as np
import os

OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 카메라 내부 파라미터 (Intrinsic K) ────────────────────────────────────────
fx, fy = 800, 800
cx, cy = 320, 240
K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)

# ── 카메라 외부 파라미터 (Extrinsic R, t) ─────────────────────────────────────
# 카메라: z축 +5 위치에서 원점을 바라봄
rvec = np.array([[0.3], [-0.4], [0.1]])          # Rodrigues 회전벡터
tvec = np.array([[0.0], [0.0], [5.0]])            # 이동벡터 (m)

# ── 3D 단위 큐브 꼭짓점 ──────────────────────────────────────────────────────
s = 1.0
cube_3d = np.array([
    [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],  # 뒷면
    [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],  # 앞면
], dtype=np.float64)

# 좌표축 (원점에서 각 축 방향 1.5 단위)
axis_3d = np.array([[0,0,0],[1.5,0,0],[0,1.5,0],[0,0,1.5]], dtype=np.float64)

# ── 3D → 2D 투영 ─────────────────────────────────────────────────────────────
pts_cube, _ = cv2.projectPoints(cube_3d,  rvec, tvec, K, None)
pts_axis, _ = cv2.projectPoints(axis_3d,  rvec, tvec, K, None)

pts_cube = pts_cube.reshape(-1, 2).astype(int)
pts_axis = pts_axis.reshape(-1, 2).astype(int)

# ── 시각화 ────────────────────────────────────────────────────────────────────
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

edges = [(0,1),(1,2),(2,3),(3,0),           # 뒷면
         (4,5),(5,6),(6,7),(7,4),           # 앞면
         (0,4),(1,5),(2,6),(3,7)]           # 옆면

for i, j in edges:
    cv2.line(canvas, tuple(pts_cube[i]), tuple(pts_cube[j]), (200, 200, 200), 1)

for pt in pts_cube:
    cv2.circle(canvas, tuple(pt), 4, (0, 255, 255), -1)

O = tuple(pts_axis[0])
cv2.arrowedLine(canvas, O, tuple(pts_axis[1]), (0, 0, 255),   2, tipLength=0.2)  # X
cv2.arrowedLine(canvas, O, tuple(pts_axis[2]), (0, 255, 0),   2, tipLength=0.2)  # Y
cv2.arrowedLine(canvas, O, tuple(pts_axis[3]), (255, 100, 0), 2, tipLength=0.2)  # Z
cv2.putText(canvas, "X", tuple(pts_axis[1] + [5, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
cv2.putText(canvas, "Y", tuple(pts_axis[2] + [5, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
cv2.putText(canvas, "Z", tuple(pts_axis[3] + [5, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,0), 1)

info = [f"fx={fx} fy={fy}", f"cx={cx} cy={cy}", "rvec=[0.3,-0.4,0.1]", "tvec=[0,0,5]"]
for i, t in enumerate(info):
    cv2.putText(canvas, t, (10, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex1_projecting_3d_points.png"), canvas)
print("Saved: ex1_projecting_3d_points.png")
print(f"Intrinsic K:\n{K}")

cv2.imshow("Projecting 3D Points (Pinhole Camera)", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
