import os
import cv2
import numpy as np

VIDEO_PATH = "../../Data/traffic.mp4"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 5프레임 건너뛰어 움직임이 잘 보이도록
for _ in range(4):
    cap.read()
ret, frame2 = cap.read()
cap.release()

gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# ── Farneback Dense Optical Flow ─────────────────────────────────────────────
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# HSV 색상 인코딩: hue=방향, value=크기
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv = np.zeros_like(frame1)
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 1] = 255
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 화살표 그리드 시각화
h, w = frame2.shape[:2]
step = 32
arrow_canvas = frame2.copy()
for y in range(step // 2, h, step):
    for x in range(step // 2, w, step):
        fx, fy = flow[y, x]
        if abs(fx) + abs(fy) > 1.0:
            ex, ey = int(x + fx * 3), int(y + fy * 3)
            cv2.arrowedLine(arrow_canvas, (x, y), (ex, ey), (0, 255, 0), 1, tipLength=0.4)

# 저장용 다운스케일 (1920x1080 → 960x540)
scale = 0.5
def rs(im):
    return cv2.resize(im, (int(w * scale), int(h * scale)))

def lbl(im, text):
    out = im.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out

row = np.hstack([lbl(rs(frame1), "Frame 1"),
                 lbl(rs(frame2), "Frame 2 (+5)"),
                 lbl(rs(flow_bgr), "Flow HSV"),
                 lbl(rs(arrow_canvas), "Flow Arrows")])

cv2.imwrite(os.path.join(OUTPUT_DIR, "ex4_dense_optical_flow.png"), row)
print("Saved: ex4_dense_optical_flow.png")
cv2.imshow("Dense Optical Flow - Farneback (traffic.mp4)", row)
cv2.waitKey(0)
cv2.destroyAllWindows()
