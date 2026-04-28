import os
import cv2
import numpy as np

VIDEO_PATH = "../../Data/traffic.mp4"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

# 첫 프레임에서 추적할 특징점 검출
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=200, qualityLevel=0.01,
                              minDistance=10, blockSize=7)

lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# 색상 팔레트 (각 트랙마다 고유 색상)
np.random.seed(42)
colors = np.random.randint(0, 255, (200, 3))

mask = np.zeros_like(frame1)
frame_count = 0
save_frame = None

while True:
    ret, frame2 = cap.read()
    if not ret or frame_count > 150:
        break

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    p1, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    good_old = p0[status == 1]
    good_new = p1[status == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = np.int32(new)
        c, d = np.int32(old)
        color = colors[i % len(colors)].tolist()
        mask = cv2.line(mask, (a, b), (c, d), color, 1)
        frame2 = cv2.circle(frame2, (a, b), 3, color, -1)

    output = cv2.add(frame2, mask)

    if frame_count == 100:
        save_frame = output.copy()

    gray1 = gray2.copy()
    p0 = good_new.reshape(-1, 1, 2)
    frame_count += 1

cap.release()

# 저장용 다운스케일
scale = 0.5
if save_frame is not None:
    save_small = cv2.resize(save_frame, (int(save_frame.shape[1] * scale),
                                         int(save_frame.shape[0] * scale)))
    cv2.putText(save_small, f"LK Tracking: {len(good_new)} pts (frame 100)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex3_tracking_keypoints_lk.png"), save_small)
    print(f"Saved: ex3_tracking_keypoints_lk.png  tracked={len(good_new)} pts")

cv2.imshow("Lucas-Kanade Keypoint Tracking (traffic.mp4)", save_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
