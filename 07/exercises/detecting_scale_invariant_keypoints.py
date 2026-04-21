import cv2
import numpy as np
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img  = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

# ── 스케일별 비교: 원본 / 1/2 / 1/4 ──────────────────────────────────────────
scales = [1.0, 0.5, 0.25]
results = []

for s in scales:
    h, w = img.shape[:2]
    resized_gray = cv2.resize(gray, (int(w * s), int(h * s)))
    resized_img  = cv2.resize(img,  (int(w * s), int(h * s)))

    kp, _ = sift.detectAndCompute(resized_gray, None)

    canvas = cv2.drawKeypoints(
        resized_img, kp, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    # 비교를 위해 원본 크기로 복원
    canvas = cv2.resize(canvas, (w, h))
    results.append(canvas)
    print(f"[SIFT scale={s:.2f}] 키포인트: {len(kp)}개")

combined = np.hstack(results)
cv2.imwrite(os.path.join(OUTPUT_DIR, "sift_scale_invariant.png"), combined)

cv2.imshow("SIFT Scale Invariant: 1.0 / 0.5 / 0.25", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
