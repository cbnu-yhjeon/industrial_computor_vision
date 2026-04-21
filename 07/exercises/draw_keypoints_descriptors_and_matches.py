import cv2
import numpy as np
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img  = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

# ── Keypoints 시각화 ───────────────────────────────────────────────────────────
canvas_kp = cv2.drawKeypoints(
    img, kp, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
cv2.imwrite(os.path.join(OUTPUT_DIR, "sift_keypoints.png"), canvas_kp)
print(f"[SIFT] 키포인트: {len(kp)}개, 디스크립터 shape: {des.shape}")

# ── 두 영상 간 매칭 (좌우 반전 영상과 매칭) ─────────────────────────────────
img2  = cv2.flip(img, 1)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des, des2, k=2)

good = [m for m, n in matches if m.distance < 0.75 * n.distance]

canvas_match = cv2.drawMatchesKnn(
    img, kp, img2, kp2,
    [[m] for m in good[:50]], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv2.imwrite(os.path.join(OUTPUT_DIR, "sift_matches.png"), canvas_match)
print(f"[SIFT] 매칭 수: {len(good)}개 (Lowe ratio test 적용)")

cv2.imshow("SIFT Keypoints", canvas_kp)
cv2.imshow("SIFT Matches (original vs flipped)", canvas_match)
cv2.waitKey(0)
cv2.destroyAllWindows()
