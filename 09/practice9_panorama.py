import os
import cv2
import numpy as np

IMAGE_PATH = "../Data/people.jpg"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

src = cv2.imread(IMAGE_PATH)
h, w = src.shape[:2]

# people.jpg를 3개의 overlapping 조각으로 분할
step = w // 3
overlap = w // 5
imgs = [
    src[:, 0                   : step + overlap],
    src[:, step - overlap      : 2 * step + overlap],
    src[:, 2 * step - overlap  : w],
]

# ── SIFT + RANSAC Homography 기반 스티칭 ──────────────────────────────────────
def stitch_pair(img_left, img_right):
    sift = cv2.SIFT_create()
    g_l = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    g_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g_l, None)
    kp2, des2 = sift.detectAndCompute(g_r, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    print(f"  matches={len(good)}", end="")

    if len(good) < 4:
        return np.hstack([img_left, img_right])

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(f"  inliers={int(mask.sum())}")

    hl, wl = img_left.shape[:2]
    hr, wr = img_right.shape[:2]
    H_inv = np.linalg.inv(H)

    warped_left = cv2.warpPerspective(img_left, H_inv, (wl + wr, hl))
    canvas = warped_left.copy()
    canvas[:hr, :wr] = img_right

    # 검은 테두리 제거
    gray_c = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray_c, 1, 255, cv2.THRESH_BINARY)
    xs = np.where(m.any(axis=0))[0]
    ys = np.where(m.any(axis=1))[0]
    if len(xs) and len(ys):
        canvas = canvas[ys[0]:ys[-1]+1, xs[0]:xs[-1]+1]
    return canvas


print("Stitching piece 0+1 ...")
mid = stitch_pair(imgs[0], imgs[1])
print("Stitching mid+2 ...")
panorama = stitch_pair(mid, imgs[2])

# ── 결과 시각화 ───────────────────────────────────────────────────────────────
TH = 160  # 썸네일 높이

def thumb(im, caption):
    r = TH / im.shape[0]
    t = cv2.resize(im, (int(im.shape[1] * r), TH))
    cv2.putText(t, caption, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return t

pieces_row = np.hstack([thumb(im, f"Piece {i+1}") for i, im in enumerate(imgs)])
src_thumb  = thumb(src,      "Original (people.jpg)")
pan_thumb  = thumb(panorama, "Panorama (SIFT+RANSAC)")

def pad_w(im, tw):
    if im.shape[1] < tw:
        pad = np.zeros((im.shape[0], tw - im.shape[1], 3), np.uint8)
        return np.hstack([im, pad])
    return im[:, :tw]

mw = max(pieces_row.shape[1], pan_thumb.shape[1], src_thumb.shape[1])
combined = np.vstack([pad_w(src_thumb, mw), pad_w(pieces_row, mw), pad_w(pan_thumb, mw)])

cv2.imwrite(os.path.join(OUTPUT_DIR, "practice9_panorama.png"),      combined)
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice9_panorama_full.png"), panorama)
print(f"\nOriginal : {w}x{h}")
print(f"Panorama : {panorama.shape[1]}x{panorama.shape[0]}")

cv2.imshow("Panorama Stitching (people.jpg)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
