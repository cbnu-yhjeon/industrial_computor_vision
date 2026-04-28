import cv2
import numpy as np
import os

IMAGE_PATH = "../../Data/scenetext01.jpg"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 전체 이미지를 3개의 overlapping 조각으로 분할 후 파노라마로 재합성
# ─────────────────────────────────────────────────────────────────────────────
src = cv2.imread(IMAGE_PATH)
if src is None:
    print("Image not found:", IMAGE_PATH)
    exit(1)

h, w = src.shape[:2]

# 조각 생성: 33% overlap
step = w // 3
overlap = w // 6
imgs = [
    src[:, 0            : step + overlap],
    src[:, step - overlap : 2 * step + overlap],
    src[:, 2 * step - overlap : w],
]

# ─────────────────────────────────────────────────────────────────────────────
# SIFT + RANSAC Homography 기반 스티칭
# ─────────────────────────────────────────────────────────────────────────────
def stitch_pair(img_left, img_right):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        print("Not enough matches:", len(good))
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum())
    print(f"  Matches: {len(good)}, Inliers: {inliers}")

    # Warp right image onto left's plane
    hl, wl = img_left.shape[:2]
    hr, wr = img_right.shape[:2]
    canvas_w = wl + wr

    # Compute offset: translate right image to align
    # H maps left→right; we need right→left, so use inv(H)
    H_inv = np.linalg.inv(H)

    warped_left = cv2.warpPerspective(img_left, H_inv, (canvas_w, hl))
    canvas = warped_left.copy()
    canvas[:hr, :wr] = img_right  # place right image at origin

    # Crop black border
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask_crop = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
    x_coords = np.where(mask_crop.any(axis=0))[0]
    y_coords = np.where(mask_crop.any(axis=1))[0]
    if len(x_coords) and len(y_coords):
        canvas = canvas[y_coords[0]:y_coords[-1]+1, x_coords[0]:x_coords[-1]+1]

    return canvas


print("Stitching img[0] + img[1]...")
mid = stitch_pair(imgs[0], imgs[1])
if mid is None:
    mid = np.hstack([imgs[0], imgs[1]])

print("Stitching mid + img[2]...")
panorama = stitch_pair(mid, imgs[2])
if panorama is None:
    panorama = np.hstack([mid, imgs[2]])

# ─────────────────────────────────────────────────────────────────────────────
# 결과 저장 및 시각화
# ─────────────────────────────────────────────────────────────────────────────
# 조각 3개 나란히 표시 (높이 동일하게 resize)
target_h = 200
pieces = []
for i, im in enumerate(imgs):
    r = target_h / im.shape[0]
    resized = cv2.resize(im, (int(im.shape[1] * r), target_h))
    cv2.putText(resized, f"Piece {i+1}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
    pieces.append(resized)
pieces_row = np.hstack(pieces)

# 파노라마 resize
r_pan = target_h / panorama.shape[0]
pan_small = cv2.resize(panorama, (int(panorama.shape[1] * r_pan), target_h))
cv2.putText(pan_small, "Panorama", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

# 원본 resize
r_src = target_h / src.shape[0]
src_small = cv2.resize(src, (int(src.shape[1] * r_src), target_h))
cv2.putText(src_small, "Original", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

# pad to same width for vstack
def pad_w(im, target_w):
    if im.shape[1] < target_w:
        pad = np.zeros((im.shape[0], target_w - im.shape[1], 3), dtype=np.uint8)
        return np.hstack([im, pad])
    return im[:, :target_w]

max_w = max(pieces_row.shape[1], pan_small.shape[1], src_small.shape[1])
combined = np.vstack([
    pad_w(src_small, max_w),
    pad_w(pieces_row, max_w),
    pad_w(pan_small, max_w),
])

cv2.imwrite(os.path.join(OUTPUT_DIR, "practice9_panorama.png"), combined)
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice9_panorama_full.png"), panorama)

print(f"\nOriginal size: {src.shape[1]}x{src.shape[0]}")
print(f"Panorama size: {panorama.shape[1]}x{panorama.shape[0]}")

cv2.imshow("Panorama Stitching", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
