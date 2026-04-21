import cv2
import numpy as np
import os

IMAGE_PATH = "../03/data/Lena.png"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img1  = cv2.imread(IMAGE_PATH)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# ── 영상(2): 회전 + 약간의 노이즈로 움직임 시뮬레이션 ─────────────────────
h, w = img1.shape[:2]
M = cv2.getRotationMatrix2D((w // 2, h // 2), angle=10, scale=0.95)
img2  = cv2.warpAffine(img1, M, (w, h))
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
noise = np.random.normal(0, 8, gray2.shape).astype(np.int16)
gray2 = np.clip(gray2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def detect_all(gray, img):
    results = {}

    # FAST
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    results["FAST"] = fast.detect(gray, None)

    # Harris
    harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    harris = cv2.dilate(harris, None)
    pts = np.argwhere(harris > 0.01 * harris.max())
    results["Harris"] = [cv2.KeyPoint(float(x), float(y), 1) for y, x in pts]

    # Good Feature to Track
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        results["GFT"] = [cv2.KeyPoint(float(x), float(y), 1) for x, y in corners.reshape(-1, 2)]
    else:
        results["GFT"] = []

    # SIFT
    sift = cv2.SIFT_create()
    kp, _ = sift.detectAndCompute(gray, None)
    results["SIFT"] = kp

    return results

kp1 = detect_all(gray1, img1)
kp2 = detect_all(gray2, img2)

# ── 결과 출력 ─────────────────────────────────────────────────────────────────
print(f"{'방법':<8} {'영상1':>6} {'영상2':>6} {'변화율':>8}")
print("-" * 32)
for name in ["FAST", "Harris", "GFT", "SIFT"]:
    n1, n2 = len(kp1[name]), len(kp2[name])
    ratio = abs(n2 - n1) / max(n1, 1) * 100
    print(f"{name:<8} {n1:>6} {n2:>6} {ratio:>7.1f}%")

# ── 시각화: 4가지 방법 각각의 영상1/영상2 나란히 비교 ────────────────────────
COLORS = {"FAST": (0,255,0), "Harris": (0,0,255), "GFT": (0,255,255), "SIFT": (255,100,0)}
rows = []
for name in ["FAST", "Harris", "GFT", "SIFT"]:
    c1 = img1.copy()
    c2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
    for kp in kp1[name]:
        cv2.circle(c1, np.int32(kp.pt), 3, COLORS[name], 1)
    for kp in kp2[name]:
        cv2.circle(c2, np.int32(kp.pt), 3, COLORS[name], 1)
    label1 = f"{name} img1: {len(kp1[name])}"
    label2 = f"{name} img2: {len(kp2[name])}"
    cv2.putText(c1, label1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[name], 1)
    cv2.putText(c2, label2, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[name], 1)
    rows.append(np.hstack([c1, c2]))

combined = np.vstack(rows)
cv2.imwrite(os.path.join(OUTPUT_DIR, "practice2_motion_comparison.png"), combined)
cv2.imshow("Practice2: Feature Detection on Motion (img1 | img2)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
