import cv2
import numpy as np
import os

IMAGE_PATH = "../../03/data/Lena.png"
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img  = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=100,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=3,
)

canvas = img.copy()
if corners is not None:
    for x, y in np.int32(corners).reshape(-1, 2):
        cv2.circle(canvas, (x, y), 4, (0, 255, 255), -1)
    print(f"[GoodFeaturesToTrack] 검출된 코너: {len(corners)}개")

cv2.imwrite(os.path.join(OUTPUT_DIR, "good_features_to_track.png"), canvas)
cv2.imshow("Good Features To Track (yellow)", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
