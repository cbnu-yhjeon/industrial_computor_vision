import cv2
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/Lena.png')
OUT_PATH = os.path.join(os.path.dirname(__file__), '../data/output')
os.makedirs(OUT_PATH, exist_ok=True)

def show(name, img):
    # WSL2 환경: 파일로 저장 후 imshow 시도
    out_file = os.path.join(OUT_PATH, name.replace(' ', '_') + '.png')
    if img.dtype != np.uint8:
        save_img = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        save_img = img
    cv2.imwrite(out_file, save_img)
    print(f'[saved] {out_file}')

# 1. Lena 이미지를 컬러영상으로 읽고 화면에 출력
image = cv2.imread(DATA_PATH)
show('Lena_color', image)

# 2. 흑백으로 변환하고 출력
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show('Lena_grey', grey)

# 2-1. 흑백 이미지에 Histogram Equalization 적용
grey_eq = cv2.equalizeHist(grey)
show('Histogram_Equalization', grey_eq)

# 2-2. 흑백 이미지에 Gamma Correction 적용 (gamma=0.5)
grey_float = grey.astype(np.float32) / 255
gamma = 0.5
grey_gamma = np.power(grey_float, gamma)
show('Gamma_Correction', grey_gamma)

# 3. HSV 컬러 스페이스로 변환 (각 값 0~255로 정규화)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

show('H_channel', h)
show('S_channel', s)
show('V_channel', v)

# 3-1. H 채널에 Median Filter 적용
h_median = cv2.medianBlur(h, 7)
show('H_Median_Filter', h_median)

# 3-2. S 채널에 Gaussian Filter 적용
s_gaussian = cv2.GaussianBlur(s, (7, 7), 0)
show('S_Gaussian_Filter', s_gaussian)

# 3-3. V 채널에 Bilateral Filter 적용
v_bilateral = cv2.bilateralFilter(v, -1, 30, 10)
show('V_Bilateral_Filter', v_bilateral)
