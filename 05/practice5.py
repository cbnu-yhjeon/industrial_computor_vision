import cv2
import numpy as np
import random
import os

IMAGE_PATH = "../03/data/Lena.png"
OUTPUT_DIR = "output"


def load_gray(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def task1_otsu(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def task2_contours(binary):
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    canvas_ext = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    canvas_int = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(canvas_ext, [cnt], -1, (0, 255, 0), 1)
        else:
            cv2.drawContours(canvas_int, [cnt], -1, (0, 0, 255), 1)

    return canvas_ext, canvas_int


def task3_connected_components(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    component_ids = list(range(1, num_labels))

    print(f"[Task3] 총 컴포넌트 수 (배경 제외): {len(component_ids)}")
    print("스페이스를 누르면 랜덤 5개 컴포넌트를 표시합니다. 'q'로 종료.")

    h, w = binary.shape
    save_count = 0

    while True:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        chosen = random.sample(component_ids, min(5, len(component_ids)))
        for cid in chosen:
            color = (random.randint(50, 255),
                     random.randint(50, 255),
                     random.randint(50, 255))
            canvas[labels == cid] = color

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"task3_components_{save_count}.png"), canvas)
        save_count += 1

        cv2.imshow("Task3: Connected Components (SPACE=next, q=quit)", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def task4_distance_transform(binary):
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
    return dist_norm, dist_color


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img, gray = load_gray(IMAGE_PATH)

    binary = task1_otsu(gray)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "task1_otsu.png"), binary)
    cv2.imshow("Task1: Otsu Thresholding", binary)
    cv2.waitKey(0)

    canvas_ext, canvas_int = task2_contours(binary)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "task2_external_contours.png"), canvas_ext)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "task2_internal_contours.png"), canvas_int)
    cv2.imshow("Task2: External Contours (green)", canvas_ext)
    cv2.imshow("Task2: Internal Contours (red)", canvas_int)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    task3_connected_components(binary)

    dist_gray, dist_color = task4_distance_transform(binary)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "task4_distance_gray.png"), dist_gray)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "task4_distance_color.png"), dist_color)
    cv2.imshow("Task4: Distance Transform (gray)", dist_gray)
    cv2.imshow("Task4: Distance Transform (color)", dist_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
