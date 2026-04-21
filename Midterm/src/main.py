"""
단안 카메라 기반 전방 위험 객체 탐지 및 위험도 등급화
KITTI Dataset 사용

실행:
    python3 main.py --data ../dataset/training/image_2
    python3 main.py --data ../dataset/training/image_2 --debug
    python3 main.py --data ../dataset/training/image_2 --save
"""
import cv2
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess, get_visibility_label
from detection      import detect_objects, segment_road_objects
from feature        import extract_harris, extract_sift
from risk           import assess_risk
from visualize      import draw_results, make_debug_panel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="../dataset/training/image_2",
                        help="KITTI image_2 디렉토리 경로")
    parser.add_argument("--debug", action="store_true",
                        help="전처리 단계 디버그 패널 표시")
    parser.add_argument("--save",  action="store_true",
                        help="결과 이미지 output/ 저장")
    parser.add_argument("--max",   type=int, default=0,
                        help="처리할 최대 이미지 수 (0=전체)")
    return parser.parse_args()


def process_image(img: np.ndarray, debug: bool = False):
    h, w = img.shape[:2]

    # ── 1. 전처리 (Lec3/4) ───────────────────────────────────────────────────
    gray, equalized, blurred, edges, vis_score = preprocess(img)

    # ── 2. 객체 검출 (Lec5/6) ────────────────────────────────────────────────
    detections = detect_objects(edges, img.shape)
    if not detections:
        detections = segment_road_objects(gray, img.shape)

    # ── 3. 특징점 추출 (Lec7) ─────────────────────────────────────────────────
    harris_counts = extract_harris(gray, detections)
    sift_counts   = extract_sift(gray, detections)

    # ── 4. 위험도 평가 ────────────────────────────────────────────────────────
    results = [assess_risk(det, w, h, vis_score) for det in detections]

    # ── 5. 시각화 ─────────────────────────────────────────────────────────────
    canvas = draw_results(img, results, vis_score, harris_counts, sift_counts)

    if debug:
        canvas = make_debug_panel(gray, equalized, edges, canvas)

    return canvas, results, vis_score


def main():
    args = parse_args()
    output_dir = os.path.join(os.path.dirname(__file__), "../output")
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 목록 수집
    exts = {".png", ".jpg", ".jpeg"}
    if os.path.isfile(args.data):
        img_paths = [args.data]
    else:
        img_paths = sorted([
            os.path.join(args.data, f)
            for f in os.listdir(args.data)
            if os.path.splitext(f)[1].lower() in exts
        ])

    if args.max > 0:
        img_paths = img_paths[:args.max]

    if not img_paths:
        print(f"이미지를 찾을 수 없습니다: {args.data}")
        return

    print(f"총 {len(img_paths)}장 처리 시작")
    print("조작: SPACE=다음, q=종료, s=저장")

    for idx, path in enumerate(img_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        canvas, results, vis_score = process_image(img, debug=args.debug)

        # 통계 출력
        from risk import DANGER, CAUTION, SAFE, LEVEL_LABEL
        danger_cnt  = sum(1 for r in results if r.level == DANGER)
        caution_cnt = sum(1 for r in results if r.level == CAUTION)
        vis_lbl     = get_visibility_label(vis_score)
        print(f"[{idx+1:04d}/{len(img_paths)}] {os.path.basename(path)} | "
              f"Vis:{vis_score:.2f}({vis_lbl}) | "
              f"DANGER:{danger_cnt} CAUTION:{caution_cnt} SAFE:{len(results)-danger_cnt-caution_cnt}")

        # 자동 저장
        if args.save:
            out_path = os.path.join(output_dir, os.path.basename(path))
            cv2.imwrite(out_path, canvas)

        cv2.imshow("Forward Danger Detection", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            out_path = os.path.join(output_dir, f"save_{idx:04d}.png")
            cv2.imwrite(out_path, canvas)
            print(f"  저장: {out_path}")

    cv2.destroyAllWindows()
    print("완료")


if __name__ == "__main__":
    main()
