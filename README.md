# Industrial Computer Vision

충북대학교 전자공학부 | 산업 컴퓨터비전 | 강의자: 황영배 교수님

---

## 디렉토리 구조

```
industrial_computor_vision/
├── Data/                         # 공용 이미지 데이터
│   ├── Lenna.png
│   ├── Lena_rotated.png
│   ├── BnW.png
│   └── scenetext01.jpg
│
├── 03/                           # Lecture 3 - Image Processing 기초
├── 05/                           # Lecture 5 - Boundary Extraction
├── 06/                           # Lecture 6 - Image Segmentation
└── 07/                           # Lecture 7 - Feature Detection
```

---

## 강의별 내용

### Lecture 3 - Image Processing 기초

| 파일 | 내용 |
|------|------|
| `src/01_matrix_manipulating.py` | NumPy 행렬 조작 |
| `src/02_converting_data_types.py` | 데이터 타입 변환 |
| `src/03_manipulating_image_channels.py` | 채널 분리/조합 |
| `src/04_converting_color_space.py` | 색공간 변환 (BGR, HSV, Gray) |
| `src/05_gamma_correction.py` | 감마 보정 |
| `src/06_histogram_equalization.py` | 히스토그램 평탄화 |
| `src/07_image_filtering.py` | 이미지 필터링 (Gaussian, Median, Bilateral) |
| `src/08_practice.py` | 실습 문제 |

---

### Lecture 5 - Boundary Extraction (2026.03.31)

**주요 이론:** Edge Detection, Hough Transform, Connected Component Labeling

#### 연습 코드 (`exercises/`)

| 파일 | 내용 |
|------|------|
| `binarization_using_otsu.py` | Otsu 알고리즘 이진화 |
| `finding_external_and_internal_contours.py` | 외부/내부 컨투어 검출 |
| `extracting_connected_component.py` | 연결 성분 추출 |
| `fitting_lines_and_circles.py` | 직선/타원 피팅 |
| `calculating_image_moments.py` | 이미지 모멘트 및 무게중심 |
| `working_with_curves.py` | Convex Hull, approxPolyDP |
| `checking_the_location_of_points.py` | 점의 내외부 판별 |
| `computing_distance_to_2d_point_set.py` | Distance Transform |

#### 실습 문제 (`practice5.py`)

| Task | 내용 |
|------|------|
| Task 1 | Otsu Thresholding |
| Task 2 | External / Internal Contour 검출 |
| Task 3 | Connected Component — 스페이스 키마다 랜덤 5개 표시 |
| Task 4 | Distance Transform 시각화 |

---

### Lecture 6 - Image Segmentation (2026.04.07)

**주요 이론:** Region Growing, Watershed, K-means Clustering, GrabCut

#### 연습 코드 (`exercises/`)

| 파일 | 내용 |
|------|------|
| `image_segmentation_using_kmeans.py` | K-means 이미지 분할 |
| `image_segmentation_using_watershed.py` | Watershed 분할 |
| `foreground_segmentation_using_grabcut.py` | GrabCut 전경 분리 |
| `canny_edge_detection.py` | Canny Edge (threshold 3단계 비교) |
| `line_and_circle_detection_using_hough_transform.py` | Hough 직선/원 검출 |

#### 실습 문제

| 파일 | 내용 |
|------|------|
| `practice1_kmeans_comparison.py` | K-means RGB vs RGBXY 비교 |
| `practice2_grabcut_interactive.py` | GrabCut 인터랙티브 (마우스 입력) |

> **practice2 조작법:** LMB 드래그=사각형 초기화/전경, RMB=배경, ENTER=실행, s=저장, r=리셋, q=종료

---

### Lecture 7 - Feature Detection (2026.04.14)

**주요 이론:** Harris Corner, FAST, Good Feature To Track, SIFT

#### 연습 코드 (`exercises/`)

| 파일 | 내용 |
|------|------|
| `corner_detection_harris_fast.py` | Harris Corner + FAST 검출 |
| `corner_detection_good_feature_to_track.py` | Shi-Tomasi 코너 검출 |
| `draw_keypoints_descriptors_and_matches.py` | SIFT 키포인트 시각화 및 매칭 |
| `detecting_scale_invariant_keypoints.py` | 스케일별 SIFT 키포인트 비교 |

#### 실습 문제

| 파일 | 내용 |
|------|------|
| `practice1_feature_comparison.py` | FAST/Harris/GFT/SIFT 위치 비교 (다른 모양으로 표시) |
| `practice2_feature_motion_comparison.py` | 원본 vs 회전+노이즈 영상의 특징 검출 안정성 비교 |

> **실습2 결과 요약:** SIFT(0.3%) > GFT(0.0%*) > Harris(18.1%) > FAST(98.4%) 순으로 안정적

---

## 실행 환경

```
Python 3.x
OpenCV 4.13.0
NumPy
```

## 실행 방법

```bash
# 예시: Lecture 5 실습 문제
cd 05
python3 practice5.py

# 예시: Lecture 7 연습 코드
cd 07/exercises
python3 corner_detection_harris_fast.py
```

결과 이미지는 각 강의 디렉토리의 `output/` 폴더에 저장됩니다.
