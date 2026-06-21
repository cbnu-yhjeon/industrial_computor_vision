# 전방 위험 탐지 시스템 — 고도화 (Final Exam)

충북대학교 산업 컴퓨터비전 | Midterm_exam 고도화 프로젝트

---

## 개요

Midterm_exam의 단안 카메라 위험 탐지 시스템을 **강의 3~12주차 전체 기법**으로 확장한 고도화 버전입니다.  
딥러닝 없이 순수 영상처리 기법만으로 Phase 1~6을 단계적으로 구현하고, 공개 주행 데이터셋으로 검증했습니다.

### Midterm → Final 비교

| 항목 | Midterm | Final |
|------|---------|-------|
| 전처리 | equalizeHist | **CLAHE** (야간·역광 대응) |
| 차선 | 없음 | **Hough Transform 차선 검출 + 위험 가중치** |
| 특징점 | Harris/SIFT 표시만 | **Harris/SIFT/HoG → 위험도 점수 반영** |
| 객체 분류 | 없음 | **HOG+SVM 보행자 검출, 차량/보행자 분류** |
| 추적 | 없음 | **Lucas-Kanade 광학 흐름 + TTC** |
| 거리 추정 | 면적 기반(부정확) | **핀홀 모델 Z = (H_real × fy) / H_pixel** |
| 분할 | 단순 컨투어 | **Watershed + 통계적 이상치 제거** |
| 출력 | 이미지 표시 | **MP4 저장 + CSV 통계 리포트** |
| 위험도 공식 | 0.5×크기 + 0.5×중앙 | **0.45×크기 + 0.45×중앙 + 0.10×특징 × vis × TTC × lane** |

---

## 파이프라인 (10단계)

```
입력 프레임
    │
    ▼
[Phase 1-A] CLAHE 전처리          preprocessing.py
    │         vis_score < 0.5 → CLAHE 적용, 아닌 경우 gray 유지
    ▼
[Phase 1-B] Canny + Hough 차선 검출   lane.py
    │         HoughLinesP → 좌/우 분리 → 차선 마스크
    ▼
[Phase 5]   객체 검출 + Watershed      detection.py
    │         Canny 컨투어 → Watershed 겹친 객체 분리
    │         → RANSAC식 통계 이상치 필터링
    ▼
[Phase 2-A] 특징점 추출               feature.py
    │         Harris / SIFT / HoG (64×64 디스크립터)
    ▼
[Phase 2-B] 객체 분류                 classifier.py
    │         aspect ratio + 특징점 수 → vehicle / pedestrian
    ▼
[Phase 3]   Lucas-Kanade 추적 + TTC   tracking.py
    │         IoU 매칭 → GoodFeaturesToTrack → LK Flow
    │         TTC = area_prev / Δarea
    ▼
[Phase 4]   핀홀 거리 추정            calibration.py
    │         Z = (H_real × fy) / H_pixel
    │         camera_mat.npy: fx=613.6, fy=613.0
    ▼
[Phase 1-C] 위험도 평가               risk.py
    │         score = (0.45×크기 + 0.45×중앙 + 0.10×특징)
    │                 × vis_weight × ttc_weight × lane_weight
    │         DANGER≥0.6 / CAUTION≥0.35 / SAFE
    ▼
[Phase 6]   시각화 + 저장             visualize.py / main.py
              바운딩박스 + 거리 + TTC + 트랙ID + 속도화살표
              MP4 저장 / CSV 통계
```

---

## 모듈 구조

```
Final_exam/
├── src/
│   ├── main.py            진입점 — 이미지 뷰어 모드 (argparse)
│   ├── process_video.py   진입점 — 비디오 배치 처리 모드
│   ├── preprocessing.py   Phase 1-A: CLAHE 전처리, 시인성 점수
│   ├── lane.py            Phase 1-B: Hough 차선 검출, 마스크 생성
│   ├── detection.py       Phase 5:   Canny 컨투어, Watershed, 이상치 제거
│   ├── feature.py         Phase 2-A: Harris, SIFT, HoG 특징 추출
│   ├── classifier.py      Phase 2-B: HOG+SVM 보행자, 규칙 기반 분류
│   ├── tracking.py        Phase 3:   LK 광학 흐름, TTC, IoU 매칭
│   ├── calibration.py     Phase 4:   핀홀 거리 추정, camera_mat 로드
│   ├── risk.py            Phase 1-C: 통합 위험도 평가, RiskResult
│   └── visualize.py       Phase 6:   시각화, 4분할 디버그 패널
└── output/                처리 결과 이미지 / 비디오 / CSV
```

---

## 적용 기법 (강의별)

| Phase | 강의 | 적용 기법 |
|-------|------|----------|
| 1-A | Lec 3 | CLAHE (Contrast Limited Adaptive Histogram Equalization) |
| 1-B | Lec 5 | Hough Transform (`HoughLinesP`), Canny Edge |
| 1-C | Lec 7 | Harris Corner, SIFT → 위험도 feature_score (10% 반영) |
| 2-A | Lec 12 | HoG Descriptor (`HOGDescriptor`, 64×64, 9-bin) |
| 2-B | Lec 12 | HOG + Pre-trained SVM (`getDefaultPeopleDetector`) |
| 3   | Lec 9  | Lucas-Kanade Optical Flow, TTC = area_prev / Δarea |
| 4   | Lec 10 | 핀홀 카메라 모델, 카메라 캘리브레이션 파라미터 |
| 5   | Lec 6  | Watershed Segmentation, 통계적 이상치 필터 (RANSAC 착상) |
| 6   | Lec 8+ | `cv2.VideoWriter` MP4 출력, CSV 통계 리포트 |

---

## 위험도 공식

```
base_score = 0.45 × size_score
           + 0.45 × center_score
           + 0.10 × feature_score

score = base_score × vis_weight × ttc_weight × lane_weight
```

| 항목 | 계산 방법 |
|------|----------|
| **size_score** | 거리 있으면 `clip(1 - (d-5)/45, 0, 1)` / 없으면 `area / (img_area × 0.08)` |
| **center_score** | `1 - √(dx²+dy²)/√2` (화면 중앙에 가까울수록 높음) |
| **feature_score** | `0.5 × (harris+sift)/500 + 0.5 × clip(hog×10, 0, 1)` |
| **vis_weight** | `1.0 + 0.3 × (1 - vis_score)` (최대 1.3×, 야간 가중) |
| **ttc_weight** | `1.0 + 0.4 × max(0, 1 - TTC/3)` (최대 1.4×, TTC < 3s) |
| **lane_weight** | `1.1` (차선 내) / `0.85` (차선 밖) |

등급: **DANGER** ≥ 0.60 · **CAUTION** ≥ 0.35 · **SAFE** < 0.35

---

## 실행 방법

### 이미지 뷰어 모드

```bash
cd Final_exam/src

# 기본 실행 (Data 디렉토리 전체)
python3 main.py --data ../../Data

# 디버그 패널 (4분할: Gray / CLAHE / Edges+Lane / Result)
python3 main.py --data ../../Data --debug

# 결과 저장 + 비디오 + CSV
python3 main.py --data ../../Data --save --video --csv

# 단일 이미지
python3 main.py --data ../../Data/people.jpg --debug
```

키 조작: `d / →` 다음 · `a / ←` 이전 · `s` 저장 · `q` 종료

### 비디오 배치 처리 모드

```bash
# 주행 영상 처리 (skip=3: 3프레임마다 1회)
python3 process_video.py --input <video.mp4> --skip 3

# 처리 프레임 수 제한 + 보행자 검출 비활성화 (빠른 테스트)
python3 process_video.py --input <video.mp4> --max 100 --no-ped
```

출력: `<입력파일명>_result.mp4` + `<입력파일명>_stats.csv`

### 공개 데이터셋 다운로드

```bash
pip install yt-dlp
mkdir -p Data/open_dataset && cd Data/open_dataset
yt-dlp -f "best[height<=480][ext=mp4]" -o "driving.%(ext)s" "<YouTube URL>"
python3 ../../Final_exam/src/process_video.py --input driving.mp4 --skip 3
```

---

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data` | `../../Data` | 이미지 파일 또는 디렉토리 |
| `--debug` | False | 4분할 전처리 패널 출력 |
| `--save` | False | 결과 이미지 output/ 저장 |
| `--video` | False | output/result.mp4 저장 |
| `--csv` | False | output/stats.csv 저장 |
| `--calib` | Data/pinhole_calib | 캘리브레이션 디렉토리 |
| `--no-ped` | False | HOG 보행자 검출 비활성화 |
| `--skip` | 3 | 비디오 처리 시 프레임 간격 |

---

## 공개 주행 데이터셋 검증 결과

대시캠 주행 영상 (72초, 640×360, 24fps) 에 파이프라인 적용:

| 항목 | 결과 |
|------|------|
| 처리 프레임 | 578프레임 (skip=3) |
| 처리 속도 | **~50 fps** (실시간 24fps의 2배) |
| 평균 객체 수 | **5.3개 / 프레임** |
| 차량 검출 | **2,911건** |
| 보행자 검출 | **160건** |
| DANGER 누적 | **352건** |
| CAUTION 누적 | **1,206건** |
| 최근접 거리 | min **4.1m** · avg 18.1m · max 61.3m |
| TTC 산출 구간 | **278프레임 (47.8%)** — 연속 추적 성공 |
| **충돌 임박 (TTC ≤ 1s)** | **186회** |
| TTC 경고 (1~3s) | 48회 |

> TTC(Time-To-Collision)는 연속 프레임에서 같은 객체를 추적해야 산출되므로  
> 단일 이미지 테스트에서는 N/A로 표시됩니다.

---

## 의존성

```
Python 3.x
OpenCV 4.13.0
NumPy
```

카메라 캘리브레이션 파라미터: `Data/pinhole_calib/camera_mat.npy`

```
fx = 613.6  fy = 613.0
cx = 331.2  cy = 235.0
```

---

## 참고 강의

| 강의 | 주제 |
|------|------|
| Lec 3 | Image Processing — Histogram, CLAHE |
| Lec 5 | Boundary Extraction — Canny, Hough Transform |
| Lec 6 | Image Segmentation — Watershed, K-means |
| Lec 7 | Feature Detection — Harris, SIFT |
| Lec 8 | Machine Learning in CV — SVM, RANSAC |
| Lec 9 | Motion Analysis — Lucas-Kanade Optical Flow |
| Lec 10 | Camera Geometry — 핀홀 모델, 캘리브레이션 |
| Lec 12 | Object Detection — HoG, HOG+SVM |
