# 단안 카메라 기반 전방 위험 객체 탐지 및 위험도 등급화

강의 3~7주차 기법을 통합하여 KITTI 자율주행 이미지에서 전방 차량/보행자를 탐지하고 위험도를 3단계로 분류하는 시스템입니다.

---

## 파이프라인

```
입력 이미지
    │
    ▼
[1] 전처리 (preprocessing.py)
    - Grayscale 변환
    - 히스토그램 기반 시인성 점수 계산 (Lec 3)
    - 시인성 낮을 시 Histogram Equalization 적용 (Lec 3)
    - Gaussian Blur 노이즈 제거 (Lec 3/4)
    - Canny Edge Detection (Lec 6)
    │
    ▼
[2] 객체 검출 (detection.py)
    - Morphology + Contour 기반 검출 (Lec 5/6)
    - 검출 실패 시 Otsu + Connected Component 보조 검출 (Lec 5/6)
    - 차량/보행자 크기 필터 (하늘 영역, 노이즈 제거)
    │
    ▼
[3] 특징점 추출 (feature.py)
    - 각 객체 ROI 내 Harris Corner 수 계산 (Lec 7)
    - 각 객체 ROI 내 SIFT 키포인트 수 계산 (Lec 7)
    │
    ▼
[4] 위험도 평가 (risk.py)
    - 크기 점수 (객체 면적 / 이미지 면적)
    - 중앙 거리 점수 (화면 중심과의 거리)
    - 시인성 가중치 (낮을수록 위험도 최대 1.3배 증가)
    - SAFE / CAUTION / DANGER 3단계 분류
    │
    ▼
[5] 시각화 (visualize.py)
    - 바운딩 박스 + 위험도 레이블 + Harris/SIFT 수치 표시
    - 우측 HUD (시인성, 객체 수, 등급별 카운트)
    - DANGER 탐지 시 경고 배너 + 중앙선 강조
```

---

## 객체 감지 상세

### 1단계: Canny Edge + Contour (주 검출)

```
Canny Edge 결과
    │
    ▼
Morphology Close (구멍 메우기, 3×3 커널 2회)
    │
    ▼
Dilate (윤곽 팽창, 1회)
    │
    ▼
findContours (외곽선 추출)
    │
    ▼
차량 크기 필터 적용 → 면적 상위 10개 반환
```

엣지로 감지된 윤곽선을 닫고 팽창시켜 객체 덩어리를 만든 뒤 컨투어를 추출합니다.

### 2단계: Otsu + Connected Component (보조 검출)

1단계에서 아무것도 검출되지 않을 때만 실행됩니다.

```
Grayscale 이미지
    │
    ▼
Otsu Thresholding (자동 이진화, 반전)
    │
    ▼
상단 30% 하늘 영역 제거
    │
    ▼
connectedComponentsWithStats (연결된 픽셀 덩어리 분석)
    │
    ▼
차량 크기 필터 적용 → 면적 상위 10개 반환
```

### 공통: 차량 크기 필터

두 방법 모두 아래 조건을 모두 통과한 객체만 인정합니다.

| 조건 | 이유 |
|------|------|
| 면적 > 이미지의 0.1% | 노이즈 제거 |
| 면적 < 이미지의 15% | 도로/배경 제거 |
| 가로 < 이미지 너비의 70% | 도로 구분선 제거 |
| 세로 ≥ 15px | 얇은 수평선 제거 |
| y 위치 > 이미지 높이의 30% | 하늘 영역 제거 |

### 객체 감지 한계

- 딥러닝 없이 순수 영상처리만 사용하므로 객체와 배경 대비가 낮으면 엣지가 잘 안 잡힘
- 겹친 차량은 하나의 덩어리로 인식될 수 있음
- 클래스 구분 없음 (차량인지 보행자인지 알 수 없음)

---

## 위험도 판단 상세

### 구성 요소

**1. 크기 점수 (size_score) — 50% 비중**

```
size_score = min(객체 면적 / (이미지 면적 × 0.08), 1.0)
```

객체 면적이 화면의 8% 이상이면 점수 1.0. 객체가 클수록 카메라에 가까이 있다고 판단합니다.

**2. 중앙 거리 점수 (center_score) — 50% 비중**

```
dist_norm    = sqrt(dx² + dy²) / sqrt(2)
center_score = 1.0 - dist_norm
```

객체 중심이 화면 정중앙에 가까울수록 점수 1.0. 정면에 있는 객체일수록 위험하다고 판단합니다.

**3. 시인성 가중치 (vis_weight) — 최대 1.3배 증폭**

```
vis_weight = 1.0 + 0.3 × (1.0 - visibility_score)
```

야간/역광 등 시인성이 낮을수록 동일한 상황도 더 위험하게 판단합니다.

| 시인성 | 가중치 |
|--------|--------|
| Good (≥ 0.6) | 1.0 ~ 1.12 |
| Moderate (0.35 ~ 0.6) | 1.12 ~ 1.20 |
| Poor (< 0.35) | 1.20 ~ 1.30 |

### 최종 계산

```
score = (크기점수 × 0.5 + 중앙거리점수 × 0.5) × 시인성가중치
```

### 위험도 기준

| 레벨 | 점수 범위 | 색상 | 의미 |
|------|----------|------|------|
| SAFE | 0.00 ~ 0.35 | 초록 | 안전 거리 확보 |
| CAUTION | 0.35 ~ 0.60 | 노랑 | 주의 필요 |
| DANGER | 0.60 ~ 1.00 | 빨강 | 즉각 위험 |

### 위험도 판단 한계

실제 거리를 측정하지 않고 **"크기가 크면 가깝다"** 는 가정에 의존합니다. 실제로 큰 트럭이 멀리 있어도 DANGER로 분류될 수 있습니다. 단안 카메라의 근본적인 한계로, 스테레오 카메라나 LiDAR 없이는 실제 거리를 알 수 없습니다.

---

## 디렉토리 구조

```
Midterm_exam/
├── README.md
├── src/
│   ├── main.py            # 진입점, 이미지 루프 및 통계 출력
│   ├── preprocessing.py   # 전처리 파이프라인 및 시인성 평가
│   ├── detection.py       # 객체 검출 (Contour / Connected Component)
│   ├── feature.py         # 특징점 추출 (Harris, SIFT)
│   ├── risk.py            # 위험도 등급화
│   └── visualize.py       # 결과 시각화
└── output/                # 저장된 결과 이미지 (--save 옵션 시 생성)
```

---

## 권장 데이터셋

### 1순위: KITTI Vision Benchmark (권장)
- 사이트: https://www.cvlibs.net/datasets/kitti/
- 경로: `dataset/training/image_2/` (좌측 컬러 카메라, 1242×375)
- 이유: 코드가 KITTI 이미지 해상도 및 차량 크기 비율 기준으로 필터 튜닝되어 있음
- 권장 세트: **Object Detection** 또는 **Raw Data** 의 도심/고속도로 시퀀스

### 2순위: BDD100K (다양한 환경 테스트)
- 사이트: https://bdd-data.berkeley.edu/
- 해상도: 1280×720
- 이유: 주간/야간/우천 등 다양한 시인성 조건 포함 → 시인성 가중치 로직 검증에 적합
- 주의: 해상도가 달라 크기 필터 임계값 조정 필요

### 3순위: 직접 촬영 영상 (간이 테스트)
- 전방 블랙박스 영상 또는 스마트폰 촬영 영상에서 프레임 추출
- `ffmpeg -i video.mp4 -vf fps=5 frames/%04d.png` 로 프레임 추출 가능
- 주의: KITTI와 카메라 높이/앵글이 다르면 하늘 영역 제거 비율(0.30) 조정 필요

---

## 실행 방법

```bash
# 기본 실행
python3 src/main.py --data ../dataset/training/image_2

# 전처리 디버그 패널 표시 (4분할 화면)
python3 src/main.py --data ../dataset/training/image_2 --debug

# 결과 이미지 output/ 에 저장
python3 src/main.py --data ../dataset/training/image_2 --save

# 처음 50장만 처리
python3 src/main.py --data ../dataset/training/image_2 --max 50
```

### 키 조작
| 키 | 동작 |
|----|------|
| SPACE | 다음 이미지 |
| s | 현재 프레임 저장 |
| q | 종료 |

---

## 의존성

```bash
pip install opencv-python numpy
```

---

## 강의 연계 요약

| 강의 | 적용 내용 |
|------|----------|
| Lec 3 | Histogram Equalization, 시인성 점수 계산 |
| Lec 4 | Gaussian Blur 노이즈 제거 |
| Lec 5 | Contour 검출, Connected Component 분석 |
| Lec 6 | Canny Edge Detection, Otsu Thresholding, Segmentation |
| Lec 7 | Harris Corner Detection, SIFT 키포인트 추출 |
