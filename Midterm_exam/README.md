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

## 위험도 기준

| 레벨 | 점수 범위 | 색상 | 의미 |
|------|----------|------|------|
| SAFE | 0.00 ~ 0.35 | 초록 | 안전 거리 확보 |
| CAUTION | 0.35 ~ 0.60 | 노랑 | 주의 필요 |
| DANGER | 0.60 ~ 1.00 | 빨강 | 즉각 위험 |

위험도 점수 = (크기 점수 × 0.5 + 중앙 거리 점수 × 0.5) × 시인성 가중치

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
