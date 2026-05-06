# 제주 SFTS 시공간 예보 시스템

기상청 2주 예보와 진드기 밀도 GAM 모델을 결합하여 제주 지역의 SFTS(중증열성혈소판감소증후군) 위험을 **읍·면·동 × 일 단위**로 예측하는 Streamlit 기반 웹 앱입니다.

> 제주감염병관리지원단 · 제주대학교병원 감염내과 SFTS 경보 시스템 고도화 프로토타입

## 주요 기능

- **시공간 예측 지도** — 14일 동안의 일별 위험도를 인터랙티브 슬라이더로 탐색
- **3중 레이어** — 4단계 경보 / 예측 발생률 λ / 진드기 밀도 D(s) 토글
- **시계열 추이** — 위험 상위 8개 지역의 14일 추이와 4단계 임계값 시각화
- **모델 진단** — 공간 baseline과 시간적 기상 영향을 분리하여 검증

## 시공간 모델

두 단계 음이항 회귀 모델로 sparse 데이터(12년, 약 120명)에서도 안정적으로 작동합니다.

```
log λ(s, t) = β₀ + β_T · T(s, t-ℓ) + β_RH · RH(s, t-ℓ) + log D(s)
```

| 변수 | 의미 | 출처 |
|---|---|---|
| `D(s)` | 지역별 진드기 상대 밀도 | GAM 공간 모델 |
| `T, RH` | ℓ일 전 기상 (lag 7-14일) | 기상청 2주 예보 API |
| `β` | 회귀계수 | 12년 SFTS 임상 데이터 학습 |

위험은 사분위수 기준으로 **관심 · 주의 · 경고 · 위험** 4단계로 분류됩니다.

## 빠른 시작 (로컬)

```bash
# 1) 저장소 복제
git clone https://github.com/<your-username>/sfts-spatiotemporal.git
cd sfts-spatiotemporal

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 앱 실행
streamlit run sfts_spatiotemporal.py
```

브라우저에서 `http://localhost:8501`이 자동으로 열립니다.

## Streamlit Community Cloud 배포

1. 본 저장소를 GitHub에 업로드합니다 (이 README가 포함된 폴더 전체).
2. [share.streamlit.io](https://share.streamlit.io)에 접속하여 GitHub 계정으로 로그인합니다.
3. **"New app"** → 저장소·브랜치(`main`)·메인 파일(`sfts_spatiotemporal.py`) 선택 → **Deploy**.
4. 약 3-5분 후 `https://<app-name>.streamlit.app` URL이 발급됩니다.

배포 후에는 GitHub에 코드를 push할 때마다 자동으로 재배포됩니다.

## 실데이터 연동 (production)

프로토타입의 모의 데이터 함수 4개를 다음과 같이 교체합니다.

| 함수 | 모의 데이터 | 실데이터 |
|---|---|---|
| `create_jeju_grid()` | 6×9 격자 폴리곤 | 행정안전부 / SGIS 읍·면·동 GeoJSON |
| `get_tick_density()` | 해안·동부 가중 시뮬레이션 | GAM 모델 예측값 (`gam_predictions.csv`) |
| `get_kma_forecast()` | 사인파 + 정규분포 노이즈 | 기상청 단기예보 + 중기예보 API |
| `compute_risk()` | 예시 회귀계수 | 12년 임상 데이터로 학습한 NegativeBinomial 계수 |

기상청 API 사용을 위해 [공공데이터포털](https://www.data.go.kr) 인증키를 발급받아 `.streamlit/secrets.toml`에 추가합니다 (해당 파일은 `.gitignore`에 포함되어 외부 노출되지 않습니다).

```toml
# .streamlit/secrets.toml
KMA_API_KEY = "your-decoded-key-here"
```

## 폴더 구조

```
sfts-spatiotemporal/
├── sfts_spatiotemporal.py     # 메인 Streamlit 앱
├── requirements.txt           # Python 의존성
├── README.md                  # 본 문서
├── .gitignore                 # Git 제외 항목
└── .streamlit/
    └── config.toml            # 테마 및 서버 설정
```

## 향후 고도화 로드맵

- 기상청 API 실시간 연동 (단기예보 + 중기예보 결합)
- 행정구역 단위 GeoJSON 적용 (43개 읍·면·동)
- Bayesian 시공간 모델 (PyMC / R-INLA) — 불확실성 정량화
- TimestampedGeoJson 기반 자동 14일 애니메이션
- 모바일 앱 LBS 푸시 연동
- 도청·KDCA 행정 대시보드 분리

## 라이선스 및 인용

본 프로토타입은 제주감염병관리지원단의 SFTS 경보 시스템 고도화 연구 일부로 개발되었습니다. 사용·인용 시 다음 정보를 명시해 주십시오.

- 제주대학교병원 감염내과 SFTS 연구팀
- 제주감염병관리지원단

문의: 유정래 (제주대학교병원 감염내과)
