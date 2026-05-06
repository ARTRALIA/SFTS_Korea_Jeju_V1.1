"""
=================================================================
제주 SFTS 시공간 예보 시스템 (Spatio-Temporal Prototype)
=================================================================
- 입력: 기상청 2주 예보 (모의) + 진드기 밀도 GAM 모델 (모의)
- 모델: λ(s,t) = D(s) × exp(β_T·T(s,t-7) + β_RH·RH(s,t-7))
- 출력: 읍·면·동 × 14일 위험 단계 지도 (시간 슬라이더)

실행: streamlit run sfts_spatiotemporal.py
의존성: streamlit folium streamlit-folium geopandas shapely altair pandas numpy
=================================================================
"""

import json
from datetime import datetime, timedelta

import altair as alt
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import Polygon
from streamlit_folium import st_folium

# ─────────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="제주 SFTS 시공간 예보",
    page_icon="🗺️",
    layout="wide",
)

st.title("제주 SFTS 시공간 예보 시스템")
st.caption(
    "기상청 2주 예보 × 진드기 밀도 GAM × 임상 회귀 → "
    "읍·면·동 일 단위 위험 단계 예측 (프로토타입)"
)

# ─────────────────────────────────────────────
# 1. 제주 행정구역 (모의 격자)
#    실제 적용: 행정안전부/SGIS 읍·면·동 GeoJSON 로드
#    예) gdf = gpd.read_file("jeju_eupmyeondong.geojson")
# ─────────────────────────────────────────────
@st.cache_data
def create_jeju_grid():
    """제주를 6×9 격자로 분할 — 실제 시스템에서는 GeoJSON 교체"""
    region_names = [
        "한경면", "한림읍", "애월읍", "노형동", "이도동", "조천읍", "구좌읍",
        "고산리", "한림항", "제주공항", "삼도동", "건입동", "함덕", "월정",
        "대정읍", "안덕면", "중문동", "서홍동", "성산읍", "표선면", "남원읍",
        "마라도", "송악산", "예래동", "정방", "보목", "쇠소깍", "우도",
    ]

    lat_steps = np.linspace(33.10, 33.55, 7)
    lon_steps = np.linspace(126.15, 126.95, 10)

    rows = []
    idx = 0
    for i in range(6):
        for j in range(9):
            if idx >= len(region_names):
                break
            poly = Polygon([
                (lon_steps[j], lat_steps[i]),
                (lon_steps[j + 1], lat_steps[i]),
                (lon_steps[j + 1], lat_steps[i + 1]),
                (lon_steps[j], lat_steps[i + 1]),
            ])
            cx, cy = poly.centroid.x, poly.centroid.y
            # 한라산 정상 인근 제외
            if (cx - 126.55) ** 2 + (cy - 33.36) ** 2 < 0.025:
                continue
            rows.append({
                "region_id": f"R{idx:02d}",
                "region_name": region_names[idx],
                "geometry": poly,
                "centroid_lat": cy,
                "centroid_lon": cx,
            })
            idx += 1

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ─────────────────────────────────────────────
# 2. 공간 baseline: 진드기 밀도 D(s)
#    실제: 김미선 교수님 GAM 모델 예측 결과 (negative binomial GLM + GAM)
#    예) tick_density = pd.read_csv("gam_predictions.csv")
# ─────────────────────────────────────────────
@st.cache_data
def get_tick_density(_gdf):
    """진드기 상대 밀도 (모의) — 해안·동부 가중"""
    rng = np.random.default_rng(42)
    densities = []
    for _, row in _gdf.iterrows():
        coast_factor = abs(row.centroid_lat - 33.36) * 8   # 해안 거리
        east_factor = max(0, row.centroid_lon - 126.7) * 5  # 동부 (구좌·성산)
        noise = rng.gamma(2, 0.4)
        densities.append(0.5 + coast_factor + east_factor + noise)
    return np.array(densities)


# ─────────────────────────────────────────────
# 3. 기상청 2주 예보
#    실제: 기상청 단기예보 (3일, 5km) + 중기예보 (10일, 시·군) API
#    https://www.data.go.kr → 동네예보·중기예보 통합 API
# ─────────────────────────────────────────────
@st.cache_data
def get_kma_forecast(_gdf, days=14):
    """14일 일별 기온·습도·강수 (모의)"""
    rng = np.random.default_rng(123)
    base_date = pd.Timestamp(datetime.now().date())
    rows = []
    for d in range(days):
        date = base_date + pd.Timedelta(days=d)
        for _, region in _gdf.iterrows():
            # 5월 제주 평균 18-22°C, RH 70%
            T = 19.5 + 2.5 * np.sin(d * 0.35) + rng.normal(0, 1.2)
            RH = 72 + 8 * np.cos(d * 0.4) + rng.normal(0, 4)
            # 동부·해안 약간 더 따뜻
            T += max(0, region.centroid_lon - 126.6) * 0.8
            P = max(0, rng.gamma(0.5, 2) - 1)
            rows.append({
                "region_id": region.region_id,
                "date": date,
                "day_offset": d,
                "temperature": round(T, 1),
                "humidity": round(min(100, max(40, RH)), 1),
                "precipitation": round(P, 1),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────
# 4. 시공간 회귀: λ(s, t) = D(s) × exp(β_T·T_lag + β_RH·RH_lag)
#    실제: 12년 임상 데이터로 회귀계수 학습 (negative binomial)
#    예) statsmodels.discrete.count_model.NegativeBinomial.fit()
# ─────────────────────────────────────────────
def compute_risk(gdf, tick_density, forecast_df, lag_days=7):
    """
    Parameters
    ----------
    lag_days : 기상 → 발병 사이의 지연 (진드기 활성화 + 잠복기)
    """
    # 회귀계수 (예시 - 실데이터로 재학습 필요)
    BETA_0 = -3.5       # 절편
    BETA_T = 0.08       # 기온 계수 (°C 당 위험 8% 증가)
    BETA_RH = 0.015     # 습도 계수
    BETA_TICK = 1.0     # log(D(s)) 계수

    density_map = dict(zip(gdf.region_id, tick_density))
    rows = []

    for region_id in gdf.region_id:
        D = density_map[region_id]
        region_fc = forecast_df[forecast_df.region_id == region_id].sort_values("day_offset")

        for d in range(14):
            # 예보 d일의 위험 ← (d - lag)일 기상 (음수면 d=0 사용)
            ref_d = max(0, d - lag_days) if d >= lag_days else d
            ref = region_fc[region_fc.day_offset == ref_d].iloc[0]

            log_lambda = (
                BETA_0
                + BETA_T * ref.temperature
                + BETA_RH * ref.humidity
                + BETA_TICK * np.log(D)
            )
            lam = float(np.exp(log_lambda))
            rows.append({
                "region_id": region_id,
                "day_offset": d,
                "date": forecast_df[
                    (forecast_df.region_id == region_id) & (forecast_df.day_offset == d)
                ].iloc[0]["date"],
                "lambda": lam,
                "tick_density": float(D),
                "ref_temperature": float(ref.temperature),
                "ref_humidity": float(ref.humidity),
            })

    risk_df = pd.DataFrame(rows)

    # 4단계 분류 (전체 기간·전체 지역의 사분위수 기준)
    q = risk_df["lambda"].quantile([0.25, 0.5, 0.75]).values

    def to_tier(x):
        if x < q[0]:
            return "관심"
        if x < q[1]:
            return "주의"
        if x < q[2]:
            return "경고"
        return "위험"

    risk_df["tier"] = risk_df["lambda"].apply(to_tier)
    risk_df = risk_df.merge(
        gdf[["region_id", "region_name"]], on="region_id"
    )
    return risk_df, q


# ─────────────────────────────────────────────
# 5. 색상 팔레트 (4단계 경보)
# ─────────────────────────────────────────────
TIER_COLORS = {
    "관심": "#2563eb",   # 파랑
    "주의": "#16a34a",   # 초록
    "경고": "#f59e0b",   # 주황
    "위험": "#dc2626",   # 빨강
}
TIER_ORDER = ["관심", "주의", "경고", "위험"]

# ─────────────────────────────────────────────
# 6. 사이드바 컨트롤
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("예보 설정")

    day_offset = st.slider(
        "예보 시점 (일 후)",
        min_value=0, max_value=13, value=7, step=1,
        help="0 = 오늘, 7 = 1주 후, 13 = 14일 후",
    )

    lag_days = st.slider(
        "기상 → 발병 지연 (일)",
        min_value=0, max_value=14, value=7,
        help="진드기 활성화(3-5일) + 잠복기(평균 9일) 반영",
    )

    layer_mode = st.radio(
        "지도 레이어",
        ["4단계 경보", "예측 발생률 λ", "진드기 밀도 D(s)"],
    )

    st.markdown("---")
    pred_date = datetime.now().date() + timedelta(days=day_offset)
    st.metric("예보 일자", pred_date.strftime("%Y-%m-%d"))
    st.metric("요일", ["월","화","수","목","금","토","일"][pred_date.weekday()] + "요일")

# ─────────────────────────────────────────────
# 7. 데이터 파이프라인 실행
# ─────────────────────────────────────────────
gdf = create_jeju_grid()
tick_density = get_tick_density(gdf)
forecast_df = get_kma_forecast(gdf)
risk_df, quantiles = compute_risk(gdf, tick_density, forecast_df, lag_days=lag_days)

day_risk = risk_df[risk_df.day_offset == day_offset].copy()
gdf_today = gdf.merge(day_risk, on="region_id")

# ─────────────────────────────────────────────
# 8. 메인 레이아웃: 지도 + 사이드 패널
# ─────────────────────────────────────────────
col_map, col_panel = st.columns([2.5, 1])

with col_map:
    st.subheader(f"위험 분포 — {pred_date.strftime('%Y-%m-%d')} ({day_offset}일 후)")

    m = folium.Map(location=[33.36, 126.55], zoom_start=10, tiles="CartoDB Positron")

    if layer_mode == "4단계 경보":
        def style_fn(feat):
            return {
                "fillColor": TIER_COLORS[feat["properties"]["tier"]],
                "color": "white", "weight": 1, "fillOpacity": 0.72,
            }
    elif layer_mode == "예측 발생률 λ":
        max_l = day_risk["lambda"].max()
        def style_fn(feat):
            r = feat["properties"]["lambda"] / max_l
            # 흰색 → 빨강 그라데이션
            return {
                "fillColor": f"#{255:02x}{int(255*(1-r)):02x}{int(255*(1-r)):02x}",
                "color": "white", "weight": 1, "fillOpacity": 0.6,
            }
    else:  # 진드기 밀도
        max_d = float(np.max(tick_density))
        def style_fn(feat):
            r = feat["properties"]["tick_density"] / max_d
            return {
                "fillColor": f"#{int(120+135*r):02x}{int(70*(1-r)):02x}{int(50*(1-r)):02x}",
                "color": "white", "weight": 1, "fillOpacity": 0.6,
            }

    # 숫자 포맷
    gdf_today["lambda_fmt"] = gdf_today["lambda"].round(3)
    gdf_today["tick_density_fmt"] = gdf_today["tick_density"].round(2)

    # JSON 직렬화: datetime.date 등 직렬화 불가 컬럼 제거 + Timestamp는 ISO 문자열로
    gdf_for_map = gdf_today.drop(columns=["date"], errors="ignore").copy()
    for col in gdf_for_map.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns:
        gdf_for_map[col] = gdf_for_map[col].astype(str)

    folium.GeoJson(
        json.loads(gdf_for_map.to_json()),
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["region_name", "tier", "lambda_fmt", "ref_temperature", "ref_humidity"],
            aliases=["지역", "경보 단계", "예측 λ", "기상 기온(°C)", "기상 습도(%)"],
        ),
    ).add_to(m)

    # 범례
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 20px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.12); font-size: 13px;">
        <b style="font-size:12px; color:#555">4단계 경보</b><br>
        {''.join(f'<span style="display:inline-block; width:10px; height:10px; '
                 f'background:{TIER_COLORS[t]}; border-radius:2px; margin-right:6px;"></span>{t}<br>'
                 for t in TIER_ORDER)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width="100%", height=520, returned_objects=[])

with col_panel:
    st.subheader("단계별 지역")

    counts = day_risk["tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
    for t in reversed(TIER_ORDER):
        c = int(counts[t])
        st.markdown(
            f"<div style='padding:8px 12px; margin-bottom:6px; "
            f"border-left:4px solid {TIER_COLORS[t]}; background:#f8fafc; "
            f"border-radius:4px;'>"
            f"<b style='color:{TIER_COLORS[t]}'>{t}</b> "
            f"&nbsp;<span style='color:#475569'>{c}개 지역</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("최고 위험 5개 지역")
    top5 = (
        day_risk.nlargest(5, "lambda")[["region_name", "tier", "lambda"]]
        .rename(columns={"region_name": "지역", "tier": "단계", "lambda": "λ"})
    )
    top5["λ"] = top5["λ"].round(3)
    st.dataframe(top5, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────
# 9. 14일 시계열: 위험 상위 지역 추이
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("14일 위험 추이 (상위 8개 지역)")

top_regions = (
    risk_df.groupby("region_name")["lambda"].max().nlargest(8).index.tolist()
)
trend_df = risk_df[risk_df.region_name.isin(top_regions)].copy()

base_chart = alt.Chart(trend_df).encode(
    x=alt.X("date:T", title="예보일"),
    y=alt.Y("lambda:Q", title="예측 발생률 λ"),
    color=alt.Color("region_name:N", title="지역"),
    tooltip=["region_name", "date:T", "tier", alt.Tooltip("lambda:Q", format=".3f")],
)

line = base_chart.mark_line(point=True, size=2)

threshold_df = pd.DataFrame({
    "threshold": quantiles,
    "label": ["관심/주의", "주의/경고", "경고/위험"],
})
rules = alt.Chart(threshold_df).mark_rule(
    strokeDash=[5, 4], color="#94a3b8", size=1
).encode(y="threshold:Q")

# 선택일 강조
selected_date = pd.Timestamp(pred_date)
vline = alt.Chart(pd.DataFrame({"x": [selected_date]})).mark_rule(
    color="#dc2626", size=2, strokeDash=[3, 3]
).encode(x="x:T")

st.altair_chart(
    (line + rules + vline).properties(height=320),
    use_container_width=True,
)

# ─────────────────────────────────────────────
# 10. 모델 설명 (확장 가능)
# ─────────────────────────────────────────────
with st.expander("📐 시공간 모델 수식 및 실데이터 연동 가이드"):
    st.markdown(r"""
    **두 단계 시공간 위험 모델**

    $$
    \log \lambda(s, t) = \beta_0 + \beta_T \cdot T(s, t-\ell) + \beta_{RH} \cdot RH(s, t-\ell) + \log D(s)
    $$

    | 변수 | 의미 | 출처 |
    |---|---|---|
    | $D(s)$ | 지역 $s$의 진드기 상대 밀도 | GAM 공간 모델 (김미선 교수님) |
    | $T(s, t-\ell)$, $RH(s, t-\ell)$ | $\ell$일 전 기상 | 기상청 2주 예보 API |
    | $\ell$ | 지연일 (진드기 활성화 + 잠복기) | 7–14일 (조정 가능) |
    | $\beta_0, \beta_T, \beta_{RH}$ | 회귀계수 | 12년 임상 데이터 학습 |

    ### 실데이터 연동 단계

    1. **`create_jeju_grid()`** → 행정안전부/SGIS 읍·면·동 GeoJSON으로 교체
       ```python
       gdf = gpd.read_file("jeju_eupmyeondong.geojson")
       ```

    2. **`get_tick_density()`** → 김미선 교수님 GAM 모델 예측값 CSV
       ```python
       df = pd.read_csv("gam_tick_density_predictions.csv")
       ```

    3. **`get_kma_forecast()`** → 기상청 API
       - 단기예보 (3일, 5km 격자): `getVilageFcst`
       - 중기예보 (10일, 시·군): `getMidTa`, `getMidLandFcst`
       - 두 API를 14일로 결합하여 격자별 일 단위 시계열 생성

    4. **`compute_risk()` 회귀계수** → 임상 데이터로 재학습
       ```python
       import statsmodels.api as sm
       model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
       ```

    ### 향후 고도화 옵션
    - **불확실성 정량화**: Bayesian (PyMC, INLA) → 경보 단계의 신뢰구간
    - **공간 자기상관**: BYM2 또는 SAR 모델 → 인접 지역 정보 공유
    - **계절성**: AR(1) 또는 Fourier 항 추가
    - **검증**: 2026–2027 실제 환자 데이터와 hindcast 비교 (이미 계획 중)
    """)
