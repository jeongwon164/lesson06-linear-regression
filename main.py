
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

기준연도 = 2025
기준연수 = 1908

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 바탕으로 "
    "연도별 기온의 추세를 살펴보고 예상 기온을 계산합니다."
)


# -----------------------------
# 데이터 불러오기 및 정리
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜와 기온 데이터 정리
    df["날짜"] = pd.to_datetime(
        df["날짜"], errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 행 제외
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 실제 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 2025년까지이며 관측일이 300일 이상인 해만 사용
    yearly = yearly[
        (yearly["연도"] <= 기준연도)
        & (yearly["관측일수"] >= 300)
    ].copy()

    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


# -----------------------------
# 데이터 불러오기
# -----------------------------
try:
    yearly = load_data()
except Exception as e:
    st.error(f"기온 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 계산하기에 유효한 연도가 부족합니다.")
    st.stop()


# -----------------------------
# 선형 회귀 계산
# -----------------------------
# 1908년을 기준으로 지난 연수를 독립 변수로 사용
x = yearly["연도"].to_numpy(dtype=float) - 기준연수
y = yearly["연평균기온"].to_numpy(dtype=float)

# 최소제곱법으로 직선 계산: y = a*x + b
a, b = np.polyfit(x, y, 1)

# 피어슨 상관계수
상관계수 = np.corrcoef(x, y)[0, 1]

# 회귀에 사용한 연도 정보
연도개수 = len(yearly)
시작연도 = int(yearly["연도"].min())
끝연도 = int(yearly["연도"].max())


# -----------------------------
# 회귀 분석 결과
# -----------------------------
st.subheader("📊 회귀 분석 결과")

col1, col2, col3, col4 = st.columns(4)

col1.metric("회귀에 사용한 연도 수", f"{연도개수}년")
col2.metric("시작 연도", f"{시작연도}년")
col3.metric("끝 연도", f"{끝연도}년")
col4.metric("상관계수", f"{상관계수:.3f}")

st.caption(
    "상관계수는 연도와 연평균기온 사이의 선형 관계를 나타냅니다. "
    "1에 가까우면 양의 관계, -1에 가까우면 음의 관계이며, "
    "0에 가까우면 선형 관계가 약합니다."
)

st.write(
    f"**회귀 직선:** 연평균기온 = "
    f"{a:.5f} × (연도 − {기준연수}) + {b:.3f}"
)

st.write(
    f"**연간 추세:** 연도가 1년 증가할 때 "
    f"회귀 직선의 예상 기온은 {a:+.4f}℃ 변합니다."
)


# -----------------------------
# 산점도와 회귀 직선
# -----------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선
line_x = np.array([시작연도, 끝연도], dtype=float)
line_y = a * (line_x - 기준연수) + b

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="선형 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    margin=dict(l=20, r=20, t=30, b=20)
)

fig.update_xaxes(dtick=10)
fig.update_yaxes(showgrid=True, zeroline=False)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "점 하나는 해당 연도의 연평균기온을 나타냅니다. "
    "직선은 선택된 관측 자료에 최소제곱법을 적용한 결과입니다."
)


# -----------------------------
# 연도 선택 및 예상 기온
# -----------------------------
st.subheader("🌡️ 연도별 예상 기온")

선택연도 = st.slider(
    "예상 기온을 확인할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

예상기온 = a * (선택연도 - 기준연수) + b

st.metric(
    label=f"{선택연도}년 예상 연평균기온",
    value=f"{예상기온:.2f}℃"
)

if 선택연도 < 시작연도 or 선택연도 > 끝연도:
    st.warning(
        f"{선택한 연도는 회귀에 사용한 기간 "
        f"{시작연도}~{끝연도}년 밖에 있습니다. "
        "이 값은 회귀 직선을 연장한 외삽값이며, "
        "실제 기온을 보장하지 않습니다."
    )
else:
    st.info(
        "선택한 연도가 회귀에 사용한 기간 안에 있습니다. "
        "예상 기온은 해당 연도의 실제 관측값이 아니라 "
        "회귀 직선에서 계산한 값입니다."
    )


# -----------------------------
# 사용한 연도별 데이터 확인
# -----------------------------
with st.expander("회귀에 사용한 연도별 데이터 보기"):
    표시데이터 = yearly.copy()
    표시데이터["연평균기온"] = 표시데이터["연평균기온"].round(2)
    st.dataframe(표시데이터, use_container_width=True, hide_index=True)
