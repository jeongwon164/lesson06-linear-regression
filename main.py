
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

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온을 분석하고, "
    "장기 추세와 최근 추세를 비교합니다."
)


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 2025년까지이며 유효 관측일이 300일 이상인 해만 사용
    yearly = yearly[
        (yearly["연도"] <= 기준연도)
        & (yearly["관측일수"] >= 300)
    ].copy()

    return yearly.sort_values("연도").reset_index(drop=True)


try:
    yearly = load_data()
except Exception as e:
    st.error(f"기온 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 계산하기에 유효한 연도가 부족합니다.")
    st.stop()


# -----------------------------
# 전체 기간 회귀 분석
# -----------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

전체기울기, 전체절편 = np.polyfit(x, y, 1)
상관계수 = np.corrcoef(x, y)[0, 1]

연도개수 = len(yearly)
시작연도 = int(yearly["연도"].min())
끝연도 = int(yearly["연도"].max())

# 연간 기울기를 100년 기준으로 환산
전체_100년변화 = 전체기울기 * 100


# -----------------------------
# 최근 20년 회귀 분석
# -----------------------------
# 가장 최근의 유효 관측 연도를 기준으로 20년 범위 설정
최근시작연도 = 끝연도 - 19

최근20년 = yearly[
    (yearly["연도"] >= 최근시작연도)
    & (yearly["연도"] <= 끝연도)
].copy()

if len(최근20년) >= 2:
    최근x = 최근20년["연도"].to_numpy(dtype=float)
    최근y = 최근20년["연평균기온"].to_numpy(dtype=float)

    최근기울기, 최근절편 = np.polyfit(최근x, 최근y, 1)
    최근_100년변화 = 최근기울기 * 100
    최근상관계수 = np.corrcoef(최근x, 최근y)[0, 1]
    최근계산가능 = True
else:
    최근기울기 = np.nan
    최근절편 = np.nan
    최근_100년변화 = np.nan
    최근상관계수 = np.nan
    최근계산가능 = False


# -----------------------------
# 핵심 결과: 100년 기준 기온 변화
# -----------------------------
st.subheader("🌡️ 100년에 기온이 몇 ℃ 변할까?")

st.caption(
    "회귀 직선의 기울기를 100년 기준으로 환산했습니다. "
    "양수는 상승 추세, 음수는 하락 추세를 뜻합니다."
)

왼쪽, 오른쪽 = st.columns(2)

with 왼쪽:
    st.markdown("### 전체 관측 기간")
    st.metric(
        label=f"{시작연도}~{끝연도}년",
        value=f"{전체_100년변화:+.2f}℃",
        help="조건을 만족하는 전체 연평균기온 자료로 계산한 100년 기준 변화량"
    )
    st.write(f"분석에 사용한 연도: **{연도개수}개**")
    st.write(f"상관계수: **{상관계수:.3f}**")

with 오른쪽:
    st.markdown("### 최근 20년 범위")
    st.caption(f"{최근시작연도}~{끝연도}년")

    if 최근계산가능:
        st.metric(
            label="최근 20년 자료로 계산",
            value=f"{최근_100년변화:+.2f}℃",
            help="최근 20년 범위 안에서 관측일 기준을 통과한 연도만 사용"
        )
        st.write(f"분석에 사용한 연도: **{len(최근20년)}개**")
        st.write(f"상관계수: **{최근상관계수:.3f}**")
    else:
        st.warning("최근 20년 범위에 유효한 연도가 2개 미만이라 기울기를 계산할 수 없습니다.")

if 최근계산가능:
    차이 = 최근_100년변화 - 전체_100년변화
    st.info(
        f"최근 20년의 100년 기준 변화량은 전체 기간보다 "
        f"{차이:+.2f}℃ 차이 납니다. "
        "이는 두 기간의 회귀 직선 기울기 차이를 비교한 값이며, "
        "실제 100년 동안 관측된 변화량은 아닙니다."
    )

st.divider()


# -----------------------------
# 회귀 분석 정보
# -----------------------------
st.subheader("📊 전체 기간 분석 정보")

c1, c2, c3 = st.columns(3)
c1.metric("회귀에 사용한 연도 수", f"{연도개수}년")
c2.metric("시작 연도", f"{시작연도}년")
c3.metric("끝 연도", f"{끝연도}년")

st.write(
    f"전체 기간 회귀 직선: "
    f"연평균기온 = {전체기울기:.5f} × 연도 + ({전체절편:.3f})"
)

st.caption(
    "상관계수는 연도와 연평균기온 사이의 선형 관계를 나타냅니다. "
    "상관관계가 있다고 해서 연도가 기온 변화의 유일한 원인이라는 뜻은 아닙니다."
)


# -----------------------------
# 산점도와 회귀 직선
# -----------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

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

# 전체 기간 회귀 직선
line_x = np.array([시작연도, 끝연도], dtype=float)
line_y = 전체기울기 * line_x + 전체절편

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 최근 20년 회귀 직선
if 최근계산가능:
    최근line_x = np.array(
        [최근20년["연도"].min(), 최근20년["연도"].max()],
        dtype=float
    )
    최근line_y = 최근기울기 * 최근line_x + 최근절편

    fig.add_trace(
        go.Scatter(
            x=최근line_x,
            y=최근line_y,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(width=3, dash="dash"),
            hovertemplate=(
                "연도: %{x:.0f}년<br>"
                "최근 20년 회귀 예상 기온: %{y:.2f}℃"
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
    margin=dict(l=20, r=20, t=40, b=20)
)

fig.update_xaxes(dtick=10)
fig.update_yaxes(showgrid=True, zeroline=False)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "실선은 전체 기간 회귀 직선, 점선은 최근 20년 범위의 회귀 직선입니다. "
    "점은 실제 연평균기온입니다."
)


# -----------------------------
# 연도 선택 및 예상 기온
# -----------------------------
st.subheader("🔮 연도별 예상 기온")

선택연도 = st.slider(
    "예상 기온을 확인할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

예상기온 = 전체기울기 * 선택연도 + 전체절편

st.metric(
    label=f"{선택연도}년 예상 연평균기온",
    value=f"{예상기온:.2f}℃"
)

if 선택연도 < 시작연도 or 선택연도 > 끝연도:
    st.warning(
        f"선택한 연도는 회귀에 사용한 기간 "
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
# 연도별 데이터 확인
# -----------------------------
with st.expander("회귀에 사용한 연도별 데이터 보기"):
    표시데이터 = yearly.copy()
    표시데이터["연평균기온"] = 표시데이터["연평균기온"].round(2)

    st.dataframe(
        표시데이터,
        use_container_width=True,
        hide_index=True
    )
