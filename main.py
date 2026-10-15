import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
BASE_END_YEAR = 2025   # 수업 기준 기간의 끝
MIN_DAYS = 300         # 관측일 최소 기준

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")


@st.cache_data
def load_yearly():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    # 2025년까지만 사용
    df = df[df["연도"] <= BASE_END_YEAR]
    # 연도별 평균과 관측일 수
    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    # 관측일이 300일 미만인 해 제외
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return yearly


st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온 데이터로 만든 회귀 직선으로 해마다의 기온을 예상해 봐요.")

try:
    yearly = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(yearly) < 2:
    st.warning("회귀 직선을 만들 데이터가 충분하지 않아요.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 직선을 만든 데이터 정보
c1, c2, c3 = st.columns(3)
c1.metric("사용한 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
st.write(
    f"직선은 **{start_year}년부터 {end_year}년까지, 총 {n_years}개 해**의 "
    f"연평균기온으로 만들었어요. (2025년 이후 데이터와 관측일이 {MIN_DAYS}일 미만인 해는 제외)"
)

# 연도 선택 슬라이더
year = st.slider("연도를 골라 보세요", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">{year}년의 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.1;">{pred:.1f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 산점도 + 회귀 직선
line_x = np.array([1900, 2100], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7),
        hovertemplate="%{x}년<br>%{y:.2f}°C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=2),
        hovertemplate="%{x}년<br>%{y:.2f}°C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=14, symbol="star", line=dict(width=1, color="white")),
        hovertemplate="%{x}년<br>예상 %{y:.2f}°C<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(range=[1900, 2100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=30, b=10),
    height=480,
)
st.plotly_chart(fig, use_container_width=True)

# 상관계수와 회귀식
m1, m2 = st.columns(2)
m1.metric("상관계수 (r)", f"{r:.3f}")
m2.metric("연간 변화량(기울기)", f"{slope:+.4f} °C/년")
st.write(f"회귀식: **기온 = {slope:.4f} × 연도 {intercept:+.2f}**")

st.caption(
    "※ 직선을 데이터 범위 밖으로 늘린 값은 단순한 추세 연장일 뿐, 실제 미래 기온을 보장하지 않아요."
)

with st.expander("연도별 데이터 보기"):
    st.dataframe(
        yearly.rename(columns={"연평균기온": "연평균기온(°C)"}).round(2),
        use_container_width=True,
        hide_index=True,
    )
