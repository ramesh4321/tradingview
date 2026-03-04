import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import numpy as np

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradingView Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
.stApp { background-color: #0d1117; }

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #8b949e !important;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 4px 0;
}
.metric-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #e6edf3; font-size: 22px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-delta-pos { color: #3fb950; font-size: 13px; font-family: 'JetBrains Mono', monospace; }
.metric-delta-neg { color: #f85149; font-size: 13px; font-family: 'JetBrains Mono', monospace; }

.tv-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 28px;
    color: #58a6ff;
    letter-spacing: -1px;
}

.ha-badge {
    background: #1f2d3d; color: #58a6ff; border: 1px solid #58a6ff;
    padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;
    letter-spacing: 1px; margin-left: 8px;
}

.signal-buy {
    background: #1a3a2a; color: #3fb950; border: 1px solid #3fb950;
    padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 1px;
}
.signal-sell {
    background: #3a1a1a; color: #f85149; border: 1px solid #f85149;
    padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 1px;
}
.signal-neutral {
    background: #1f2937; color: #8b949e; border: 1px solid #8b949e;
    padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 1px;
}

.stDataFrame { background: #161b22; }
div[data-testid="stMetricValue"] { color: #e6edf3 !important; }
button[data-baseweb="tab"] { color: #8b949e !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
INTRADAY = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","4h":"4h"}
DAILY    = {"1D":"1d","3D":"3d","5D":"5d"}
WEEKLY   = {"1W":"1wk","2W":"2wk"}
MONTHLY  = {"1M":"1mo"}

# Map display label -> yfinance interval code
TF_LABEL_TO_CODE = {**INTRADAY, **DAILY, **WEEKLY, **MONTHLY}

PERIODS = {
    "1 Day":"1d","5 Days":"5d","1 Month":"1mo","3 Months":"3mo",
    "6 Months":"6mo","1 Year":"1y","2 Years":"2y","5 Years":"5y","Max":"max",
}

POPULAR = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BTC-USD","ETH-USD","SPY","QQQ"]
CHART_TYPES = ["Candlestick", "Heikin-Ashi", "Line"]


# ── Data Helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Convert standard OHLC to Heikin-Ashi candles."""
    ha = df.copy()
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2)

    ha["HA_Open"] = ha_open
    ha["HA_High"] = ha[["High", "HA_Open", "HA_Close"]].max(axis=1)
    ha["HA_Low"]  = ha[["Low",  "HA_Open", "HA_Close"]].min(axis=1)
    return ha


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    df["EMA20"]  = ta.trend.ema_indicator(close, window=20)
    df["EMA50"]  = ta.trend.ema_indicator(close, window=50)
    df["EMA200"] = ta.trend.ema_indicator(close, window=200)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_mid"]   = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()

    macd_obj = ta.trend.MACD(close)
    df["MACD"]        = macd_obj.macd()
    df["MACD_signal"] = macd_obj.macd_signal()
    df["MACD_hist"]   = macd_obj.macd_diff()

    df["RSI"]       = ta.momentum.rsi(close, window=14)
    df["Volume_MA"] = volume.rolling(20).mean()
    df["ATR"]       = ta.volatility.average_true_range(high, low, close, window=14)
    return df


def compute_signal(df: pd.DataFrame):
    last  = df.iloc[-1]
    score = 0
    sigs  = []

    rsi = float(last.get("RSI", 50))
    if rsi < 30:   score += 2; sigs.append("✅ RSI Oversold (<30)")
    elif rsi > 70: score -= 2; sigs.append("🔴 RSI Overbought (>70)")
    else:                      sigs.append(f"➖ RSI Neutral ({rsi:.1f})")

    close = float(last["Close"])
    ema20 = float(last.get("EMA20", close))
    ema50 = float(last.get("EMA50", close))
    if close > ema20 > ema50:
        score += 2; sigs.append("✅ EMA Bullish Stack (Price > EMA20 > EMA50)")
    elif close < ema20 < ema50:
        score -= 2; sigs.append("🔴 EMA Bearish Stack (Price < EMA20 < EMA50)")
    else:
        sigs.append("➖ EMA Mixed")

    macd     = float(last.get("MACD", 0))
    macd_sig = float(last.get("MACD_signal", 0))
    if macd > macd_sig: score += 1; sigs.append("✅ MACD above Signal Line")
    else:               score -= 1; sigs.append("🔴 MACD below Signal Line")

    verdict = "BUY" if score >= 3 else "SELL" if score <= -3 else "NEUTRAL"
    return verdict, score, sigs


def signal_badge(verdict):
    cls = {"BUY":"signal-buy","SELL":"signal-sell","NEUTRAL":"signal-neutral"}[verdict]
    return f'<span class="{cls}">{verdict}</span>'


# ── Chart Builder ─────────────────────────────────────────────────────────────
def build_chart(df, ticker, show_ema, show_bb, show_volume, chart_type):
    rows    = 3 if show_volume else 2
    heights = [0.55, 0.25, 0.20] if show_volume else [0.65, 0.35]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
    )

    # ── Candle / Line ──
    if chart_type == "Heikin-Ashi":
        ha = compute_heikin_ashi(df)
        fig.add_trace(go.Candlestick(
            x=ha.index,
            open=ha["HA_Open"], high=ha["HA_High"],
            low=ha["HA_Low"],   close=ha["HA_Close"],
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950",  decreasing_fillcolor="#f85149",
            name=f"{ticker} HA", line=dict(width=1),
        ), row=1, col=1)

    elif chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950",  decreasing_fillcolor="#f85149",
            name=ticker, line=dict(width=1),
        ), row=1, col=1)

    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"].squeeze(),
            line=dict(color="#58a6ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
            name=ticker,
        ), row=1, col=1)

    # ── EMAs ──
    if show_ema:
        for col, color, label in [
            ("EMA20", "#f0883e", "EMA 20"),
            ("EMA50", "#bc8cff", "EMA 50"),
            ("EMA200","#ff7b72", "EMA 200"),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    line=dict(color=color, width=1),
                    name=label, opacity=0.9,
                ), row=1, col=1)

    # ── Bollinger Bands ──
    if show_bb and "BB_upper" in df.columns:
        for col, color, lbl in [
            ("BB_upper","rgba(139,148,158,0.7)","BB Upper"),
            ("BB_mid",  "rgba(139,148,158,0.4)","BB Mid"),
            ("BB_lower","rgba(139,148,158,0.7)","BB Lower"),
        ]:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                line=dict(color=color, width=1, dash="dot"),
                name=lbl, opacity=0.8,
            ), row=1, col=1)

    # ── MACD ──
    if "MACD" in df.columns:
        hist_colors = ["#3fb950" if v >= 0 else "#f85149" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=hist_colors,
                              name="MACD Hist", opacity=0.7), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
                                  line=dict(color="#58a6ff", width=1), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"],
                                  line=dict(color="#f0883e", width=1), name="Signal"), row=2, col=1)

    # ── Volume ──
    if show_volume and rows == 3:
        volume     = df["Volume"].squeeze()
        vol_colors = ["#3fb950" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#f85149"
                      for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=volume, marker_color=vol_colors,
                              name="Volume", opacity=0.65), row=3, col=1)
        if "Volume_MA" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["Volume_MA"],
                                      line=dict(color="#bc8cff", width=1.2),
                                      name="Vol MA20"), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="JetBrains Mono"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11),
                    orientation="h", y=1.02),
        xaxis_rangeslider_visible=False,
        height=680,
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor="#21262d", showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", showgrid=True, row=i, col=1,
                          tickfont=dict(size=10))

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="tv-title">📈 TradingView</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#8b949e;font-size:13px">Python · Streamlit Dashboard</div>', unsafe_allow_html=True)
    st.divider()

    ticker_input = st.text_input("Symbol", value="AAPL", placeholder="AAPL, BTC-USD …").upper().strip()
    if not ticker_input:
        ticker_input = "AAPL"

    st.markdown("**Quick Select**")
    qcols = st.columns(3)
    for idx, sym in enumerate(POPULAR[:9]):
        if qcols[idx % 3].button(sym, use_container_width=True, key=f"qs_{sym}"):
            ticker_input = sym

    st.divider()
    period_label = st.selectbox("Period", list(PERIODS.keys()), index=5)

    st.divider()
    st.markdown("**Chart Type**")
    chart_type = st.radio("", CHART_TYPES, horizontal=False, label_visibility="collapsed")
    if chart_type == "Heikin-Ashi":
        st.markdown("🕯️ *Smoothed trend candles. Indicators still use real OHLC.*")

    st.divider()
    show_ema    = st.checkbox("EMA Lines (20 / 50 / 200)", value=True)
    show_bb     = st.checkbox("Bollinger Bands", value=False)
    show_volume = st.checkbox("Volume Panel", value=True)

    st.divider()
    st.caption("Data via Yahoo Finance · Cache: 60 s")


period = PERIODS[period_label]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
header_col, _ = st.columns([3, 1])
with header_col:
    st.markdown(f'<div class="tv-title">📈 {ticker_input}</div>', unsafe_allow_html=True)

# ── Timeframe Toolbar ─────────────────────────────────────────────────────────
if "sel_tf" not in st.session_state:
    st.session_state.sel_tf = "1D"   # default: 1-Day candles

st.markdown("##### ⏱ Timeframe")

# ─ Intraday row ─
st.markdown('<span style="color:#8b949e;font-size:10px;text-transform:uppercase;letter-spacing:1px">Intraday</span>', unsafe_allow_html=True)
intra_keys = list(INTRADAY.keys())   # ["1m","5m","15m","30m","1h","4h"]
intra_cols = st.columns(len(intra_keys))
for i, lbl in enumerate(intra_keys):
    is_active = st.session_state.sel_tf == lbl
    if intra_cols[i].button(lbl, key=f"tf_{lbl}",
                             type="primary" if is_active else "secondary",
                             use_container_width=True):
        st.session_state.sel_tf = lbl

# ─ Daily row ─
st.markdown('<span style="color:#8b949e;font-size:10px;text-transform:uppercase;letter-spacing:1px">Daily</span>', unsafe_allow_html=True)
daily_keys = list(DAILY.keys())      # ["1D","3D","5D"]
daily_cols = st.columns(len(daily_keys))
for i, lbl in enumerate(daily_keys):
    is_active = st.session_state.sel_tf == lbl
    if daily_cols[i].button(lbl, key=f"tf_{lbl}",
                              type="primary" if is_active else "secondary",
                              use_container_width=True):
        st.session_state.sel_tf = lbl

# ─ Weekly / Monthly row ─
st.markdown('<span style="color:#8b949e;font-size:10px;text-transform:uppercase;letter-spacing:1px">Weekly · Monthly</span>', unsafe_allow_html=True)
wm_keys = list(WEEKLY.keys()) + list(MONTHLY.keys())   # ["1W","2W","1M"]
wm_cols = st.columns(len(wm_keys))
for i, lbl in enumerate(wm_keys):
    is_active = st.session_state.sel_tf == lbl
    if wm_cols[i].button(lbl, key=f"tf_{lbl}",
                          type="primary" if is_active else "secondary",
                          use_container_width=True):
        st.session_state.sel_tf = lbl

interval = TF_LABEL_TO_CODE[st.session_state.sel_tf]

st.markdown("---")

# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker_input} · {st.session_state.sel_tf} · {period_label}…"):
    df = fetch_data(ticker_input, period, interval)

if df.empty:
    st.error(
        f"❌ No data for **{ticker_input}** with interval `{interval}` / period `{period}`. "
        "Try a longer period or a different interval (e.g. intraday needs a short period)."
    )
    st.stop()

df = add_indicators(df)
verdict, score, signals = compute_signal(df)

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change     = last_close - prev_close
pct_change = (change / prev_close) * 100 if prev_close else 0

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Last Price</div>
      <div class="metric-value">${last_close:,.2f}</div>
      <div class="{'metric-delta-pos' if change>=0 else 'metric-delta-neg'}">
        {'▲' if change>=0 else '▼'} {abs(change):.2f} ({abs(pct_change):.2f}%)
      </div>
    </div>""", unsafe_allow_html=True)

with c2:
    extra = ' <span class="ha-badge">HA</span>' if chart_type == "Heikin-Ashi" else ""
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Signal{extra}</div>
      <div style="margin-top:8px">{signal_badge(verdict)}</div>
      <div class="metric-delta-pos" style="margin-top:6px">Score: {score:+d}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    hi = float(df["High"].max())
    lo = float(df["Low"].min())
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Period High / Low</div>
      <div class="metric-value" style="color:#3fb950">${hi:,.2f}</div>
      <div class="metric-delta-neg">${lo:,.2f}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    rsi_val   = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 0
    rsi_color = "#f85149" if rsi_val > 70 else ("#3fb950" if rsi_val < 30 else "#e6edf3")
    rsi_lbl   = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">RSI (14)</div>
      <div class="metric-value" style="color:{rsi_color}">{rsi_val:.1f}</div>
      <div class="metric-label">{rsi_lbl}</div>
    </div>""", unsafe_allow_html=True)

with c5:
    vol     = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
    vol_str = f"{vol/1e6:.2f}M" if vol >= 1_000_000 else f"{vol/1e3:.1f}K"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Volume</div>
      <div class="metric-value">{vol_str}</div>
      <div class="metric-label">Last candle</div>
    </div>""", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
st.plotly_chart(
    build_chart(df, ticker_input, show_ema, show_bb, show_volume, chart_type),
    use_container_width=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Indicators", "🕯️ Heikin-Ashi Data", "📋 OHLCV", "🔍 Signal"])

with tab1:
    tc = st.columns(3)
    indicators = {
        "EMA 20":   f"${float(df['EMA20'].iloc[-1]):,.2f}"      if "EMA20"       in df.columns else "N/A",
        "EMA 50":   f"${float(df['EMA50'].iloc[-1]):,.2f}"      if "EMA50"       in df.columns else "N/A",
        "EMA 200":  f"${float(df['EMA200'].iloc[-1]):,.2f}"     if "EMA200"      in df.columns else "N/A",
        "BB Upper": f"${float(df['BB_upper'].iloc[-1]):,.2f}"   if "BB_upper"    in df.columns else "N/A",
        "BB Lower": f"${float(df['BB_lower'].iloc[-1]):,.2f}"   if "BB_lower"    in df.columns else "N/A",
        "ATR (14)": f"${float(df['ATR'].iloc[-1]):,.2f}"        if "ATR"         in df.columns else "N/A",
        "MACD":     f"{float(df['MACD'].iloc[-1]):,.4f}"        if "MACD"        in df.columns else "N/A",
        "MACD Sig": f"{float(df['MACD_signal'].iloc[-1]):,.4f}" if "MACD_signal" in df.columns else "N/A",
        "RSI (14)": f"{float(df['RSI'].iloc[-1]):,.2f}"         if "RSI"         in df.columns else "N/A",
    }
    for idx, (k, v) in enumerate(indicators.items()):
        tc[idx % 3].metric(k, v)

with tab2:
    ha_df = compute_heikin_ashi(df)
    ha_display = ha_df[["HA_Open","HA_High","HA_Low","HA_Close"]].tail(50).sort_index(ascending=False).round(4)
    ha_display.columns = ["HA Open","HA High","HA Low","HA Close"]
    st.caption("🕯️ Heikin-Ashi candles remove noise and emphasise trend direction. Formula: HA_Close=(O+H+L+C)/4 · HA_Open=(prev_HA_Open+prev_HA_Close)/2")
    st.dataframe(ha_display, use_container_width=True, height=380)

with tab3:
    ohlcv = df[["Open","High","Low","Close","Volume"]].tail(50).sort_index(ascending=False).round(4)
    st.dataframe(ohlcv, use_container_width=True, height=380)

with tab4:
    st.markdown("#### Signal Breakdown")
    for s in signals:
        st.markdown(f"- {s}")
    st.markdown(f"**Overall Score:** `{score:+d}` → **{verdict}**")
    st.info("Score ≥ +3 = BUY · Score ≤ -3 = SELL · Otherwise NEUTRAL")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚡ Python · Streamlit · Plotly · yfinance · ta  |  Data: Yahoo Finance  |  Not financial advice")
