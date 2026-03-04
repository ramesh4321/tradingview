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

# ── CSS ───────────────────────────────────────────────────────────────────────
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
section[data-testid="stSidebar"] label {
    color: #8b949e !important;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Metric cards ── */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 4px 0;
}
.metric-label { color: #8b949e; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-bottom:3px; }
.metric-value { color: #e6edf3; font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-delta-pos { color: #3fb950; font-size: 12px; font-family: 'JetBrains Mono', monospace; }
.metric-delta-neg { color: #f85149; font-size: 12px; font-family: 'JetBrains Mono', monospace; }

/* ── Title ── */
.tv-title { font-family:'Syne',sans-serif; font-weight:800; font-size:26px; color:#58a6ff; letter-spacing:-0.5px; }

/* ── Toolbar container ── */
.toolbar-container {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 12px;
}
.toolbar-group-label {
    color: #484f58;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.toolbar-divider {
    height: 1px;
    background: #21262d;
    margin: 10px 0;
}

/* ── st.pills styling override ── */
div[data-testid="stPills"] {
    gap: 4px !important;
}
div[data-testid="stPills"] button {
    background: #21262d !important;
    color: #8b949e !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 4px 10px !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stPills"] button:hover {
    background: #2d333b !important;
    color: #e6edf3 !important;
    border-color: #484f58 !important;
}
div[data-testid="stPills"] button[aria-pressed="true"],
div[data-testid="stPills"] button[aria-selected="true"],
div[data-testid="stPills"] button.selected {
    background: #1f3d5c !important;
    color: #58a6ff !important;
    border-color: #58a6ff !important;
}

/* ── Signal badges ── */
.signal-buy    { background:#1a3a2a; color:#3fb950; border:1px solid #3fb950; padding:3px 14px; border-radius:20px; font-size:12px; font-weight:700; }
.signal-sell   { background:#3a1a1a; color:#f85149; border:1px solid #f85149; padding:3px 14px; border-radius:20px; font-size:12px; font-weight:700; }
.signal-neutral{ background:#1f2937; color:#8b949e; border:1px solid #8b949e; padding:3px 14px; border-radius:20px; font-size:12px; font-weight:700; }
.ha-badge      { background:#1f2d3d; color:#58a6ff; border:1px solid #58a6ff; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:700; }

/* ── Active info bar ── */
.active-bar {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 6px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #8b949e;
    margin-bottom: 10px;
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
}
.active-item { display: flex; gap: 5px; align-items: center; }
.active-key  { color: #484f58; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }
.active-val  { color: #e6edf3; font-weight: 700; }

/* ── Tabs ── */
button[data-baseweb="tab"]                       { color: #8b949e !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
div[data-testid="stMetricValue"]                 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Timeframe definitions ──
TF_INTRADAY = ["1m","5m","15m","30m","1h","4h"]
TF_DAILY    = ["1D","3D","5D"]
TF_WEEKLY   = ["1W","2W"]
TF_MONTHLY  = ["1M","3M"]

# Map display label → yfinance interval code
TF_TO_INTERVAL = {
    "1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","4h":"4h",
    "1D":"1d","3D":"3d","5D":"5d",
    "1W":"1wk","2W":"2wk",
    "1M":"1mo","3M":"3mo",
}

PERIODS = {
    "1 Day":"1d","5 Days":"5d","1 Month":"1mo","3 Months":"3mo",
    "6 Months":"6mo","1 Year":"1y","2 Years":"2y","5 Years":"5y","Max":"max",
}

POPULAR = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BTC-USD","ETH-USD","SPY","QQQ"]

# Chart type options  display label → internal key
CHART_OPTIONS = {
    "🕯️ Candlestick": "Candlestick",
    "✳️ Heikin-Ashi":  "Heikin-Ashi",
    "📈 Line":         "Line",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2)
    ha["HA_Open"] = ha_open
    ha["HA_High"] = ha[["High","HA_Open","HA_Close"]].max(axis=1)
    ha["HA_Low"]  = ha[["Low", "HA_Open","HA_Close"]].min(axis=1)
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
    last = df.iloc[-1]; score = 0; sigs = []
    rsi = float(last.get("RSI", 50))
    if   rsi < 30: score += 2; sigs.append("✅ RSI Oversold (<30)")
    elif rsi > 70: score -= 2; sigs.append("🔴 RSI Overbought (>70)")
    else:                      sigs.append(f"➖ RSI Neutral ({rsi:.1f})")

    close = float(last["Close"])
    ema20 = float(last.get("EMA20", close))
    ema50 = float(last.get("EMA50", close))
    if   close > ema20 > ema50: score += 2; sigs.append("✅ EMA Bullish Stack")
    elif close < ema20 < ema50: score -= 2; sigs.append("🔴 EMA Bearish Stack")
    else:                                    sigs.append("➖ EMA Mixed")

    macd = float(last.get("MACD",0)); msig = float(last.get("MACD_signal",0))
    if macd > msig: score += 1; sigs.append("✅ MACD above Signal")
    else:           score -= 1; sigs.append("🔴 MACD below Signal")

    return ("BUY" if score >= 3 else "SELL" if score <= -3 else "NEUTRAL"), score, sigs


def signal_badge(v):
    cls = {"BUY":"signal-buy","SELL":"signal-sell","NEUTRAL":"signal-neutral"}[v]
    return f'<span class="{cls}">{v}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_chart(df, ticker, show_ema, show_bb, show_volume, chart_type):
    rows    = 3 if show_volume else 2
    heights = [0.55, 0.25, 0.20] if show_volume else [0.65, 0.35]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=heights)

    if chart_type == "Heikin-Ashi":
        ha = compute_heikin_ashi(df)
        fig.add_trace(go.Candlestick(
            x=ha.index, open=ha["HA_Open"], high=ha["HA_High"],
            low=ha["HA_Low"], close=ha["HA_Close"],
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950",  decreasing_fillcolor="#f85149",
            name=f"{ticker} HA", line=dict(width=1),
        ), row=1, col=1)
    elif chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950",  decreasing_fillcolor="#f85149",
            name=ticker, line=dict(width=1),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"].squeeze(),
            line=dict(color="#58a6ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.07)", name=ticker,
        ), row=1, col=1)

    if show_ema:
        for col, color, lbl in [("EMA20","#f0883e","EMA 20"),("EMA50","#bc8cff","EMA 50"),("EMA200","#ff7b72","EMA 200")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col],
                    line=dict(color=color, width=1), name=lbl, opacity=0.9), row=1, col=1)

    if show_bb and "BB_upper" in df.columns:
        for col, color, lbl in [("BB_upper","rgba(139,148,158,0.7)","BB Upper"),
                                  ("BB_mid",  "rgba(139,148,158,0.4)","BB Mid"),
                                  ("BB_lower","rgba(139,148,158,0.7)","BB Lower")]:
            fig.add_trace(go.Scatter(x=df.index, y=df[col],
                line=dict(color=color, width=1, dash="dot"), name=lbl, opacity=0.8), row=1, col=1)

    if "MACD" in df.columns:
        hc = ["#3fb950" if v >= 0 else "#f85149" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=hc, name="Hist", opacity=0.7), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#58a6ff", width=1), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], line=dict(color="#f0883e", width=1), name="Signal"), row=2, col=1)

    if show_volume and rows == 3:
        vc = ["#3fb950" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#f85149" for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"].squeeze(), marker_color=vc, name="Volume", opacity=0.65), row=3, col=1)
        if "Volume_MA" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["Volume_MA"],
                                      line=dict(color="#bc8cff", width=1.2), name="Vol MA20"), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="JetBrains Mono"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h", y=1.02),
        xaxis_rangeslider_visible=False,
        height=660,
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor="#21262d", showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", showgrid=True, row=i, col=1, tickfont=dict(size=10))

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
    qc = st.columns(3)
    for i, sym in enumerate(POPULAR[:9]):
        if qc[i % 3].button(sym, use_container_width=True, key=f"qs_{sym}"):
            ticker_input = sym

    st.divider()
    period_label = st.selectbox("Period", list(PERIODS.keys()), index=5)

    st.divider()
    show_ema    = st.checkbox("EMA Lines (20 / 50 / 200)", value=True)
    show_bb     = st.checkbox("Bollinger Bands",            value=False)
    show_volume = st.checkbox("Volume Panel",               value=True)

    st.divider()
    st.caption("Data via Yahoo Finance · Cache: 60 s")

period = PERIODS[period_label]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="tv-title">📈 {ticker_input}</div>', unsafe_allow_html=True)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TOOLBAR  —  Timeframe pills  +  Chart type pills                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with st.container():
    st.markdown('<div class="toolbar-container">', unsafe_allow_html=True)

    # ── Row 1: Timeframe ──────────────────────────────────────────────────────
    left_tf, right_ct = st.columns([3, 1], gap="large")

    with left_tf:
        st.markdown('<div class="toolbar-group-label">⏱ Timeframe</div>', unsafe_allow_html=True)

        tf_row1, tf_row2 = st.columns([1.2, 1], gap="small")

        with tf_row1:
            st.markdown('<span style="color:#484f58;font-size:10px;font-family:JetBrains Mono,monospace;letter-spacing:1px">INTRADAY</span>', unsafe_allow_html=True)
            sel_intra = st.pills(
                label="intraday",
                options=TF_INTRADAY,
                default="1h",
                key="pills_intraday",
                label_visibility="collapsed",
            )

        with tf_row2:
            st.markdown('<span style="color:#484f58;font-size:10px;font-family:JetBrains Mono,monospace;letter-spacing:1px">DAY · WEEK · MONTH</span>', unsafe_allow_html=True)
            sel_dwm = st.pills(
                label="dwm",
                options=TF_DAILY + TF_WEEKLY + TF_MONTHLY,
                default="1D",
                key="pills_dwm",
                label_visibility="collapsed",
            )

    # ── Row 1 Right: Chart Type ───────────────────────────────────────────────
    with right_ct:
        st.markdown('<div class="toolbar-group-label">🕯️ Chart Type</div>', unsafe_allow_html=True)
        sel_chart_label = st.pills(
            label="chart_type",
            options=list(CHART_OPTIONS.keys()),
            default="🕯️ Candlestick",
            key="pills_chart",
            label_visibility="collapsed",
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ── Resolve selections ────────────────────────────────────────────────────────
# Whichever pill group was last changed takes precedence for timeframe
# We track via two independent pill groups; pick the most recently non-None one.
# Default: prioritise daily group; fall back to intraday.
if sel_dwm is not None and sel_intra is not None:
    # Both selected — use session state to detect which changed last
    prev_intra = st.session_state.get("_prev_intra", "1h")
    prev_dwm   = st.session_state.get("_prev_dwm",   "1D")
    if sel_intra != prev_intra:
        active_tf = sel_intra
    elif sel_dwm != prev_dwm:
        active_tf = sel_dwm
    else:
        active_tf = st.session_state.get("_active_tf", "1D")
elif sel_intra is not None:
    active_tf = sel_intra
elif sel_dwm is not None:
    active_tf = sel_dwm
else:
    active_tf = "1D"

st.session_state["_prev_intra"] = sel_intra or active_tf
st.session_state["_prev_dwm"]   = sel_dwm   or active_tf
st.session_state["_active_tf"]  = active_tf

interval   = TF_TO_INTERVAL.get(active_tf, "1d")
chart_type = CHART_OPTIONS.get(sel_chart_label, "Candlestick") if sel_chart_label else "Candlestick"

# ── Active state info bar ─────────────────────────────────────────────────────
ha_tag = ' <span class="ha-badge">HA</span>' if chart_type == "Heikin-Ashi" else ""
st.markdown(f"""
<div class="active-bar">
  <div class="active-item"><span class="active-key">Timeframe</span><span class="active-val">{active_tf}</span></div>
  <div class="active-item"><span class="active-key">Interval</span><span class="active-val">{interval}</span></div>
  <div class="active-item"><span class="active-key">Period</span><span class="active-val">{period_label}</span></div>
  <div class="active-item"><span class="active-key">Chart</span><span class="active-val">{chart_type}{ha_tag}</span></div>
</div>""", unsafe_allow_html=True)

# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker_input} · {active_tf} · {period_label}…"):
    df = fetch_data(ticker_input, period, interval)

if df.empty:
    st.error(
        f"❌ No data for **{ticker_input}** · `{interval}` · `{period}`. "
        "Try a longer period, or switch to a Daily/Weekly timeframe."
    )
    st.stop()

df = add_indicators(df)
verdict, score, signals = compute_signal(df)

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change     = last_close - prev_close
pct_change = (change / prev_close) * 100 if prev_close else 0

# ── Metrics ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Last Price</div>
      <div class="metric-value">${last_close:,.2f}</div>
      <div class="{'metric-delta-pos' if change>=0 else 'metric-delta-neg'}">
        {'▲' if change>=0 else '▼'} {abs(change):.2f} ({abs(pct_change):.2f}%)
      </div>
    </div>""", unsafe_allow_html=True)

with m2:
    badge_extra = ' <span class="ha-badge">HA</span>' if chart_type == "Heikin-Ashi" else ""
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Signal{badge_extra}</div>
      <div style="margin-top:6px">{signal_badge(verdict)}</div>
      <div class="metric-delta-pos" style="margin-top:4px">Score: {score:+d}</div>
    </div>""", unsafe_allow_html=True)

with m3:
    hi = float(df["High"].max()); lo = float(df["Low"].min())
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Period High / Low</div>
      <div class="metric-value" style="color:#3fb950">${hi:,.2f}</div>
      <div class="metric-delta-neg">${lo:,.2f}</div>
    </div>""", unsafe_allow_html=True)

with m4:
    rsi_val   = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 0
    rsi_color = "#f85149" if rsi_val > 70 else ("#3fb950" if rsi_val < 30 else "#e6edf3")
    rsi_lbl   = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">RSI (14)</div>
      <div class="metric-value" style="color:{rsi_color}">{rsi_val:.1f}</div>
      <div class="metric-label">{rsi_lbl}</div>
    </div>""", unsafe_allow_html=True)

with m5:
    vol = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Indicators", "🕯️ Heikin-Ashi Table", "📋 OHLCV", "🔍 Signal"])

with tab1:
    tc = st.columns(3)
    ind = {
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
    for i, (k, v) in enumerate(ind.items()):
        tc[i % 3].metric(k, v)

with tab2:
    ha_df = compute_heikin_ashi(df)
    ha_d  = ha_df[["HA_Open","HA_High","HA_Low","HA_Close"]].tail(50).sort_index(ascending=False).round(4)
    ha_d.columns = ["HA Open","HA High","HA Low","HA Close"]
    st.caption("HA_Close=(O+H+L+C)/4 · HA_Open=(prev_HA_Open+prev_HA_Close)/2")
    st.dataframe(ha_d, use_container_width=True, height=380)

with tab3:
    st.dataframe(
        df[["Open","High","Low","Close","Volume"]].tail(50).sort_index(ascending=False).round(4),
        use_container_width=True, height=380,
    )

with tab4:
    st.markdown("#### Signal Breakdown")
    for s in signals:
        st.markdown(f"- {s}")
    st.markdown(f"**Score:** `{score:+d}` → **{verdict}**")
    st.info("≥ +3 = BUY · ≤ -3 = SELL · else NEUTRAL")

st.divider()
st.caption("⚡ Python · Streamlit · Plotly · yfinance · ta  |  Yahoo Finance  |  Not financial advice")
