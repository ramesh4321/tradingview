import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import numpy as np

# ── Page Config ──────────────────────────────────────────────────────────────
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

/* Sidebar */
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

/* Metric Cards */
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

/* Header */
.tv-header {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid #30363d;
    padding: 12px 0 20px 0;
    margin-bottom: 16px;
}
.tv-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 28px;
    color: #58a6ff;
    letter-spacing: -1px;
}
.tv-subtitle { color: #8b949e; font-size: 13px; }

/* Signal badges */
.signal-buy {
    background: #1a3a2a; color: #3fb950; border: 1px solid #3fb950;
    padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 700;
    letter-spacing: 1px;
}
.signal-sell {
    background: #3a1a1a; color: #f85149; border: 1px solid #f85149;
    padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 700;
    letter-spacing: 1px;
}
.signal-neutral {
    background: #1f2937; color: #8b949e; border: 1px solid #8b949e;
    padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 700;
    letter-spacing: 1px;
}

/* Table styling */
.stDataFrame { background: #161b22; }
div[data-testid="stMetricValue"] { color: #e6edf3 !important; }

/* Tabs */
button[data-baseweb="tab"] { color: #8b949e !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
INTERVALS = {
    "1 Min": "1m", "5 Min": "5m", "15 Min": "15m", "30 Min": "30m",
    "1 Hour": "1h", "4 Hour": "4h", "1 Day": "1d", "1 Week": "1wk", "1 Month": "1mo"
}
PERIODS = {
    "1 Day": "1d", "5 Days": "5d", "1 Month": "1mo", "3 Months": "3mo",
    "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y",
}
POPULAR_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BTC-USD", "ETH-USD", "SPY", "QQQ"]

@st.cache_data(ttl=60)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
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
    df["MACD_signal"]  = macd_obj.macd_signal()
    df["MACD_hist"]    = macd_obj.macd_diff()

    df["RSI"]    = ta.momentum.rsi(close, window=14)
    df["Volume_MA"] = volume.rolling(20).mean()

    df["ATR"]  = ta.volatility.average_true_range(high, low, close, window=14)
    return df

def compute_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    score = 0
    signals = []

    rsi = float(last.get("RSI", 50))
    if rsi < 30:   score += 2; signals.append("RSI Oversold")
    elif rsi > 70: score -= 2; signals.append("RSI Overbought")

    close = float(last["Close"])
    ema20 = float(last.get("EMA20", close))
    ema50 = float(last.get("EMA50", close))
    if close > ema20 > ema50: score += 2; signals.append("EMA Bullish Stack")
    elif close < ema20 < ema50: score -= 2; signals.append("EMA Bearish Stack")

    macd = float(last.get("MACD", 0))
    macd_sig = float(last.get("MACD_signal", 0))
    if macd > macd_sig: score += 1; signals.append("MACD Bullish")
    else:               score -= 1; signals.append("MACD Bearish")

    if score >= 3:   verdict = "BUY"
    elif score <= -3: verdict = "SELL"
    else:             verdict = "NEUTRAL"
    return verdict, score, signals

def signal_badge(verdict):
    cls = {"BUY": "signal-buy", "SELL": "signal-sell", "NEUTRAL": "signal-neutral"}[verdict]
    return f'<span class="{cls}">{verdict}</span>'

def build_chart(df, ticker, show_ema, show_bb, show_volume, chart_type):
    rows   = 3 if show_volume else 2
    heights= [0.55, 0.25, 0.20] if show_volume else [0.65, 0.35]
    specs  = [[{"secondary_y": False}]] * rows

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=heights,
        specs=specs,
    )

    # ── Candlestick / Line ──
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950", decreasing_fillcolor="#f85149",
            name=ticker, line=dict(width=1),
        ), row=1, col=1)
    else:
        close_vals = df["Close"].squeeze()
        fig.add_trace(go.Scatter(
            x=df.index, y=close_vals,
            line=dict(color="#58a6ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
            name=ticker,
        ), row=1, col=1)

    # ── EMAs ──
    if show_ema:
        for col, color, label in [("EMA20","#f0883e","EMA 20"),("EMA50","#bc8cff","EMA 50"),("EMA200","#ff7b72","EMA 200")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1),
                                         name=label, opacity=0.9), row=1, col=1)

    # ── Bollinger Bands ──
    if show_bb:
        for col, color, lbl in [("BB_upper","rgba(139,148,158,0.6)","BB Upper"),
                                  ("BB_mid",  "rgba(139,148,158,0.3)","BB Mid"),
                                  ("BB_lower","rgba(139,148,158,0.6)","BB Lower")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col],
                                          line=dict(color=color, width=1, dash="dot"),
                                          name=lbl, opacity=0.7), row=1, col=1)

    # ── MACD ──
    if "MACD" in df.columns:
        colors = ["#3fb950" if v >= 0 else "#f85149" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=colors, name="MACD Hist", opacity=0.7), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#58a6ff", width=1), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], line=dict(color="#f0883e", width=1), name="Signal"), row=2, col=1)

    # ── Volume ──
    if show_volume and rows == 3:
        volume = df["Volume"].squeeze()
        vol_colors = ["#3fb950" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#f85149"
                      for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=volume, marker_color=vol_colors,
                              name="Volume", opacity=0.7), row=3, col=1)
        if "Volume_MA" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["Volume_MA"],
                                      line=dict(color="#bc8cff", width=1), name="Vol MA20"), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="JetBrains Mono"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        height=680,
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor="#21262d", showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", showgrid=True, row=i, col=1, tickfont=dict(size=10))

    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="tv-title">📈 TradingView</div>', unsafe_allow_html=True)
    st.markdown('<div class="tv-subtitle">Python · Streamlit Dashboard</div>', unsafe_allow_html=True)
    st.divider()

    ticker_input = st.text_input("Symbol", value="AAPL", placeholder="AAPL, BTC-USD …").upper().strip()
    if not ticker_input:
        ticker_input = "AAPL"

    st.markdown("**Quick Select**")
    cols = st.columns(3)
    for idx, sym in enumerate(POPULAR_TICKERS[:9]):
        if cols[idx % 3].button(sym, use_container_width=True, key=f"btn_{sym}"):
            ticker_input = sym

    st.divider()
    period_label   = st.selectbox("Period",   list(PERIODS.keys()),   index=5)
    interval_label = st.selectbox("Interval", list(INTERVALS.keys()), index=6)

    st.divider()
    chart_type  = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
    show_ema    = st.checkbox("EMA Lines (20/50/200)", value=True)
    show_bb     = st.checkbox("Bollinger Bands", value=False)
    show_volume = st.checkbox("Volume Panel", value=True)

    st.divider()
    st.caption("Data via Yahoo Finance · Refresh: 60 s")

period   = PERIODS[period_label]
interval = INTERVALS[interval_label]


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="tv-title">📈 {ticker_input}</div>', unsafe_allow_html=True)

with st.spinner(f"Fetching {ticker_input}…"):
    df = fetch_data(ticker_input, period, interval)

if df.empty:
    st.error(f"❌ No data found for **{ticker_input}**. Check the symbol and try again.")
    st.stop()

df = add_indicators(df)
info    = yf.Ticker(ticker_input).fast_info
verdict, score, signals = compute_signal(df)

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change     = last_close - prev_close
pct_change = (change / prev_close) * 100 if prev_close else 0

# ── Metric Row ────────────────────────────────────────────────────────────────
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
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Signal</div>
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
    rsi_val = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 0
    rsi_color = "#f85149" if rsi_val > 70 else ("#3fb950" if rsi_val < 30 else "#e6edf3")
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">RSI (14)</div>
      <div class="metric-value" style="color:{rsi_color}">{rsi_val:.1f}</div>
      <div class="metric-label">{'Overbought' if rsi_val>70 else 'Oversold' if rsi_val<30 else 'Neutral'}</div>
    </div>""", unsafe_allow_html=True)

with c5:
    vol = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Volume</div>
      <div class="metric-value">{vol/1e6:.2f}M</div>
      <div class="metric-label">Last candle</div>
    </div>""", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
st.plotly_chart(
    build_chart(df, ticker_input, show_ema, show_bb, show_volume, chart_type),
    use_container_width=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Technical Analysis", "📋 OHLCV Data", "🔍 Signal Breakdown"])

with tab1:
    cols = st.columns(3)
    indicators = {
        "EMA 20":  f"${float(df['EMA20'].iloc[-1]):,.2f}"  if "EMA20"  in df.columns else "N/A",
        "EMA 50":  f"${float(df['EMA50'].iloc[-1]):,.2f}"  if "EMA50"  in df.columns else "N/A",
        "EMA 200": f"${float(df['EMA200'].iloc[-1]):,.2f}" if "EMA200" in df.columns else "N/A",
        "BB Upper":f"${float(df['BB_upper'].iloc[-1]):,.2f}" if "BB_upper" in df.columns else "N/A",
        "BB Lower":f"${float(df['BB_lower'].iloc[-1]):,.2f}" if "BB_lower" in df.columns else "N/A",
        "ATR (14)":f"${float(df['ATR'].iloc[-1]):,.2f}"     if "ATR" in df.columns else "N/A",
        "MACD":    f"{float(df['MACD'].iloc[-1]):,.4f}"     if "MACD" in df.columns else "N/A",
        "MACD Sig":f"{float(df['MACD_signal'].iloc[-1]):,.4f}" if "MACD_signal" in df.columns else "N/A",
        "RSI (14)":f"{float(df['RSI'].iloc[-1]):,.2f}"      if "RSI" in df.columns else "N/A",
    }
    for idx, (k, v) in enumerate(indicators.items()):
        cols[idx % 3].metric(k, v)

with tab2:
    display_df = df[["Open","High","Low","Close","Volume"]].tail(50).sort_index(ascending=False)
    display_df = display_df.round(4)
    st.dataframe(display_df, use_container_width=True, height=400)

with tab3:
    st.markdown("#### Signal Analysis")
    for s in signals:
        st.markdown(f"- {s}")
    st.markdown(f"**Overall Score:** `{score:+d}` → **{verdict}**")
    st.info("Score ≥ +3 = BUY · Score ≤ -3 = SELL · Otherwise NEUTRAL")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚡ Built with Python · Streamlit · Plotly · yfinance · ta-lib  |  Data from Yahoo Finance  |  Not financial advice")
