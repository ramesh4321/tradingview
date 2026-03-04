# 📈 TradingView-Style Dashboard — Python + Streamlit

A professional stock & crypto dashboard inspired by TradingView, built with **Python**, **Streamlit**, **Plotly**, and **yfinance**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple?style=flat-square&logo=plotly)

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📊 **Candlestick & Line Charts** | Interactive Plotly charts with zoom / pan |
| 📈 **EMA Lines** | 20 / 50 / 200 period Exponential Moving Averages |
| 🎯 **Bollinger Bands** | 20-period BB with 2 std deviations |
| 📉 **MACD Panel** | MACD, Signal line, Histogram |
| 💧 **Volume Panel** | Volume bars color-coded by candle direction |
| 🔢 **RSI (14)** | Overbought / Oversold levels |
| 🧠 **Auto Signal** | BUY / SELL / NEUTRAL based on indicator stack |
| 🪙 **Any Symbol** | Stocks, ETFs, Crypto (BTC-USD, ETH-USD), Forex |
| ⚡ **60-second cache** | Fast re-renders via `st.cache_data` |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/tradingview-dashboard.git
cd tradingview-dashboard
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
tradingview-dashboard/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as main file
4. Click **Deploy** — live URL in ~60 seconds!

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `yfinance` | Yahoo Finance data |
| `plotly` | Interactive charts |
| `ta` | Technical analysis indicators |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and does **not** constitute financial advice. Always do your own research before making investment decisions.

---

## 📜 License

MIT License — free to use, modify, and distribute.
