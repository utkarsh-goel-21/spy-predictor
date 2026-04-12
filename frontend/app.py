import os
import threading
import time
from datetime import datetime

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
SELF_PING_URL = "https://spy-predictor-ui.onrender.com"


@st.cache_resource
def ensure_self_ping_started():
    def ping_loop():
        session = requests.Session()
        while True:
            time.sleep(600)
            try:
                response = session.get(SELF_PING_URL, timeout=30)
                response.raise_for_status()
            except Exception:
                pass

    thread = threading.Thread(target=ping_loop, daemon=True)
    thread.start()
    return True


ensure_self_ping_started()

st.set_page_config(
    page_title="SPY · Predictor",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #111111 !important;
    color: #e2e2e2 !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stHeader"] { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebarCollapsedControl"] { display: none; }

.block-container {
    max-width: 1200px !important;
    padding: 3rem 2rem !important;
}

/* Top nav bar */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 3.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1f1f1f;
}

.topbar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}

.logo-mark {
    width: 32px;
    height: 32px;
    background: #e2e2e2;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    color: #111;
    font-weight: 800;
}

.logo-text {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e2e2;
    letter-spacing: 0.5px;
}

.topbar-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Section labels */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #777;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

/* Cards */
.card {
    background: #161616;
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    padding: 1.5rem;
    height: 100%;
}

.card:hover {
    border-color: #2a2a2a;
}

/* Direction display */
.direction-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 5rem;
    font-weight: 500;
    line-height: 1;
    letter-spacing: -3px;
}

.up { color: #4ade80; }
.down { color: #f87171; }
.neutral { color: #71717a; }

/* Confidence bar */
.conf-track {
    height: 2px;
    background: #1f1f1f;
    border-radius: 2px;
    margin-top: 1.25rem;
}

/* Signal badge */
.signal-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    letter-spacing: 0.5px;
}

.pill-green {
    background: rgba(74, 222, 128, 0.08);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.2);
}

.pill-red {
    background: rgba(248, 113, 113, 0.08);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.2);
}

.pill-yellow {
    background: rgba(250, 204, 21, 0.08);
    color: #facc15;
    border: 1px solid rgba(250, 204, 21, 0.2);
}

.pill-grey {
    background: rgba(113, 113, 122, 0.08);
    color: #71717a;
    border: 1px solid rgba(113, 113, 122, 0.2);
}

/* Stat block */
.stat-block {
    margin-bottom: 1.5rem;
}

.stat-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem;
    font-weight: 400;
    color: #e2e2e2;
    line-height: 1;
}

.stat-key {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.35rem;
}

/* Divider */
.row-divider {
    border: none;
    border-top: 1px solid #1f1f1f;
    margin: 2.5rem 0;
}

/* News */
.news-item {
    padding: 1.1rem 0;
    border-bottom: 1px solid #1a1a1a;
}

.news-item:last-child { border-bottom: none; }

.news-headline {
    font-size: 0.875rem;
    font-weight: 500;
    color: #d4d4d4;
    line-height: 1.5;
    margin-bottom: 0.4rem;
}

.news-body {
    font-size: 0.78rem;
    color: #666;
    line-height: 1.6;
    margin-bottom: 0.5rem;
}

.news-footer {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #555;
}

.sent-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 1px 7px;
    border-radius: 4px;
}

.sent-bull { background: rgba(74,222,128,0.08); color: #4ade80; }
.sent-bear { background: rgba(248,113,113,0.08); color: #f87171; }
.sent-neut { background: rgba(113,113,122,0.1); color: #71717a; }

/* Feature tags */
.feat-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 8px;
    border-radius: 4px;
    background: #1a1a1a;
    color: #666;
    margin: 2px;
    border: 1px solid #222;
}

/* Backtest table */
.bt-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
}

.bt-table th {
    text-align: left;
    color: #555;
    font-size: 0.62rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0 0.75rem 0.75rem 0;
    border-bottom: 1px solid #1f1f1f;
}

.bt-table td {
    padding: 0.6rem 0.75rem 0.6rem 0;
    border-bottom: 1px solid #191919;
    color: #aaa;
    vertical-align: middle;
}

.bt-table tr:last-child td { border-bottom: none; }

.bt-correct { color: #4ade80; font-weight: 600; }
.bt-wrong   { color: #f87171; font-weight: 600; }

.bt-summary-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 400;
    color: #e2e2e2;
    line-height: 1;
}

.bt-summary-key {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
}

/* Date input styling */
[data-testid="stDateInput"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #555 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}

[data-testid="stDateInput"] input {
    background: #161616 !important;
    border: 1px solid #1f1f1f !important;
    color: #d4d4d4 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Button override */
.stButton > button {
    background: #e2e2e2 !important;
    color: #111 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.75rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    background: #fff !important;
    transform: translateY(-1px) !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: #444 !important; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 6rem 0;
}

.empty-icon {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    color: #555;
    margin-bottom: 1.5rem;
}

.empty-title {
    font-size: 1rem;
    font-weight: 600;
    color: #aaa;
    margin-bottom: 0.5rem;
}

.empty-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #555;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# Top bar
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <div class="logo-mark">◆</div>
        <div class="logo-text">SPY Predictor</div>
    </div>
    <div class="topbar-tag">LSTM · News Sentiment · Live</div>
</div>
""", unsafe_allow_html=True)

from datetime import date, timedelta

# ── Controls row ─────────────────────────────────────────────────────────────
col_pred, col_gap, col_d1, col_d2, col_bt = st.columns([1.2, 0.3, 1, 1, 1.2])

with col_pred:
    st.markdown("<div style='padding-top:1.75rem;'>", unsafe_allow_html=True)
    run = st.button("Run Prediction →")
    st.markdown("</div>", unsafe_allow_html=True)

with col_d1:
    bt_start = st.date_input(
        "Backtest From",
        value=date.today() - timedelta(days=90),
        max_value=date.today() - timedelta(days=2),
        label_visibility="visible"
    )

with col_d2:
    bt_end = st.date_input(
        "Backtest To",
        value=date.today() - timedelta(days=1),
        max_value=date.today() - timedelta(days=1),
        label_visibility="visible"
    )

with col_bt:
    st.markdown("<div style='padding-top:1.75rem;'>", unsafe_allow_html=True)
    bt_run = st.button("Run Backtest →")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Backtest results ──────────────────────────────────────────────────────────
if bt_run:
    with st.spinner("Running backtest..."):
        try:
            bt_resp = requests.get(
                f"{API_URL}/backtest",
                params={"start_date": str(bt_start), "end_date": str(bt_end)},
                timeout=120
            ).json()
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

    if "detail" in bt_resp:
        st.error(bt_resp["detail"])
        st.stop()

    s = bt_resp["summary"]
    results = bt_resp["results"]

    st.markdown("<hr class='row-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Backtest Results</div>", unsafe_allow_html=True)

    # ── Summary cards ─────────────────────────────────────────────
    acc_color = "#4ade80" if s["accuracy"] >= 0.55 else "#f87171" if s["accuracy"] < 0.48 else "#facc15"

    bs1, bs2, bs3, bs4, bs5 = st.columns(5)

    with bs1:
        st.markdown(f"""
        <div class="card">
            <div class="bt-summary-val" style="color:{acc_color};">{s['accuracy']*100:.1f}%</div>
            <div class="bt-summary-key">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with bs2:
        st.markdown(f"""
        <div class="card">
            <div class="bt-summary-val">{s['correct_count']} <span style="font-size:0.9rem; color:#555;">/ {s['total']}</span></div>
            <div class="bt-summary-key">Correct / Total</div>
        </div>
        """, unsafe_allow_html=True)

    with bs3:
        up_acc = f"{s['up_accuracy']*100:.1f}%" if s['up_accuracy'] is not None else "—"
        st.markdown(f"""
        <div class="card">
            <div class="bt-summary-val" style="color:#4ade80;">{up_acc}</div>
            <div class="bt-summary-key">UP Call Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with bs4:
        dn_acc = f"{s['down_accuracy']*100:.1f}%" if s['down_accuracy'] is not None else "—"
        st.markdown(f"""
        <div class="card">
            <div class="bt-summary-val" style="color:#f87171;">{dn_acc}</div>
            <div class="bt-summary-key">DOWN Call Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with bs5:
        avg_c = f"{s['avg_confidence_correct']*100:.1f}%" if s['avg_confidence_correct'] else "—"
        avg_w = f"{s['avg_confidence_wrong']*100:.1f}%" if s['avg_confidence_wrong'] else "—"
        st.markdown(f"""
        <div class="card">
            <div class="bt-summary-val" style="font-size:1.1rem; line-height:1.6;">
                <span style="color:#4ade80;">{avg_c}</span>
                <span style="color:#333; font-size:0.8rem;"> / </span>
                <span style="color:#f87171;">{avg_w}</span>
            </div>
            <div class="bt-summary-key">Conf: Right / Wrong</div>
        </div>
        """, unsafe_allow_html=True)

    if s.get("range_overlaps_training"):
        in_sample_acc = s["in_sample_accuracy"] * 100 if s["in_sample_accuracy"] is not None else None
        out_sample_acc = s["out_of_sample_accuracy"] * 100 if s["out_of_sample_accuracy"] is not None else None
        out_sample_text = f"{out_sample_acc:.1f}%" if out_sample_acc is not None else "—"
        st.markdown(f"""
        <div class="card" style="margin-top:1rem; border-color:rgba(250,204,21,0.2); background:rgba(250,204,21,0.06);">
            <div class="section-label" style="color:#facc15;">Backtest Warning</div>
            <div style="font-size:0.92rem; line-height:1.6; color:#d4d4d4;">
                This range overlaps the training set through <span style="font-family:'IBM Plex Mono',monospace;">{s['training_end_date']}</span>.
                In-sample accuracy is <span style="color:#facc15;">{in_sample_acc:.1f}%</span> on {s['in_sample_count']} days,
                while unseen accuracy is <span style="color:#facc15;">{out_sample_text}</span> on {s['out_of_sample_count']} days.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Results table ──────────────────────────────────────────────
    st.markdown("<div style='margin-top:2rem;'>", unsafe_allow_html=True)
    table_rows = []
    for r in results:
        table_rows.append({
            "Date": r["date"],
            "Predicted": r["predicted_direction"],
            "Actual": r["actual_direction"],
            "Result": "✓" if r["correct"] else "✗",
            "Confidence": f"{r['confidence']*100:.1f}%",
            "Seen In Training": "Yes" if r.get("in_sample") else "No",
        })

    st.dataframe(
        pd.DataFrame(table_rows),
        hide_index=True,
        width="stretch",
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr class='row-divider'>", unsafe_allow_html=True)

if run:
    with st.spinner("Fetching live data..."):
        try:
            dashboard = requests.get(f"{API_URL}/dashboard", timeout=90).json()
            pred = dashboard["prediction"]
            chart = dashboard["chart"]
            news = dashboard["news"]
            info = dashboard["model_info"]
        except Exception as e:
            st.error(f"API connection failed: {e}")
            st.stop()

    direction = pred["direction"]
    confidence = pred["confidence"]
    sentiment = pred["sentiment"]
    signal = pred["combined_signal"]
    score = sentiment["sentiment_score"]

    if pred.get("market_data_warning"):
        st.markdown(f"""
        <div class="card" style="margin-bottom:1.25rem; border-color:rgba(250,204,21,0.2); background:rgba(250,204,21,0.06);">
            <div class="section-label" style="color:#facc15;">Market Data Status</div>
            <div style="font-size:0.92rem; line-height:1.6; color:#d4d4d4;">
                {pred['market_data_warning']}
                <div style="margin-top:0.6rem; font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#a1a1aa;">
                    Raw SPY bar: {pred.get('raw_data_last_date', '—')} · Feature row used: {pred['as_of_date']} · Expected latest bar: {pred.get('expected_latest_bar_date', '—')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 1: Core metrics ──────────────────────────────
    st.markdown("<div class='section-label'>Prediction</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

    with c1:
        dir_class = "up" if direction == "UP" else "down"
        dir_symbol = "▲" if direction == "UP" else "▼"
        conf_color = "#4ade80" if direction == "UP" else "#f87171"
        st.markdown(f"""
        <div class="card">
            <div class="section-label">Direction</div>
            <div class="direction-number {dir_class}">{dir_symbol}</div>
            <div style="margin-top:0.75rem; font-size:2rem; font-weight:800; color:{'#4ade80' if direction=='UP' else '#f87171'};">{direction}</div>
            <div class="conf-track">
                <div style="height:2px; width:{confidence*100:.0f}%; background:{conf_color}; border-radius:2px;"></div>
            </div>
            <div style="margin-top:0.5rem; font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#777;">{confidence*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        if "Strong UP" in signal:
            pill = "pill-green"; icon = "↑↑"
        elif "Strong DOWN" in signal:
            pill = "pill-red"; icon = "↓↓"
        elif "Weak" in signal:
            pill = "pill-yellow"; icon = "~"
        else:
            pill = "pill-grey"; icon = "?"

        st.markdown(f"""
        <div class="card">
            <div class="section-label">Signal</div>
            <div style="margin: 0.75rem 0;">
                <span class="signal-pill {pill}">{icon} {signal}</span>
            </div>
            <div style="margin-top:1.5rem;">
                <div class="stat-key">Predicting For</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem; color:#d4d4d4; margin-top:0.35rem;">{pred['predicting_for']}</div>
            </div>
            <div style="margin-top:0.75rem;">
                <div class="stat-key">As Of</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem; color:#888; margin-top:0.35rem;">{pred['as_of_date']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        if score >= 0.15:
            s_color = "#4ade80"
        elif score <= -0.15:
            s_color = "#f87171"
        else:
            s_color = "#71717a"

        st.markdown(f"""
        <div class="card">
            <div class="section-label">Sentiment</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:2.2rem; font-weight:400; color:{s_color}; margin: 0.5rem 0;">{score:+.3f}</div>
            <div style="font-size:0.85rem; font-weight:600; color:{s_color};">{sentiment['sentiment_label']}</div>
            <div style="margin-top:1rem;">
                <div class="stat-key">Articles</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem; color:#d4d4d4; margin-top:0.35rem;">{sentiment['article_count']}</div>
            </div>
            <div style="margin-top:0.75rem;">
                <div class="stat-key">Date</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem; color:#888; margin-top:0.35rem;">{sentiment['sentiment_date']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="card">
            <div class="section-label">Model</div>
            <div class="stat-block">
                <div class="stat-val">{info['cv_accuracy']*100:.1f}%</div>
                <div class="stat-key">CV Accuracy</div>
            </div>
            <div class="stat-block">
                <div class="stat-val">{info['sequence_length']}d</div>
                <div class="stat-key">Lookback Window</div>
            </div>
            <div class="stat-block">
                <div class="stat-val">{info['num_features']}</div>
                <div class="stat-key">Features</div>
            </div>
            <div style="font-size:0.75rem; font-weight:600; color:#555;">{info['model']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='row-divider'>", unsafe_allow_html=True)

    # ── Row 2: Chart ────────────────────────────────────
    st.markdown("<div class='section-label'>SPY · 3 Month</div>", unsafe_allow_html=True)

    dates = [d["date"] for d in chart["data"]]
    closes = [d["close"] for d in chart["data"]]
    price_color = "#4ade80" if closes[-1] >= closes[0] else "#f87171"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        mode="lines",
        line=dict(color=price_color, width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba({'74,222,128' if price_color == '#4ade80' else '248,113,113'}, 0.04)",
        hovertemplate="<b>%{x}</b><br>$%{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#666", family="IBM Plex Mono"),
        xaxis=dict(
            showgrid=False, showline=False, zeroline=False,
            tickfont=dict(size=10, color="#555"),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#191919",
            showline=False, zeroline=False,
            tickfont=dict(size=10, color="#555"),
            tickprefix="$",
        ),
        margin=dict(l=0, r=0, t=8, b=0),
        height=240,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1a1a1a",
            bordercolor="#2a2a2a",
            font=dict(color="#e2e2e2", family="IBM Plex Mono", size=11)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='row-divider'>", unsafe_allow_html=True)

    # ── Row 3: News + Features ───────────────────────────
    col_news, col_feat = st.columns([1.6, 1])

    with col_news:
        st.markdown("<div class='section-label'>Recent SPY News</div>", unsafe_allow_html=True)
        articles = news.get("articles", [])
        if articles:
            import html as html_lib
            for a in articles[:7]:
                label = a["sentiment_label"]
                sc = a["sentiment_score"]
                if "Bullish" in label:
                    tag_class = "sent-bull"
                elif "Bearish" in label:
                    tag_class = "sent-bear"
                else:
                    tag_class = "sent-neut"

                try:
                    dt = datetime.strptime(a["time_published"], "%Y%m%dT%H%M%S")
                    t_str = dt.strftime("%b %d · %H:%M")
                except:
                    t_str = a["time_published"]

                # Escape special chars that break HTML injection
                safe_title = html_lib.escape(a['title'])
                safe_summary = html_lib.escape(a['summary'])
                safe_source = html_lib.escape(a['source'])

                st.markdown(f"""
                <div class="news-item">
                    <div class="news-headline">{safe_title}</div>
                    <div class="news-body">{safe_summary}</div>
                    <div class="news-footer">
                        <span class="sent-tag {tag_class}">{label} {sc:+.2f}</span>
                        <span>{safe_source}</span>
                        <span>{t_str}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#555; font-size:0.85rem; padding:1rem 0;'>No recent articles found.</div>", unsafe_allow_html=True)

    with col_feat:
        # ── "Features Used" label removed, card content kept ──
        st.markdown("<div class='card' style='margin-top:0;'>", unsafe_allow_html=True)

        prob_up = pred["prob_up"]
        prob_down = pred["prob_down"]

        st.markdown(f"""
        <div style="margin-bottom:1.5rem;">
            <div class="stat-key" style="margin-bottom:0.75rem;">Probability Breakdown</div>
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#4ade80; width:40px;">UP</div>
                <div style="flex:1; height:4px; background:#1a1a1a; border-radius:2px;">
                    <div style="width:{prob_up*100:.0f}%; height:4px; background:#4ade80; border-radius:2px;"></div>
                </div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#4ade80;">{prob_up*100:.1f}%</div>
            </div>
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#f87171; width:40px;">DN</div>
                <div style="flex:1; height:4px; background:#1a1a1a; border-radius:2px;">
                    <div style="width:{prob_down*100:.0f}%; height:4px; background:#f87171; border-radius:2px;"></div>
                </div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#f87171;">{prob_down*100:.1f}%</div>
            </div>
        </div>
        <div class="stat-key" style="margin-bottom:0.6rem;">Feature Set</div>
        <div>
            {''.join([f'<span class="feat-tag">{f}</span>' for f in info['features']])}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif not bt_run:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">◆</div>
        <div class="empty-title">Ready to predict</div>
        <div class="empty-sub">Click "Run Prediction" to fetch live data and run the model</div>
    </div>
    """, unsafe_allow_html=True)
