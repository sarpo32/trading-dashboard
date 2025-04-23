import streamlit as st
st.set_page_config(page_title="DarkEx Dashboard", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
import time

# ---------- PDF REPORT GENERATOR ----------
def generate_pdf_report(user, symbol, total_profit, avg_pl, win_rate,
                        sharpe, drawdown, trade_count, pl_ratio, total_fees):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "DarkEx Trading Performance Dashboard", ln=True, align="C")
    pdf.ln(5)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"User ID: {user}", ln=True)
    pdf.cell(0, 8, f"Symbol: {symbol}", ln=True)
    pdf.ln(3)
    pdf.cell(0, 8, f"Total Profit: ${total_profit:,.2f}", ln=True)
    pdf.cell(0, 8, f"Average P/L: ${avg_pl:,.2f}", ln=True)
    pdf.cell(0, 8, f"Win Rate: {win_rate:.2f}%", ln=True)
    pdf.cell(0, 8, f"Sharpe Ratio: {sharpe}", ln=True)
    pdf.cell(0, 8, f"Max Drawdown: ${drawdown:,.2f}", ln=True)
    pdf.cell(0, 8, f"Total Trades: {trade_count}", ln=True)
    pdf.cell(0, 8, f"Total Wins: {int((trade_count and (None)) or 0)}", ln=True)  # placeholder
    pdf.cell(0, 8, f"Total Losses: {int((trade_count and (None)) or 0)}", ln=True)  # placeholder
    pdf.cell(0, 8, f"Total Funding Fees: ${total_fees:,.2f}", ln=True)
    if pl_ratio is not None:
        pdf.cell(0, 8, f"P/L Ratio: {pl_ratio:.2f}", ln=True)
    return pdf.output(dest='S').encode('latin1')


# ---------- METRIC CALCULATORS ----------
@st.cache_data
def compute_sharpe_ratio(df):
    daily_pnl = df.groupby(df['date'].dt.date)['profit_loss'].sum()
    daily_ret = daily_pnl.pct_change().dropna()
    risk_free = 0.02 / 252
    if daily_ret.std() == 0:
        return 0.0
    return round((daily_ret.mean() - risk_free) / daily_ret.std(), 2)

@st.cache_data
def compute_drawdown(df):
    cum_pnl = df['profit_loss'].cumsum()
    peak    = cum_pnl.cummax()
    dd      = cum_pnl - peak
    return round(dd.min(), 2)

def compute_pl_ratio(df):
    wins = df.loc[df['profit_loss'] > 0, 'profit_loss']
    loss = df.loc[df['profit_loss'] < 0, 'profit_loss']
    if wins.empty or loss.empty:
        return None
    return round(wins.mean() / abs(loss.mean()), 2)


# ---------- BENCHMARK SCORING ----------
def min_max_norm(val, lo, hi):
    return (val - lo) / (hi - lo) if hi != lo else 0.0

def calculate_benchmark_score(df):
    n = len(df)
    if n == 0:
        return {}, 0.0

    win_rate   = df['profit_loss'].gt(0).sum() / n
    pl_ratio   = compute_pl_ratio(df) or 0.0
    pnl_vol    = df['profit_loss'].std()
    duration_h = (df['date'].max() - df['date'].min()).total_seconds() / 3600 + 1
    freq       = n / duration_h

    if {'entryprice','exitprice'}.issubset(df.columns):
        e = pd.to_numeric(df['entryprice'], errors='coerce').abs()
        x = pd.to_numeric(df['exitprice'],  errors='coerce').abs()
        df['trade_size'] = e + x
        avg_size = df['trade_size'].mean()
    else:
        avg_size = 0.0

    ref = {
        'win_rate':        {'min':0.0, '95th':0.65},
        'pl_ratio':        {'min':0.0, '95th':1.0},
        'avg_trade_size':  {'min':0.0, '95th':30000},
        'pnl_volatility':  {'min':0.0, '95th':5000},
        'trade_frequency': {'min':0.0, '95th':10},
    }

    def norm(x, metric):
        lo, hi = ref[metric]['min'], ref[metric]['95th']
        return round(min_max_norm(min(x, hi), lo, hi), 3)

    normd = {
        'win_rate_norm':        norm(win_rate,        'win_rate'),
        'pl_ratio_norm':        norm(pl_ratio,        'pl_ratio'),
        'avg_trade_size_norm':  norm(avg_size,        'avg_trade_size'),
        'pnl_volatility_norm':  norm(pnl_vol,         'pnl_volatility'),
        'trade_frequency_norm': norm(freq,            'trade_frequency'),
    }

    weights = {
        'win_rate_norm':        0.1127,
        'pl_ratio_norm':        0.1756,
        'avg_trade_size_norm':  0.2629,
        'pnl_volatility_norm':  0.2700,
        'trade_frequency_norm': 0.1789,
    }

    score = round(sum(normd[k] * weights[k] for k in weights), 3)
    label = 'Hold'     if score < 0.33 else \
            'Monitor'  if score < 0.66 else \
            'Transmit'

    normd.update({'benchmark_score': score, 'risk_label': label})
    return normd, score


# ---------- PASSWORD PROTECTION ----------
def check_password():
    def _enter():
        st.session_state.auth = (st.session_state.pw == "mydashboard123")
    if 'auth' not in st.session_state:
        st.text_input("Enter password:", type="password", key='pw', on_change=_enter)
        return False
    if not st.session_state.auth:
        st.text_input("Enter password:", type="password", key='pw', on_change=_enter)
        st.error("Incorrect password")
        return False
    return True


# ---------- MAIN APP ----------
if check_password():
    st.title("DarkEx Trading Performance Dashboard")
    uploaded = st.file_uploader("Upload your trades CSV", type="csv")
    if not uploaded:
        st.info("Please upload a CSV to proceed.")
        st.stop()

    t0 = time.time()
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.lower()

    # 1) Profit/Loss calculation
    df['profit_loss'] = 0.0
    if 'accpnl' in df.columns:
        df['profit_loss'] = pd.to_numeric(df['accpnl'], errors='coerce').fillna(0.0)
    if {'entryprice','exitprice'}.issubset(df.columns):
        e  = pd.to_numeric(df['entryprice'], errors='coerce')
        x  = pd.to_numeric(df['exitprice'],  errors='coerce')
        fb = (x - e).fillna(0.0)
        df.loc[df['profit_loss']==0, 'profit_loss'] = fb[df['profit_loss']==0]

    # 2) Parse funding fees
    fee_col = next((c for c in ['fundingfee','funding_fee'] if c in df.columns), None)
    df['funding_fee'] = pd.to_numeric(df[fee_col], errors='coerce').fillna(0.0) if fee_col else 0.0

    # 3) Date normalization & holding
    for c in ['createtime','updatetime','timestamp','date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    df['date'] = df.filter(['createtime','timestamp','updatetime','date']).bfill(axis=1).iloc[:,0]
    df.dropna(subset=['date'], inplace=True)
    df['holding'] = ((df['updatetime'] - df['createtime']).dt.total_seconds()/60
                     if {'createtime','updatetime'}.issubset(df.columns)
                     else np.nan)

    # 4) Filters
    st.subheader("Filter by Date & Symbol")
    lo, hi = df['date'].min().date(), df['date'].max().date()
    start_date, end_date = st.date_input("Date range", value=(lo,hi), min_value=lo, max_value=hi)
    df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]

    if 'contract' in df.columns:
        syms = df['contract'].dropna().unique().tolist()
        sel = st.selectbox("Symbol", ["All"] + sorted(syms))
        if sel != "All":
            df = df[df['contract'] == sel]
    else:
        sel = "All"

    # 5) Customer selector
    uidc = next((c for c in ['userid','useruid'] if c in df.columns), None)
    if uidc:
        uids   = sorted(df[uidc].dropna().unique())
        choice = st.selectbox("Customer", ["All"] + uids)
        if choice != "All":
            df = df[df[uidc] == choice]
    else:
        st.warning("No UID column found."); st.stop()

    # 6) Summary metrics
    tfull        = df.copy()
    total_trades = len(tfull)
    tot_profit   = tfull['profit_loss'].sum()
    avg_pl       = tfull['profit_loss'].mean()
    win_rate     = tfull['profit_loss'].gt(0).mean() * 100 if total_trades else 0.0
    sharpe       = compute_sharpe_ratio(df)
    drawdown     = compute_drawdown(df)
    pl_ratio     = compute_pl_ratio(df)
    avg_lev      = tfull.get('leverage', np.nan).mean()
    avg_hold     = tfull['holding'].mean()

    # New: wins & losses count
    total_wins   = int(tfull['profit_loss'].gt(0).sum())
    total_losses = int(tfull['profit_loss'].lt(0).sum())

    # 7) Funding‐fee summaries
    total_fees     = tfull['funding_fee'].sum()
    fees_by_symbol = (
        tfull.groupby('contract')['funding_fee']
             .sum()
             .rename('funding_fees')
             .reset_index()
    )

    # 8) Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Performance", "📈 Benchmark Risk", "⏰ Symbol/Time"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Profit",    f"${tot_profit:,.2f}")
        c2.metric("Average P/L",     f"${avg_pl:,.2f}")
        c3.metric("Win Rate",        f"{win_rate:.2f}%")
        d1, d2, d3 = st.columns(3)
        d1.metric("Sharpe Ratio",    f"{sharpe}")
        d2.metric("Max Drawdown",    f"${drawdown:,.2f}")
        d3.metric("Total Trades",    total_trades)
        if not np.isnan(avg_lev):
            st.metric("Avg Leverage",  f"{avg_lev:.2f}x")
        if not np.isnan(avg_hold):
            st.metric("Avg Hold Time", f"{avg_hold:.1f} min")
        if pl_ratio is not None:
            st.metric("P/L Ratio",     f"{pl_ratio:.2f}")

        # wins & losses
        w1, w2, _ = st.columns(3)
        w1.metric("✅ Total Wins",   total_wins)
        w2.metric("❌ Total Losses", total_losses)

        # Funding fees
        f1, _, _ = st.columns(3)
        f1.metric("💸 Total Funding Fees", f"${total_fees:,.2f}")

        # Leaderboard
        if choice == "All":
            st.subheader("🏅 Top Customers Leaderboard")
            lb = tfull.groupby(uidc).apply(lambda g: pd.Series({
                'Total_Profit':   g['profit_loss'].sum(),
                'Total_Trades':   len(g),
                'Win_Rate':       g['profit_loss'].gt(0).mean() * 100,
                'Avg_PL':         g['profit_loss'].mean(),
                'Sharpe_Ratio':   compute_sharpe_ratio(g)
            }))
            lb = lb.sort_values('Total_Profit', ascending=False).head(20)
            lb[['Win_Rate','Avg_PL','Sharpe_Ratio']] = lb[['Win_Rate','Avg_PL','Sharpe_Ratio']].round(2)
            st.dataframe(lb)

        # Cumulative P/L
        if total_trades:
            st.subheader("Cumulative P/L Over Time")
            df['cum'] = df['profit_loss'].cumsum()
            chart = df.set_index('date')['cum'].resample('1H').mean().ffill()
            st.line_chart(chart)

        # Fees by symbol
        st.subheader("💱 Funding Fees by Symbol")
        st.dataframe(
            fees_by_symbol.assign(funding_fees=lambda d: d['funding_fees'].map(lambda x: f"${x:,.2f}")),
            use_container_width=True
        )

    with tab2:
        st.subheader("Customer Risk Benchmark Scoring")
        norm, sc = calculate_benchmark_score(tfull)
        if norm:
            colour = {"Hold":"green","Monitor":"gold","Transmit":"red"}[norm['risk_label']]
            st.markdown(
                f"**Risk Label:** <span style='color:{colour};'>{norm['risk_label']}</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Benchmark Score:** <span style='color:{colour};'>{norm['benchmark_score']}</span>",
                unsafe_allow_html=True
            )
            st.dataframe(pd.DataFrame(norm, index=["Value"]).T, use_container_width=True)
        else:
            st.warning("No trades to score.")

    with tab3:
        st.subheader("Profit by Day & Hour")
        dcol, hcol = st.columns(2)
        with dcol:
            dayp = df.groupby(df['date'].dt.day_name())['profit_loss'].sum()
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            st.bar_chart(dayp.reindex(order))
        with hcol:
            df['hour'] = df['date'].dt.hour
            hrp = df.groupby('hour')['profit_loss'].sum()
            st.bar_chart(hrp)

        st.subheader("Symbol Breakdown")
        symp = df.groupby('contract')['profit_loss'].sum().sort_values(ascending=False)
        symc = df['contract'].value_counts()
        ca, cb = st.columns(2)
        with ca:
            fig, ax = plt.subplots()
            symp.head(10).plot.barh(ax=ax)
            ax.invert_yaxis()
            st.pyplot(fig)
        with cb:
            fig, ax = plt.subplots()
            symc.head(10).plot.barh(ax=ax)
            ax.invert_yaxis()
            st.pyplot(fig)

        st.subheader("Top 5 Pie Charts")
        p1, p2 = st.columns(2)
        with p1:
            top5 = symp[symp > 0].head(5)
            if len(top5) > 1:
                fig, ax = plt.subplots()
                ax.pie(top5, labels=top5.index, autopct='%1.1f%%')
                ax.axis('equal')
                st.pyplot(fig)
        with p2:
            topc = symc.head(5)
            if len(topc) > 1:
                fig, ax = plt.subplots()
                ax.pie(topc, labels=topc.index, autopct='%1.1f%%')
                ax.axis('equal')
                st.pyplot(fig)

    # PDF download
    if choice != "All":
        pdf_data = generate_pdf_report(
            choice, sel, tot_profit, avg_pl, win_rate,
            sharpe, drawdown, total_trades, pl_ratio,
            total_fees
        )
        st.download_button(
            "⬇️ Download PDF Report",
            pdf_data,
            f"{choice}_{sel}_report.pdf",
            "application/pdf"
        )

    st.write(f"⏱️ Loaded in {time.time() - t0:.2f} seconds")
