# 🔐 PASSWORD PROTECTION, METRICS, PDF, FULL DASHBOARD

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import time

# ---------- PASSWORD ----------
def check_password():
    def password_entered():
        if st.session_state["password"] == "mydashboard123":
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False
    if "authenticated" not in st.session_state:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["authenticated"]:
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        return False
    else:
        return True

# ---------- METRICS ----------
@st.cache_data
def compute_sharpe_ratio(df):
    daily_pnl = df.groupby(df['date'].dt.date)['profit_loss'].sum()
    daily_returns = daily_pnl.pct_change().dropna()
    risk_free = 0.02 / 252
    return round((daily_returns.mean() - risk_free) / daily_returns.std(), 2) if daily_returns.std() != 0 else 0

@st.cache_data
def compute_drawdown(df):
    cum_pnl = df['profit_loss'].cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    return round(drawdown.min(), 2)

def compute_pl_ratio(df):
    winners = df[df['profit_loss'] > 0]['profit_loss']
    losers = df[df['profit_loss'] < 0]['profit_loss']
    if not winners.empty and not losers.empty:
        return round(winners.mean() / abs(losers.mean()), 2)
    return None

# ---------- PDF GENERATION ----------
def generate_pdf_report(user, symbol, total_profit, avg_pl, win_rate, sharpe, drawdown, trade_count, pl_ratio):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="DarkEx Trading Performance Dashboard", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"User ID: {user}", ln=True)
    pdf.cell(200, 10, txt=f"Symbol: {symbol}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Total Profit: ${total_profit:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Average P/L: ${avg_pl:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Win Rate: {win_rate:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Sharpe Ratio: {sharpe}", ln=True)
    pdf.cell(200, 10, txt=f"Max Drawdown: ${drawdown:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Total Trades: {trade_count}", ln=True)
    if pl_ratio is not None:
        pdf.cell(200, 10, txt=f"P/L Ratio: {pl_ratio:.2f}", ln=True)
    return pdf.output(dest='S').encode('latin1')

# ---------- MAIN APP ----------
if check_password():
    st.set_page_config(page_title="DarkEx Dashboard", layout="wide")
    st.title("DarkEx Trading Performance Dashboard")

    uploaded_file = st.file_uploader("\U0001F4C1 Upload CSV", type=["csv"])
    if uploaded_file:
        start = time.time()
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower()

        minimal_view = st.checkbox("Show minimal view (summary only)", value=False)

        if 'contract' in df.columns:
            df['contract'] = df['contract'].astype(str)

        uid_column = next((col for col in ['userid', 'useruid'] if col in df.columns), None)
        if uid_column:
            uids = sorted(df[uid_column].dropna().unique().tolist())
            selected_uid = st.selectbox("Select User ID", ["All Customers"] + uids)
            if selected_uid != "All Customers":
                df = df[df[uid_column] == selected_uid]
                st.success(f"Showing data for UID: {selected_uid}")
            else:
                st.info("Showing performance for ALL customers (metrics only)")
        else:
            st.warning("No UID column found.")
            st.stop()

        df['profit_loss'] = 0.0
        if 'accpnl' in df.columns:
            df['profit_loss'] = pd.to_numeric(df['accpnl'], errors='coerce').fillna(0)
        if {'entryprice', 'exitprice'}.issubset(df.columns):
            df['entryprice'] = pd.to_numeric(df['entryprice'], errors='coerce')
            df['exitprice'] = pd.to_numeric(df['exitprice'], errors='coerce')
            fallback = (df['exitprice'] - df['entryprice']).fillna(0)
            df.loc[df['profit_loss'] == 0, 'profit_loss'] = fallback[df['profit_loss'] == 0]

        if {'creattime', 'updatetime'}.issubset(df.columns):
            df['creattime'] = pd.to_datetime(df['creattime'], errors='coerce')
            df['updatetime'] = pd.to_datetime(df['updatetime'], errors='coerce')
            df['holding_time'] = (df['updatetime'] - df['creattime']).dt.total_seconds() / 60
        else:
            df['holding_time'] = np.nan

        date_column = next((col for col in ['date', 'creattime', 'timestamp', 'updatetime'] if col in df.columns), None)
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df['date'] = df[date_column].fillna(pd.Timestamp("1970-01-01"))

        st.subheader("\U0001F4C5 Filter by Date")
        start_date, end_date = st.date_input("Select date range", value=[df['date'].min(), df['date'].max()])
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

        st.info(f"Showing trades from **{start_date}** to **{end_date}** — Total trades: **{len(df)}**")

        symbol_col = 'contract'
        symbols = df[symbol_col].dropna().unique().tolist()
        selected_symbol = st.selectbox("Select Symbol", ["All Symbols"] + sorted(symbols))
        if selected_symbol != "All Symbols":
            df = df[df[symbol_col] == selected_symbol]
            st.success(f"Filtered by Symbol: {selected_symbol}")
        else:
            selected_symbol = "All Symbols"

        if 'leverage' in df.columns:
            df['leverage'] = pd.to_numeric(df['leverage'], errors='coerce')
        else:
            df['leverage'] = np.nan

        avg_leverage = df['leverage'].dropna().mean()
        avg_holding = df['holding_time'].dropna().mean()
        pl_ratio = compute_pl_ratio(df)

        df_full = df.copy()
        df = df[df['date'] > pd.Timestamp("2000-01-01")]

        total_profit = df_full['profit_loss'].sum()
        average_pl = df_full['profit_loss'].mean()
        total_trades = len(df_full)
        win_rate = (df_full['profit_loss'] > 0).sum() / total_trades * 100 if total_trades else 0
        sharpe = compute_sharpe_ratio(df)
        drawdown = compute_drawdown(df)

        st.subheader("\U0001F4C8 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("\U0001F4B0 Total Profit", f"${total_profit:,.2f}")
        col2.metric("\U0001F4CA Average P/L", f"${average_pl:,.2f}")
        col3.metric("\U0001F3C6 Win Rate", f"{win_rate:.2f}%")
        col4, col5, col6 = st.columns(3)
        col4.metric("\U0001F4C8 Sharpe Ratio", f"{sharpe}")
        col5.metric("\U0001F4C9 Max Drawdown", f"${drawdown:,.2f}")
        col6.metric("\U0001F522 Total Trades", total_trades)
        if not pd.isna(avg_leverage):
            st.metric("\U0001F4CF Avg Leverage", f"{avg_leverage:.2f}x")
        if not pd.isna(avg_holding):
            st.metric("⏱️ Avg Holding Time", f"{avg_holding:.1f} min")
        if pl_ratio is not None:
            st.metric("⚖️ P/L Ratio", f"{pl_ratio:.2f}")

        # ✅ Leaderboard (restored)
        if selected_uid == "All Customers" and uid_column:
            st.subheader("\U0001F3C6 Top Customers Leaderboard")
            leaderboard = df_full.groupby(df_full[uid_column]).agg(
                Total_Profit=('profit_loss', 'sum'),
                Total_Trades=('profit_loss', 'count'),
                Win_Rate=('profit_loss', lambda x: (x > 0).sum() / len(x) * 100)
            ).sort_values(by='Total_Profit', ascending=False)
            st.dataframe(leaderboard.head(20))

        # ✅ Graphs restored
        if selected_uid != "All Customers" and not minimal_view:
            tab1, tab2, tab3 = st.tabs(["\U0001F4C8 Summary", "\U0001F551 Time Analysis", "\U0001F967 Symbol Breakdown"])
            with tab1:
                st.subheader("Cumulative Profit Over Time")
                df['cumulative'] = df['profit_loss'].cumsum()
                chart_df = df.set_index('date')[['cumulative']].resample('1H').mean().dropna()
                chart_df['cumulative'] = chart_df['cumulative'].ffill()
                st.line_chart(chart_df['cumulative'])
            with tab2:
                st.subheader("Profit by Day & Hour")
                col_day, col_hour = st.columns(2)
                with col_day:
                    day_pnl = df.groupby(df['date'].dt.day_name())['profit_loss'].sum()
                    st.bar_chart(day_pnl)
                with col_hour:
                    df['hour'] = df['date'].dt.hour
                    hour_pnl = df.groupby('hour')['profit_loss'].sum()
                    st.bar_chart(hour_pnl)
            with tab3:
                st.subheader("Symbol Performance")
                symbol_profit = df.groupby(symbol_col)['profit_loss'].sum().sort_values(ascending=False)
                symbol_count = df[symbol_col].value_counts()
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.markdown("### \U0001F4B0 Profit by Symbol")
                    fig1, ax1 = plt.subplots(figsize=(5, 3))
                    symbol_profit.head(10).plot(kind='barh', ax=ax1, color='skyblue')
                    ax1.set_xlabel("Profit")
                    ax1.invert_yaxis()
                    st.pyplot(fig1)
                with chart_col2:
                    st.markdown("### \U0001F4CA Trade Count by Symbol")
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    symbol_count.head(10).plot(kind='barh', ax=ax2, color='lightgreen')
                    ax2.set_xlabel("Trades")
                    ax2.invert_yaxis()
                    st.pyplot(fig2)
                st.subheader("Top 5 Symbol Breakdown")
                pie1, pie2 = st.columns(2)
                with pie1:
                    top_profit = symbol_profit[symbol_profit > 0].head(5)
                    if len(top_profit) > 1:
                        fig3, ax3 = plt.subplots(figsize=(3.5, 3.5))
                        ax3.pie(top_profit, labels=top_profit.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
                        ax3.axis('equal')
                        st.pyplot(fig3)
                    elif len(top_profit) == 1:
                        st.info(f"Only one profitable symbol: {top_profit.index[0]}")
                    else:
                        st.warning("No profitable symbols to display.")
                with pie2:
                    top_count = symbol_count.head(5)
                    if len(top_count) > 1:
                        fig4, ax4 = plt.subplots(figsize=(3.5, 3.5))
                        ax4.pie(top_count, labels=top_count.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
                        ax4.axis('equal')
                        st.pyplot(fig4)
                    elif len(top_count) == 1:
                        st.info(f"Only one traded symbol: {top_count.index[0]}")
                    else:
                        st.warning("No trades to show.")

        if selected_uid != "All Customers":
            st.subheader("\U0001F4E5 Download PDF Report")
            pdf_bytes = generate_pdf_report(
                user=selected_uid,
                symbol=selected_symbol,
                total_profit=total_profit,
                avg_pl=average_pl,
                win_rate=win_rate,
                sharpe=sharpe,
                drawdown=drawdown,
                trade_count=total_trades,
                pl_ratio=pl_ratio
            )
            st.download_button(
                label="⬇️ Download PDF Summary",
                data=pdf_bytes,
                file_name=f"{selected_uid}_{selected_symbol}_report.pdf".replace(" ", "_"),
                mime="application/pdf"
            )

        st.write(f"⏱️ Loaded in {time.time() - start:.2f} seconds")
    else:
        st.info("\U0001F4C1 Upload your trade CSV to begin.")