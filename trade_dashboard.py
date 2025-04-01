import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def check_password():
    def password_entered():
        if st.session_state["password"] == "mydashboard123":
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input("🔐 Enter password to continue:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["authenticated"]:
        st.text_input("🔐 Enter password to continue:", type="password", on_change=password_entered, key="password")
        st.error("❌ Incorrect password")
        return False
    else:
        return True

@st.cache_data
def compute_sharpe_ratio(df):
    daily_pnl = df.groupby(df['date'].dt.date)['profit_loss'].sum()
    daily_returns = daily_pnl.pct_change().dropna()
    risk_free_rate = 0.02 / 252
    if daily_returns.std() == 0:
        return 0
    sharpe = (daily_returns.mean() - risk_free_rate) / daily_returns.std()
    return round(sharpe, 2)

@st.cache_data
def compute_drawdown(df):
    cum_pnl = df['profit_loss'].cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    return round(drawdown.min(), 2)

@st.cache_data
def get_top_symbols(df, symbol_col):
    return df[symbol_col].value_counts().head(5)

if check_password():
    st.set_page_config(page_title="Trading Dashboard", layout="wide")
    st.title("📊 Ultimate Final Trading Dashboard")

    uploaded_file = st.file_uploader("📁 Upload trade CSV", type=["csv"])

    if uploaded_file:
        start = time.time()
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower()

        minimal_view = st.checkbox("🧹 Show minimal view (summary only)", value=False)

        # -------- User filter --------
        uid_column = next((col for col in ['userid', 'useruid'] if col in df.columns), None)
        if uid_column:
            uids = df[uid_column].dropna().unique().tolist()
            selected_uid = st.selectbox("👤 Select User ID", sorted(uids))
            df = df[df[uid_column] == selected_uid]
            st.success(f"Showing data for UID: {selected_uid}")
        else:
            st.warning("⚠️ No UID column found.")

        # -------- Profit calculation --------
        df['profit_loss'] = 0.0
        if 'accpnl' in df.columns:
            df['accpnl'] = pd.to_numeric(df['accpnl'], errors='coerce').fillna(0)
            df['profit_loss'] = df['accpnl']

        if {'entryprice', 'exitprice'}.issubset(df.columns):
            df['entryprice'] = pd.to_numeric(df['entryprice'], errors='coerce')
            df['exitprice'] = pd.to_numeric(df['exitprice'], errors='coerce')
            calc_pnl = (df['exitprice'] - df['entryprice']).fillna(0)
            df.loc[df['profit_loss'] == 0, 'profit_loss'] = calc_pnl[df['profit_loss'] == 0]

        # -------- Date handling --------
        date_column = next((col for col in ['date', 'createtime', 'opentime', 'timestamp', 'updatetime'] if col in df.columns), None)
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df['date'] = df[date_column].fillna(pd.Timestamp("1970-01-01"))
        else:
            st.error("❌ No valid date column found.")
            st.stop()

        # Separate full dataset vs chart-safe dataset
        df_full = df.copy()
        df = df[df['date'] > pd.Timestamp("2000-01-01")]

        # -------- Metrics --------
        total_profit = df_full['profit_loss'].sum()
        average_pl = df_full['profit_loss'].sum() / len(df_full)
        total_trades = len(df_full)
        winning_trades = len(df_full[df_full['profit_loss'] > 0])
        losing_trades = len(df_full[df_full['profit_loss'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades else 0

        sharpe = compute_sharpe_ratio(df)
        max_dd = compute_drawdown(df)

        with st.expander("🧪 Debug Info", expanded=False):
            st.write(f"📊 Rows used: {len(df_full)}")
            st.write(f"🧮 Profit sum: {total_profit:,.2f}")
            st.write(f"📈 Average P/L: {average_pl:,.2f}")

        # -------- Performance Metrics --------
        st.subheader("📈 Performance Metrics")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("💰 Total Profit", f"${total_profit:,.2f}")
        mcol2.metric("📊 Average P/L", f"${average_pl:,.2f}")
        mcol3.metric("🏆 Win Rate", f"{win_rate:.2f}%")

        mcol4, mcol5, mcol6 = st.columns(3)
        mcol4.metric("📈 Sharpe Ratio", f"{sharpe}")
        mcol5.metric("📉 Max Drawdown", f"${max_dd:,.2f}")
        mcol6.metric("🔢 Total Trades", total_trades)

        if not minimal_view:
            st.subheader("📅 Cumulative Profit Over Time")
            df['cumulative'] = df['profit_loss'].cumsum()
            st.line_chart(df.set_index('date')['cumulative'])

            st.subheader("⏱️ Time-Based Profit Breakdown")
            day_pnl = df.groupby(df['date'].dt.day_name())['profit_loss'].sum()
            df['hour'] = df['date'].dt.hour
            hour_pnl = df.groupby('hour')['profit_loss'].sum()

            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.write("📅 Profit by Day of Week")
                st.bar_chart(day_pnl)
            with tcol2:
                st.write("🕒 Profit by Hour")
                st.bar_chart(hour_pnl)

            # -------- Symbol Charts + Pies --------
            symbol_col = next((col for col in ['symbol', 'pair', 'contract', 'ticker'] if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])), None)

            if symbol_col:
                st.subheader("📊 Symbol Performance")

                symbol_profit = df.groupby(symbol_col)['profit_loss'].sum().sort_values(ascending=False)
                symbol_count = df[symbol_col].value_counts()

                st.bar_chart(symbol_profit.head(10))
                st.bar_chart(symbol_count.head(10))

                # 🔁 PIE CHARTS: Top 5 Profit & Count
                st.subheader("🥧 Top 5 Symbol Breakdown")
                pie1, pie2 = st.columns(2)

                with pie1:
                    st.markdown("**💰 By Profit**")
                    top_profit = symbol_profit.head(5)
                    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
                    ax1.pie(
                        top_profit,
                        labels=top_profit.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        textprops={'fontsize': 9}
                    )
                    ax1.axis('equal')
                    st.pyplot(fig1)

                with pie2:
                    st.markdown("**📊 By Trade Count**")
                    top_count = symbol_count.head(5)
                    fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
                    ax2.pie(
                        top_count,
                        labels=top_count.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        textprops={'fontsize': 9}
                    )
                    ax2.axis('equal')
                    st.pyplot(fig2)

        st.write(f"⏱️ Loaded in {time.time() - start:.2f} seconds")

    else:
        st.info("📁 Please upload your trade CSV to get started.")
