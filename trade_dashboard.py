import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setup
st.set_page_config(page_title="Trading Performance Dashboard", layout="wide")
st.title("📊 Compact Trading Dashboard")

uploaded_file = st.file_uploader("Upload your trade CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()
    st.write("🧪 Columns found:", df.columns.tolist())

    uid_column = next((col for col in ['userid', 'useruid'] if col in df.columns), None)
    if uid_column:
        uids = df[uid_column].dropna().unique()
        selected_uid = st.selectbox("👤 Select User ID", sorted(uids))
        df = df[df[uid_column] == selected_uid]
        st.success(f"Showing data for UID: {selected_uid}")
    else:
        st.warning("⚠️ No valid UID column found.")

    if 'profit_loss' in df.columns:
        df['profit_loss'] = pd.to_numeric(df['profit_loss'], errors='coerce')
    elif 'accpnl' in df.columns:
        df['profit_loss'] = pd.to_numeric(df['accpnl'], errors='coerce')
    elif {'entryprice', 'exitprice'}.issubset(df.columns):
        df['profit_loss'] = pd.to_numeric(df['exitprice'], errors='coerce') - pd.to_numeric(df['entryprice'],
                                                                                            errors='coerce')
    else:
        st.error("❌ No profit column found.")
        st.stop()

    date_column = next(
        (col for col in ['date', 'createtime', 'opentime', 'timestamp', 'updatetime'] if col in df.columns), None)
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        df = df.sort_values(by=date_column)
        df['date'] = df[date_column]
    else:
        st.error("❌ No valid date column found.")
        st.stop()

    total_profit = df['profit_loss'].sum()
    average_pl = df['profit_loss'].mean()
    total_trades = len(df)
    winning_trades = len(df[df['profit_loss'] > 0])
    losing_trades = len(df[df['profit_loss'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades else 0

    sharpe = None
    max_dd = None
    try:
        daily_pnl = df.groupby(df['date'].dt.date)['profit_loss'].sum()
        daily_returns = daily_pnl.pct_change().dropna()
        sharpe = (daily_returns.mean() - 0.02 / 252) / daily_returns.std()
    except:
        pass

    try:
        cum_pnl = df['profit_loss'].cumsum()
        peak = cum_pnl.cummax()
        drawdown = cum_pnl - peak
        max_dd = drawdown.min()
    except:
        pass

    st.subheader("📈 Performance Metrics")
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("💰 Total Profit", f"${total_profit:,.2f}")
    mcol2.metric("📊 Average P/L", f"${average_pl:,.2f}")
    mcol3.metric("🏆 Win Rate", f"{win_rate:.2f}%")

    mcol4, mcol5, mcol6 = st.columns(3)
    mcol4.metric("📈 Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")
    mcol5.metric("📉 Max Drawdown", f"${max_dd:,.2f}" if max_dd else "N/A")
    mcol6.metric("🔢 Total Trades", total_trades)

    # ========== Cumulative Profit ==========
    st.subheader("📅 Cumulative Profit Over Time")
    df['cumulative'] = df['profit_loss'].cumsum()
    st.line_chart(df.set_index('date')['cumulative'])

    # ========== Day & Hour Charts Side-by-Side ==========
    st.subheader("⏱️ Time-Based Profit Breakdown")
    day_pnl = df.groupby(df['date'].dt.day_name())['profit_loss'].sum()

    # Optional: timezone shift
    df['hour'] = df['date'].dt.tz_localize('UTC').dt.tz_convert('Europe/Istanbul').dt.hour if df[
                                                                                                  'date'].dt.tz is None else \
    df['date'].dt.hour
    hour_pnl = df.groupby('hour')['profit_loss'].sum()

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.write("📅 Profit by Day of Week")
        st.bar_chart(day_pnl)
    with tcol2:
        st.write("🕒 Profit by Hour")
        st.bar_chart(hour_pnl)

    # ========== Symbol Analysis ==========
    symbol_col = next((col for col in ['symbol', 'pair', 'contract', 'ticker'] if
                       col in df.columns and not pd.api.types.is_numeric_dtype(df[col])), None)
    if symbol_col:
        st.subheader("📊 Symbol-Based Analysis")
        sc1, sc2 = st.columns(2)

        top_symbols = df.groupby(symbol_col)['profit_loss'].sum().sort_values(ascending=False).head(10)
        top_counts = df[symbol_col].value_counts().head(10)

        with sc1:
            st.write("💹 Profit by Symbol")
            st.bar_chart(top_symbols)

        with sc2:
            st.write("📊 Trade Count by Symbol")
            st.bar_chart(top_counts)

        # ========== Pie Charts: Symbol Share & Long/Short ==========
        st.subheader("🥧 Symbol & Position Share")

        pc1, pc2 = st.columns(2)

        with pc1:
            st.write("🔄 Top 5 Symbol Share")
            pie_data = top_counts.head(5)
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.pie(
                pie_data,
                labels=pie_data.index,
                autopct='%1.1f%%',
                startangle=90,
                labeldistance=1.1,
                textprops={'fontsize': 10}
            )
            ax1.axis('equal')
            st.pyplot(fig1)

        # === Symbol Filtered Long vs Short Pie ===
        position_col = next((col for col in ['position', 'side', 'direction'] if col in df.columns), None)
        if position_col:
            with pc2:
                st.write("📐 Long vs Short by Symbol")
                available_symbols = df[symbol_col].dropna().unique()
                selected_symbol = st.selectbox("🔎 Select Symbol", available_symbols)
                df_symbol = df[df[symbol_col] == selected_symbol]
                pos_counts = df_symbol[position_col].value_counts()
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                ax2.pie(
                    pos_counts,
                    labels=pos_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 10}
                )
                ax2.axis('equal')
                st.pyplot(fig2)
    else:
        st.warning("⚠️ No valid symbol column found.")
else:
    st.info("👆 Upload your CSV to begin.")

