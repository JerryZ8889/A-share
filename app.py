import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

# 页面配置
st.set_page_config(page_title="量化大师-全量决策看板", layout="wide")

st.title("🛡️ 量化大师：500只全量扫描 + 决策看板")
st.write("提示：系统会自动加载趋势背景。点击下方按钮可启动针对 500 只成份股的‘深度体检’。")

# ==========================================
# 1. 基础数据准备 (指数与宏观)
# ==========================================
@st.cache_data(ttl=3600)
def load_base_data():
    # 资金热度数据
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    
    # 趋势线背景数据 (采样50只以保证加载速度)
    stock_list_sample = ak.index_stock_cons(symbol="000905")['品种代码'].tolist()[:50]
    ma20_matrix = pd.DataFrame()
    new_high_matrix = pd.DataFrame()
    
    for code in stock_list_sample:
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20250701", adjust="qfq")
            df['date'] = pd.to_datetime(df['日期'])
            df.set_index('date', inplace=True)
            ma20_matrix[code] = (df['收盘'] > df['收盘'].rolling(20).mean()).astype(int)
            new_high_matrix[code] = (df['收盘'] >= df['最高'].rolling(60).max()).astype(int)
        except: continue
        
    hist_breadth = pd.DataFrame({
        'ma20': ma20_matrix.mean(axis=1) * 100,
        'new_high': new_high_matrix.mean(axis=1) * 100
    })
    return df_idx, hist_breadth

df_idx, hist_breadth = load_base_data()

# ==========================================
# 2. 500只全量扫描逻辑 (手动触发)
# ==========================================
@st.cache_resource(show_spinner=False)
def run_full_scan():
    all_stocks = ak.index_stock_cons(symbol="000905")['品种代码'].tolist()
    results = []
    bar = st.progress(0)
    status = st.empty()
    error_count = 0 # 记录失败次数
    
    for i, code in enumerate(all_stocks):
        status.text(f"正在深度扫描 500 指数成份股: {i+1}/500 (失败: {error_count})")
        bar.progress((i + 1) / 500)
        try:
            # 线上环境建议减小 start_date 范围以提高成功率
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20251001", adjust="qfq")
            if df is not None and len(df) >= 20: # 降低门槛进行测试
                c = df['收盘'].iloc[-1]
                m = df['收盘'].rolling(20).mean().iloc[-1]
                h = df['最高'].rolling(60).max().iloc[-1] if len(df) >= 60 else df['最高'].max()
                results.append({'m': 1 if c > m else 0, 'h': 1 if c >= h else 0})
            else:
                error_count += 1
        except:
            error_count += 1
            continue
    
    bar.empty()
    
    # --- 核心修复逻辑 ---
    if not results:
        status.error(f"❌ 扫描完成，但未获取到有效数据。失败次数: {error_count}。可能是云端 IP 被数据源封锁。")
        return 0.0, 0.0  # 返回默认值防止崩盘
    
    status.success(f"✅ 扫描完成！成功: {len(results)}, 失败: {error_count}")
    res = pd.DataFrame(results)
    return res['m'].mean() * 100, res['h'].mean() * 100

# ==========================================
# 3. 布局：双轴看板展示
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 资金热度 (Volume Z-Score)")
    vol = df_idx['volume']
    z = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    
    # 修正：只取最近100天，且明确指定 X 轴为日期
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_data = z.tail(100)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data>=0), color='red', alpha=0.3)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader("📊 市场广度 (60日趋势)")
    fig2, ax_l = plt.subplots(figsize=(10, 5))
    plot_df = hist_breadth.tail(60)
    ax_l.plot(plot_df.index, plot_df['ma20'], color='tab:blue', linewidth=2, label='MA20 Ratio')
    ax_l.set_ylim(0, 100)
    ax_l.set_ylabel('Above MA20 (%)', color='tab:blue')
    
    ax_r = ax_l.twinx()
    ax_r.bar(plot_df.index, plot_df['new_high'], color='tab:orange', alpha=0.5, label='New High')
    ax_r.set_ylabel('New High (%)', color='tab:orange')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# ==========================================
# 4. 触发全量扫描与结论
# ==========================================
st.divider()
curr_ma20, curr_nh = hist_breadth['ma20'].iloc[-1], hist_breadth['new_high'].iloc[-1]

if st.button('🚀 启动今日 500 只全量深度体检'):
    curr_ma20, curr_nh = run_full_scan()
    st.balloons()

# 自动研报
st.header("📝 最终诊断结论")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("站上 MA20 比例", f"{curr_ma20:.1f}%")
    st.write("反映市场整体水位")
with c2:
    st.metric("创 60日新高比例", f"{curr_nh:.1f}%")
    st.write("反映领头羊赚钱效应")
with c3:
    score = 50
    if curr_ma20 > 50: score += 20
    if curr_ma20 > 80 and curr_nh < 2: score -= 30
    st.metric("综合多空评分", f"{score}/100")

if score > 60:
    st.success("结论：行情健康，建议持仓。")
elif score < 40:
    st.error("结论：逻辑背离或走弱，建议防御。")
else:
    st.info("结论：震荡行情，控仓观望。")