import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(page_title="量化大师-专业版", layout="wide")
st.title("🛡️ 量化大师：全量扫描结果看板 (C方案)")

# 1. 基础数据 (指数走势实时抓取，单次请求无封禁风险)
@st.cache_data(ttl=3600)
def load_index_data():
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    return df_idx

df_idx = load_index_data()

# 2. 读取累积的扫描结果 (由本地 local_scan.py 生成并上传)
try:
    history_df = pd.read_csv("scan_results.csv")
    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df.set_index('date', inplace=True)
    
    # 获取最新一天的数值用于展示
    curr_ma20 = history_df['ma20_ratio'].iloc[-1]
    curr_nh = history_df['new_high_ratio'].iloc[-1]
    scan_date = history_df.index[-1].strftime('%Y-%m-%d')
except Exception as e:
    st.error(f"数据加载失败: 请确保 scan_results.csv 已上传至 GitHub。错误: {e}")
    st.stop()

# 3. 布局：左右双图
st.info(f"📅 本次体检数据日期：{scan_date} (由本地算力强力驱动)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 资金热度 (Z-Score)")
    vol = df_idx['volume']
    # 计算 Z-Score: (当前值 - 60日均值) / 60日标准差
    z = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_data = z.tail(100) # 显示最近100个交易日
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data>=0), color='red', alpha=0.3)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--', label='1.5 警戒线')
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader("📊 市场广度 (全量历史趋势)")
    fig2, ax_l = plt.subplots(figsize=(10, 5))
    
    # 绘制站上 MA20 比例趋势
    ax_l.plot(history_df.index, history_df['ma20_ratio'], color='tab:blue', marker='o', linewidth=2, label='MA20 %')
    ax_l.set_ylim(0, 100)
    ax_l.set_ylabel('Above MA20 (%)', color='tab:blue')
    
    # 绘制 60日新高 比例柱状图
    ax_r = ax_l.twinx()
    ax_r.bar(history_df.index, history_df['new_high_ratio'], color='tab:orange', alpha=0.4, label='New High %')
    ax_r.set_ylabel('New High (%)', color='tab:orange')
    
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# 4. 底部诊断结论
st.divider()
    
score = 50
if curr_ma20 > 50: score += 20
if curr_ma20 > 80 and curr_nh < 2: score -= 30  # 背离扣分

st.subheader(f"综合多空评分：{score}/100")

if score > 60: 
    st.success("【结论】逻辑共振：多头情绪浓厚，建议维持高仓位。")
elif score < 40: 
    st.error("【结论】逻辑风险：广度与动能背离或严重超买，注意防范回撤。")
else: 
    st.info("【结论】震荡行情：市场进入存量博弈，建议控仓观察。")