import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="量化大师-专业版", layout="wide")
st.title("🛡️ 量化大师：全量扫描结果看板 (C方案)")

# 1. 基础数据 (指数走势依然可以实时抓取，因为只有 1 条请求，不会被封)
@st.cache_data(ttl=3600)
def load_index_data():
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    return df_idx

df_idx = load_index_data()

# 2. 读取你上传的扫描结果
try:
    scan_res = pd.read_csv("scan_results.csv")
    curr_ma20 = scan_res['ma20_ratio'].iloc[0]
    curr_nh = scan_res['new_high_ratio'].iloc[0]
    scan_date = scan_res['date'].iloc[0]
except:
    st.error("未找到扫描结果文件 scan_results.csv，请先在本地运行扫描并上传。")
    st.stop()

# 3. 布局展示
st.info(f"📅 本次体检数据日期：{scan_date} (由本地算力强力驱动)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔥 资金热度 (实时)")
    vol = df_idx['volume']
    z = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.fill_between(z.tail(100).index, z.tail(100), 0, where=(z.tail(100)>=0), color='red', alpha=0.3)
    ax1.fill_between(z.tail(100).index, z.tail(100), 0, where=(z.tail(100)<0), color='blue', alpha=0.3)
    st.pyplot(fig1)

with col2:
    st.header("📝 深度诊断结论")
    st.metric("全量站上 MA20 比例", f"{curr_ma20:.1f}%")
    st.metric("全量创 60日新高比例", f"{curr_nh:.1f}%")
    
    score = 50
    if curr_ma20 > 50: score += 20
    if curr_ma20 > 80 and curr_nh < 2: score -= 30
    st.subheader(f"综合多空分：{score}/100")

    if score > 60: st.success("结论：逻辑共振，维持多头思维。")
    elif score < 40: st.error("结论：高位背离，注意减仓防守。")
    else: st.info("结论：震荡行情，控仓观望。")