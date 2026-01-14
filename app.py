import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(page_title="量化大师-专业版", layout="wide")
st.title("🛡️ 量化大师：全量扫描结果看板 (C方案)")

# 1. 基础数据加载 (指数走势实时抓取)
@st.cache_data(ttl=3600)
def load_index_data():
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    return df_idx

df_idx = load_index_data()

# 2. 读取累积的扫描结果
try:
    history_df = pd.read_csv("scan_results.csv")
    
    # 核心数据清洗：强制转日期并扔掉空行
    history_df['date'] = pd.to_datetime(history_df['date'], errors='coerce')
    history_df = history_df.dropna(subset=['date']).sort_values('date')
    history_df.set_index('date', inplace=True)
    
    # 获取最新数据用于展示
    last_row = history_df.iloc[-1]
    curr_ma20 = last_row['ma20_ratio']
    curr_nh = last_row['new_high_ratio']
    scan_date = history_df.index[-1].strftime('%Y-%m-%d')
    update_time = f" | 更新时间：{last_row['update_time']}" if 'update_time' in last_row else ""
    
    # --- 顶部的成功提示框 (确保对齐) ---
    st.success(f"✅ 深度扫描数据同步成功！ 数据日期：{scan_date}{update_time}")
    
except Exception as e:
    st.error(f"⚠️ 数据同步中或格式有误。 详情: {e}")
    st.stop()

# 3. 布局：左右双图
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 资金热度 (Z-Score)")
    vol = df_idx['volume']
    z = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_data = z.tail(100)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data>=0), color='red', alpha=0.3)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader("📊 市场广度 (全量历史趋势)")
    fig2, ax_l = plt.subplots(figsize=(10, 5))
    # 绘制站上 MA20 比例
    ax_l.plot(history_df.index, history_df['ma20_ratio'], color='tab:blue', marker='o', linewidth=2, label='MA20 %')
    ax_l.set_ylim(0, 100)
    ax_l.set_ylabel('Above MA20 (%)', color='tab:blue')
    # 绘制新高比例
    ax_r = ax_l.twinx()
    ax_r.bar(history_df.index, history_df['new_high_ratio'], color='tab:orange', alpha=0.4)
    ax_r.set_ylabel('New High (%)', color='tab:orange')
    plt.xticks(rotation=45)
    fig2.tight_layout()
    st.pyplot(fig2)

# 4. 底部诊断结论
st.divider()
score = 50
if curr_ma20 > 50: score += 20
if curr_ma20 > 80 and curr_nh < 2: score -= 30
st.subheader(f"综合多空分：{score}/100")

if score > 60: st.success("结论：逻辑共振，维持多头思维。")
elif score < 40: st.error("结论：高位背离，注意减仓防守。")
else: st.info("结论：震荡行情，控仓观望。")
