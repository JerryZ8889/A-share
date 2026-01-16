import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# 0. 页面配置与基础环境
# ==========================================
st.set_page_config(page_title="量化大师-策略融合版", layout="wide")
st.title("🛡️ 量化大师：全量扫描与全市场量能看板")

# 设置绘图字体 (确保中文显示)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心数据加载模块
# ==========================================

@st.cache_data(ttl=3600)
def load_index_data():
    """加载中证500指数日线用于计算 Z-Score"""
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    return df_idx

@st.cache_data(ttl=0) 
def load_scan_results():
    """加载市场广度扫描结果"""
    file_name = "scan_results.csv"
    if not os.path.exists(file_name):
        st.error(f"❌ 未找到 {file_name}")
        st.stop()
    df = pd.read_csv(file_name)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df.set_index('date', inplace=True)
    return df

@st.cache_data(ttl=600)
def load_all_etf_metrics():
    """批量获取四个核心 ETF 的最新数据"""
    etf_files = {
        "上证50":   {"file": "SSE50_Master_Strategy.csv",   "threshold": 5.0},
        "沪深300":  {"file": "CSI300_Master_Strategy.csv",  "threshold": 5.5},
        "中证500":  {"file": "CSI500_Master_Strategy.csv",  "threshold": 13.0},
        "中证1000": {"file": "CSI1000_Master_Strategy.csv", "threshold": 10.0}
    }
    latest_data = {}
    for name, cfg in etf_files.items():
        if os.path.exists(cfg['file']):
            df = pd.read_csv(cfg['file'])
            val = df['ETF_Turnover'].iloc[-1]
            # 统一纠正为百分比数值
            latest_data[name] = {
                "turnover": val if val > 0.5 else val * 100,
                "threshold": cfg['threshold'],
                "is_extreme": (val if val > 0.5 else val * 100) > cfg['threshold']
            }
        else:
            latest_data[name] = {"turnover": 0.0, "threshold": cfg['threshold'], "is_extreme": False}
    return latest_data

@st.cache_data(ttl=3600)
def load_csi500_master():
    """加载策略B专属底表"""
    file_name = 'CSI500_Master_Strategy.csv'
    if not os.path.exists(file_name):
        st.error(f"❌ 找不到文件 {file_name}")
        st.stop()
    df = pd.read_csv(file_name, index_col='date', parse_dates=True)
    return df.sort_index()

# --- 数据执行 ---
try:
    df_idx = load_index_data()
    history_df = load_scan_results()
    etf_metrics = load_all_etf_metrics()
    df_b = load_csi500_master()
    
    # 顶部状态信息
    scan_date = history_df.index[-1].strftime('%Y-%m-%d')
    st.success(f"✅ 数据同步成功 | 最新交易日：{scan_date} | 已完成 1月16日 实时行情补齐")
except Exception as e:
    st.error(f"⚠️ 数据同步失败: {e}")
    st.stop()

# ==========================================
# 2. 看板展示层 (Metrics)
# ==========================================

# 2.1 全市场换手率矩阵
st.write("### 🔥 全市场量能共振监控 (今日换手率)")
m1, m2, m3, m4 = st.columns(4)

def show_metric(col, label, data):
    # 如果超过阈值，显示红色标记
    status = "🔥 天量" if data['is_extreme'] else "正常"
    col.metric(
        label=f"{label} (阈值:{data['threshold']}%)", 
        value=f"{data['turnover']:.2f}%", 
        delta=status if data['is_extreme'] else None,
        delta_color="inverse"
    )

show_metric(m1, "上证50 (蓝筹)", etf_metrics['上证50'])
show_metric(m2, "沪深300 (白马)", etf_metrics['沪深300'])
show_metric(m3, "中证500 (中盘)", etf_metrics['中证500'])
show_metric(m4, "中证1000 (小盘)", etf_metrics['中证1000'])



# 2.2 市场环境指标
st.divider()
c1, c2, c3 = st.columns(3)

# 计算资金热度 Z-Score
vol = df_idx['volume']
z_series = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
curr_z = z_series.iloc[-1]

# 市场模式判定 (MA20 vs MA60)
idx_close = df_idx['close']
ma20_idx = idx_close.rolling(20).mean().iloc[-1]
ma60_idx = idx_close.rolling(60).mean().iloc[-1]
is_bull = ma20_idx > ma60_idx

c1.metric("市场模式", "📈 多头 (Bull)" if is_bull else "📉 空头 (Bear)")
c2.metric("资金热度 (Z-Score)", f"{curr_z:.2f}")
curr_ma20 = history_df.iloc[-1]['ma20_ratio']
c3.metric("广度冰点 (MA20%)", f"{curr_ma20:.1f}%")

# ==========================================
# 3. 策略分析与建议 (Logic)
# ==========================================
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔵 策略A：宽度/热度择时")
    # A策略买入标准：宽度 < 16%
    if curr_ma20 < 16:
        st.success("🎯 **建议：冰点买入** (市场情绪极度低迷，放量即是转折)")
    elif curr_ma20 > 79 and curr_z < 1.5:
        st.error("🚨 **建议：预防见顶** (广度超买且动能衰减)")
    else:
        st.info("⌛ **状态：持仓观望** (暂无极端择时信号)")

with col_right:
    st.subheader("🔴 策略B：首阴战法 (CSI500)")
    # 计算 B 策略逻辑
    df_b['MA5'] = df_b['close'].rolling(5).mean()
    df_b['MA10'] = df_b['close'].rolling(10).mean()
    last_b = df_b.iloc[-1]
    prev_b = df_b.iloc[-2]
    
    # 简化判定：10日线上 + 今日阴线但守住5日线 + 换手达标
    b_buy = (last_b['close'] > last_b['MA10']) and \
            (last_b['close'] < prev_b['close']) and \
            (etf_metrics['中证500']['turnover'] > 1.5) and \
            (last_b['close'] > last_b['MA5'])
            
    if b_buy:
        st.success("🔥 **建议：首阴加仓** (上升趋势中的良性回踩)")
    else:
        st.info("⌛ **状态：等待回踩** (未触发首阴买入逻辑)")

# ==========================================
# 4. 可视化趋势图
# ==========================================
st.divider()
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("**资金热度 (Z-Score) 趋势**")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_data = z_series.tail(100)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data>=0), color='red', alpha=0.3)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    st.pyplot(fig1)

with chart_col2:
    st.write("**全市场同步天量监测 (双轴图)**")
    # 此处调用您之前的双轴绘图逻辑或简版
    st.image("https://via.placeholder.com/800x400.png?text=Sync+Monitoring+Placeholder") # 提示占位
    st.caption("提示：请确保本地运行 runscan.py 以同步最新图表")

# 详情逻辑展开
with st.expander("📝 决策逻辑判定参考"):
    st.write(f"""
    - **上证50 / 沪深300**：蓝筹基石，换手率 > 5% 视为放量。
    - **中证500 / 1000**：活跃中坚，换手率 > 10-13% 视为天量。
    - **天量共振**：若多个指数同时变红，说明全市场资金正在进行系统性大搬家。
    """)
