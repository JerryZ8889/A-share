import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 页面配置与基础环境
# ==========================================
st.set_page_config(page_title="量化大师-策略融合版", layout="wide")
st.title("🛡️ 量化大师：全量扫描与首阴战法综合看板")

# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载模块
# ==========================================

@st.cache_data(ttl=3600)
def load_index_data():
    """加载指数日线数据"""
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    return df_idx

@st.cache_data(ttl=0)
def load_scan_results():
    """加载 A代码 市场广度结果 (根目录直读)"""
    file_name = "scan_results.csv"
    if not os.path.exists(file_name):
        st.error(f"❌ 未找到 {file_name}，请确保该文件已上传到 GitHub 根目录。")
        st.stop()
    df = pd.read_csv(file_name)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df.set_index('date', inplace=True)
    return df

@st.cache_data(ttl=3600)
def load_master_data():
    """加载 B代码 首阴战法数据 (根目录直读)"""
    # 【核心修改】：删除了 'csi500_data' 文件夹路径，直接读取文件名
    file_name = 'CSI500_Master_Strategy.csv'
    
    if not os.path.exists(file_name):
        st.error(f"❌ 找不到文件 {file_name}。请确认文件已直接上传至 GitHub 仓库根目录。")
        st.stop()
    df = pd.read_csv(file_name, index_col='date', parse_dates=True)
    return df.sort_index()

# 执行加载
try:
    df_idx = load_index_data()
    history_df = load_scan_results()
    df_b = load_master_data()
    
    last_row_a = history_df.iloc[-1]
    curr_ma20 = last_row_a['ma20_ratio']
    curr_nh = last_row_a['new_high_ratio']
    scan_date = history_df.index[-1].strftime('%Y-%m-%d')
    st.success(f"✅ 全量数据同步成功！ 信号日期：{scan_date}")
except Exception as e:
    st.error(f"⚠️ 数据载入失败: {e}")
    st.stop()

# ==========================================
# 2. 逻辑计算中心
# ==========================================

# --- A策略计算 (宽度/热度) ---
vol = df_idx['volume']
z_series = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
curr_z = z_series.iloc[-1]
ma20_idx = df_idx['close'].rolling(20).mean().iloc[-1]
ma60_idx = df_idx['close'].rolling(60).mean().iloc[-1]
is_bull = ma20_idx > ma60_idx

# --- B策略计算 (首阴战法) ---
# 计算指标
df_b['MA5'] = df_b['close'].rolling(window=5).mean()
df_b['MA10'] = df_b['close'].rolling(window=10).mean()
df_b['Is_Up'] = (df_b['close'] > df_b['close'].shift(1)).astype(int)
df_b['Streak'] = df_b['Is_Up'].groupby((df_b['Is_Up'] != df_b['Is_Up'].shift()).cumsum()).cumcount() + 1
df_b['Consec_Gains'] = np.where(df_b['Is_Up'] == 1, df_b['Streak'], 0)

# 提取最新数据
last_b = df_b.iloc[-1]
prev_b = df_b.iloc[-2]

# B-买入判定
b_cond1 = last_b['close'] > last_b['MA10']
b_cond2 = prev_b['Consec_Gains'] >= 2
b_cond3 = last_b['close'] < prev_b['close']
# 换手率单位自适应
t_val = last_b['ETF_Turnover'] if last_b['ETF_Turnover'] > 1 else last_b['ETF_Turnover'] * 100
b_cond4 = t_val > 1.5
b_cond5 = last_b['close'] > last_b['MA5']

b_buy_signal = b_cond1 and b_cond2 and b_cond3 and b_cond4 and b_cond5

# B-卖出判定
recent_3_rets = df_b['close'].pct_change().tail(3)
b_rule_6_sell = (recent_3_rets < 0).all()

# ==========================================
# 3. 布局：数据可视化
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 资金热度 (Z-Score)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_data = z_series.tail(100)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data>=0), color='red', alpha=0.3)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    st.pyplot(fig1)

with col2:
    st.subheader("📊 市场广度趋势 (MA20 %)")
    fig2, ax_l = plt.subplots(figsize=(10, 5))
    ax_l.plot(history_df.index, history_df['ma20_ratio'], color='tab:blue', linewidth=2)
    ax_l.set_ylim(0, 100)
    ax_r = ax_l.twinx()
    ax_r.bar(history_df.index, history_df['new_high_ratio'], color='tab:orange', alpha=0.3)
    st.pyplot(fig2)

# ==========================================
# 4. 动态逻辑看板
# ==========================================
st.divider()
st.subheader("🛡️ 综合决策报告")

m1, m2, m3, m4 = st.columns(4)
m1.metric("指数模式", "多头 (Bull)" if is_bull else "空头 (Bear)")
m2.metric("热度 Z", f"{curr_z:.2f}")
m3.metric("市场宽度", f"{curr_ma20:.1f}%")
m4.metric("中证500换手", f"{t_val:.2f}%")

st.write("---")
# A 策略
st.markdown("#### 🟢 策略A：宽度/热度择时")
buy_a = curr_ma20 < 16
if is_bull:
    sell_a = (curr_ma20 > 79) and (curr_z < 1.5) and (curr_nh < 10)
    sell_msg = "宽度过热且动能枯竭"
else:
    sell_a = (curr_ma20 > 40) and (curr_z < 1.0) and (curr_nh < 25)
    sell_msg = "反抽遇阻"

if buy_a: st.success("🎯 **A建议：买入/补仓** (冰点触发)")
elif sell_a: st.error(f"🚨 **A建议：减仓/清仓** ({sell_msg})")
else: st.info("⌛ **A状态**：中性观望")

# B 策略
st.markdown("#### 🔴 策略B：中证500首阴回踩")
if b_buy_signal:
    st.success("🔥 **B建议：【加仓】** —— 满足首阴回踩逻辑。")
    with st.expander("逻辑详情"):
        st.write(f"- 趋势/连阳/首阴/换手/支撑 全部达标 ✅")
elif b_rule_6_sell:
    st.error("🚨 **B建议：【减仓】** —— 触发连续3日下跌止损。")
else:
    st.info("⌛ **B状态**：未触发信号")

# 综合结论
st.divider()
if buy_a and b_buy_signal:
    st.warning("🚀 **综合结论：重仓共振！** A策略冰点与B策略首阴同时出现。")
elif b_buy_signal:
    st.info("🔎 **综合结论：局部加仓。** 中证500出现短线回踩机会。")
else:
    st.write("✅ **综合结论：保持现状。**")
