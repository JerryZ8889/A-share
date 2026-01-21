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
st.set_page_config(page_title="量化大师-旗舰进化版", layout="wide")
st.title("🛡️ 量化大师：MA30过滤旗舰进化版综合看板")

# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心数据加载模块 (使用你原来的结构)
# ==========================================

@st.cache_data(ttl=0)
def load_index_data():
    """1. 加载指数日线数据 ( sh000905 )"""
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    return df_idx

@st.cache_data(ttl=60)
def load_scan_results():
    """2. 加载市场广度结果 (scan_results.csv)"""
    file_name = "scan_results.csv"
    if not os.path.exists(file_name):
        st.error(f"❌ 未找到 {file_name}")
        st.stop()
    df = pd.read_csv(file_name)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df.set_index('date', inplace=True)
    return df

@st.cache_data(ttl=60)
def load_master_data():
    """3. 加载策略主数据 (CSI500_Master_Strategy.csv)"""
    file_name = 'CSI500_Master_Strategy.csv'
    if not os.path.exists(file_name):
        st.error(f"❌ 找不到文件 {file_name}")
        st.stop()
    df = pd.read_csv(file_name, index_col='date', parse_dates=True)
    return df.sort_index()

@st.cache_data(ttl=60)
def get_summary_turnovers():
    """4. 从汇总表获取实时换手率"""
    file_name = "master_summary.csv"
    turnovers = {"SSE50": 0.0, "CSI300": 0.0, "CSI500": 0.0, "CSI1000": 0.0}
    if os.path.exists(file_name):
        df_sum = pd.read_csv(file_name)
        for _, row in df_sum.iterrows():
            label = row['Index_Label']
            val = row['ETF_Turnover']
            turnovers[label] = val if val > 1 else val * 100
    return turnovers

# --- 执行数据加载 ---
try:
    df_idx = load_index_data()
    history_df = load_scan_results()
    df_master = load_master_data()
    all_turnovers = get_summary_turnovers()
    
    # 顶部状态显示
    last_scan = history_df.iloc[-1]
    curr_ma20 = last_scan['ma20_ratio']
    scan_date = history_df.index[-1].strftime('%Y-%m-%d')
    st.success(f"✅ 数据同步成功！最新数据日期：{scan_date}")
except Exception as e:
    st.error(f"⚠️ 数据同步失败: {e}")
    st.stop()

# ==========================================
# 2. 旗舰进化逻辑计算引擎 (核心逻辑)
# ==========================================
def calculate_flagship_signals(df_price, df_breadth):
    df = df_price.copy()
    # 计算均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA30'] = df['close'].rolling(30).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 合并广度数据
    df = df.join(df_breadth[['ma20_ratio', 'new_high_ratio']], how='left').ffill()
    
    # 计算热度 Z-Score (基于指数成交量)
    vol = df_idx['volume']
    idx_heat_z = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    df['Heat_Z'] = idx_heat_z
    
    # 计算连阳
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Consec_Gains'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Consec_Gains'], 0)
    
    # 信号生成
    df['signal'] = 0  # 1:买, -1:卖
    df['logic_type'] = ""
    in_pos, logic_state, entry_high, hold_days = False, "", 0, 0

    for i in range(1, len(df)):
        curr, prev = df.iloc[i], df.iloc[i-1]
        t_val = curr['ETF_Turnover'] if curr['ETF_Turnover'] > 1 else curr['ETF_Turnover'] * 100
        
        if in_pos:
            hold_days += 1
            is_overheat = (curr['ma20_ratio'] > 79) and (curr['Heat_Z'] < 1.5)
            exit_flag = False
            if logic_state == "Strategic":
                if is_overheat: exit_flag = True
            else: # Tactical
                is_below_ma30 = curr['close'] < curr['MA30']
                if is_overheat or (is_below_ma30 and (curr['close'] < prev['close'] or (hold_days >= 5 and curr['close'] < entry_high))):
                    exit_flag = True
            if exit_flag:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                in_pos, logic_state = False, ""
        else:
            if curr['ma20_ratio'] < 16:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Strategic"
                in_pos, logic_state, hold_days = True, "Strategic", 0
            elif (curr['close'] > curr['MA30'] and curr['close'] > curr['MA10'] and curr['close'] > curr['MA5'] and 
                  prev['Consec_Gains'] >= 3 and curr['close'] < prev['close'] and t_val > 1.0):
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Tactical"
                in_pos, logic_state, hold_days, entry_high = True, "Tactical", 0, curr['high']
    return df

# 执行计算
df_final = calculate_flagship_signals(df_master, history_df)
last_row = df_final.iloc[-1]

# ==========================================
# 3. 布局：左右双图 (原面板布局)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 资金热度 (Z-Score)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_data = df_final['Heat_Z'].tail(100)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data>=0), color='red', alpha=0.3)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader("📊 市场广度 (全量历史趋势)")
    fig2, ax_l = plt.subplots(figsize=(10, 5))
    ax_l.plot(history_df.index, history_df['ma20_ratio'], color='tab:blue', marker='o', linewidth=2, label='MA20 %')
    ax_l.set_ylim(0, 100)
    ax_l.set_ylabel('Above MA20 (%)', color='tab:blue')
    ax_r = ax_l.twinx()
    ax_r.bar(history_df.index, history_df['new_high_ratio'], color='tab:orange', alpha=0.4)
    ax_r.set_ylabel('New High (%)', color='tab:orange')
    plt.xticks(rotation=45)
    fig2.tight_layout()
    st.pyplot(fig2)

# ==========================================
# 4. 诊断报告看板 (原看板内容)
# ==========================================
st.divider()
st.subheader("🛡️ 动态逻辑诊断报告")

m1, m2, m3 = st.columns(3)
# 修复 KeyError，确保使用 MA 列
is_bull = last_row['MA20'] > last_row['MA60'] if 'MA20' in last_row else df_idx['close'].rolling(20).mean().iloc[-1] > df_idx['close'].rolling(60).mean().iloc[-1]

m1.metric("市场模式", "📈 多头 (Bull)" if is_bull else "📉 空头 (Bear)")
m2.metric("资金热度 (Z)", f"{last_row['Heat_Z']:.2f}")
m3.metric("市场宽度 (MA20%)", f"{curr_ma20:.1f}%")

st.write("🔥 **全市场量能共振监测 (实时换手率)**")
t1, t2, t3, t4 = st.columns(4)
t1.metric("上证50", f"{all_turnovers['SSE50']:.2f}%")
t2.metric("沪深300", f"{all_turnovers['CSI300']:.2f}%")
t3.metric("中证500", f"{all_turnovers['CSI500']:.2f}%")
t4.metric("中证1000", f"{all_turnovers['CSI1000']:.2f}%")

st.info(f"**模式分析**：{'📈 当前为：多头趋势环境' if is_bull else '📉 当前为：空头趋势环境'}")

# ==========================================
# 5. 最终结论与走势图 (新逻辑集成)
# ==========================================
st.divider()
st.subheader("💡 最终操作建议 (旗舰进化版)")

if last_row['signal'] == 1:
    if last_row['logic_type'] == "Strategic":
        st.warning("🚀 **综合结论：战略买入触发！** 全市场广度进入冰点区（<16%），宏观盈亏比极高，建议建立中长线底仓。")
    else:
        st.success("🔥 **综合结论：战术加仓触发！** MA30多头环境下完成首阴回踩，且放量共振，短期爆发力强。")
elif last_row['signal'] == -1:
    st.error("🚨 **综合结论：防御减仓！** 触发复合止损逻辑（趋势破位或时间失效），建议收缩头寸，保护利润。")
else:
    if last_row['ma20_ratio'] > 75:
        st.warning("⌛ **综合结论：持股待涨。** 广度进入高位过热边缘，不宜追高，关注信号。")
    else:
        st.info("✅ **综合结论：目前处于平稳期。** 逻辑未变，建议按原有比例持仓，耐心等待。")

# --- 新增：中证500 日 K 线标注图 ---
st.markdown("#### 📅 中证500 (sh000905) 走势与信号标注 (2024至今)")
df_plot = df_final.loc["2024-01-01":]
fig3, ax3 = plt.subplots(figsize=(16, 8))
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.5, label='收盘价')
ax3.plot(df_plot.index, df_plot['MA30'], color='blue', linestyle='--', alpha=0.4, label='MA30趋势过滤')

# 标注买点
buys = df_plot[df_plot['signal'] == 1]
ax3.scatter(buys.index, buys['close'], color='red', marker='^', s=120, zorder=5, label='买入点 (战略/战术)')
# 标注卖点
sells = df_plot[df_plot['signal'] == -1]
ax3.scatter(sells.index, sells['close'], color='green', marker='v', s=120, zorder=5, label='卖出点 (复合止损)')

ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.2)
st.pyplot(fig3)

# --- 决策逻辑判定详情 ---
with st.expander("查看【MA30过滤版 旗舰进化】决策逻辑判定详情"):
    st.write(f"""
    - **战略买入**：市场广度 < 16% (当前: {curr_ma20:.1f}%)
    - **战术买入**：MA30线上 + 10日线上 + 连阳后首阴 + 换手>1.0% + 5日线不破
    - **复合止损**：
        1. 宏观过热 (广度 > 79% 且 资金热度衰减)
        2. 战术破位 (价格 < MA30 且 (今日收阴 或 5日不创新高))
    """)
