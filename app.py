import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# 0. 页面配置与字体修复
# ==========================================
st.set_page_config(page_title="量化大师-生产回测对齐版", layout="wide")
st.title("🛡️ 量化大师：MA30过滤旗舰进化版 (逻辑完全同步)")

def set_chinese_font():
    font_list = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif']
    plt.rcParams['font.sans-serif'] = font_list + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# ==========================================
# 1. 核心数据加载
# ==========================================
@st.cache_data(ttl=0)
def load_all_data():
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    
    df_scan = pd.read_csv("scan_results.csv", index_col='date', parse_dates=True).sort_index()
    df_main = pd.read_csv("CSI500_Master_Strategy.csv", index_col='date', parse_dates=True).sort_index()
    df_summary = pd.read_csv("master_summary.csv") if os.path.exists("master_summary.csv") else pd.DataFrame()
    
    return df_idx, df_scan, df_main, df_summary

try:
    df_idx, df_scan, df_main, df_summary = load_all_data()
except Exception as e:
    st.error(f"⚠️ 数据加载失败: {e}")
    st.stop()

# ==========================================
# 2. 仿真引擎：完全复用代码2的回测逻辑
# ==========================================
def calculate_synchronized_signals(df_p, df_b):
    temp = df_p.copy()
    # 对齐列名：生产环境的 ma20_ratio 即回测的 breadth
    temp = temp.join(df_b[['ma20_ratio']], how='left').ffill()
    temp.rename(columns={'ma20_ratio': 'breadth'}, inplace=True)

    # --- 1. 特征计算 (严格对齐代码2) ---
    temp['MA_Filter'] = temp['close'].rolling(30).mean()
    temp['MA_Support'] = temp['close'].rolling(5).mean()
    temp['MA_Trend'] = temp['close'].rolling(10).mean()
    temp['MA60'] = temp['close'].rolling(60).mean() # UI展示用
    
    # Heat_Z 计算
    amt_col = 'amount' if 'amount' in temp.columns else 'volume'
    temp['Heat_Z'] = (temp[amt_col] - temp[amt_col].rolling(20).mean()) / temp[amt_col].rolling(20).std()
    
    # 连阳逻辑
    temp['Is_Up'] = (temp['close'] > temp['close'].shift(1)).astype(int)
    temp['Streak'] = temp['Is_Up'].groupby((temp['Is_Up'] != temp['Is_Up'].shift()).cumsum()).cumcount() + 1
    temp['Consec_Gains'] = np.where(temp['Is_Up'] == 1, temp['Streak'], 0)
    
    # 换手率归一化
    temp['Turnover_Pct'] = np.where(temp['ETF_Turnover'] > 1, temp['ETF_Turnover'], temp['ETF_Turnover'] * 100)

    # --- 2. 信号预判定 (严格对齐代码2) ---
    cond_comp_b = (temp['breadth'] < 16)
    cond_comp_s = (temp['breadth'] > 79) & (temp['Heat_Z'] < 1.5)
    
    # 战术买入基准条件 (注意 Consec_Gains.shift(1))
    cond_fn_b_base = (temp['close'] > temp['MA_Trend']) & \
                     (temp['Consec_Gains'].shift(1) >= 3) & \
                     (temp['close'] < temp['close'].shift(1)) & \
                     (temp['Turnover_Pct'] > 1.0) & \
                     (temp['close'] > temp['MA_Support'])

    # --- 3. 仿真循环 (状态机对齐代码2) ---
    temp['pos'] = 0
    temp['signal'] = 0
    temp['logic_type'] = ""
    temp['marker'] = ""
    
    in_pos = False
    logic_state = "" 
    entry_idx, entry_high = 0, 0

    for i in range(len(temp)):
        if i == 0: continue
        
        current_close = temp['close'].iloc[i]
        prev_close = temp['close'].iloc[i-1]
        current_ma30 = temp['MA_Filter'].iloc[i]
        
        if in_pos:
            # 身份升级
            if logic_state == "FirstNeg" and cond_comp_b.iloc[i]:
                logic_state = "Composite"
                temp.iloc[i, temp.columns.get_loc('marker')] = "升级"

            exit_flag = False
            is_1d = current_close < prev_close
            is_5d = (i - entry_idx >= 5) and not (temp['close'].iloc[entry_idx:i+1] > entry_high).any()
            is_below_ma = current_close < current_ma30

            if logic_state == "Composite":
                if cond_comp_s.iloc[i]: exit_flag = True
            else: 
                if cond_comp_s.iloc[i]: exit_flag = True
                elif is_below_ma and (is_1d or is_5d): exit_flag = True
            
            if exit_flag:
                temp.iloc[i, temp.columns.get_loc('signal')] = -1
                temp.iloc[i, temp.columns.get_loc('pos')] = 0
                in_pos, logic_state = False, ""
            else:
                temp.iloc[i, temp.columns.get_loc('pos')] = 1
        
        else: # 未持仓
            buy_triggered = False
            if cond_comp_b.iloc[i]: 
                temp.iloc[i, temp.columns.get_loc('logic_type')] = "Strategic"
                logic_state = "Composite"
                buy_triggered = True
            elif cond_fn_b_base.iloc[i] and (current_close > current_ma30):
                temp.iloc[i, temp.columns.get_loc('logic_type')] = "Tactical"
                logic_state = "FirstNeg"
                buy_triggered = True
            
            if buy_triggered:
                temp.iloc[i, temp.columns.get_loc('signal')] = 1
                temp.iloc[i, temp.columns.get_loc('pos')] = 1
                in_pos = True
                entry_idx, entry_high = i, temp['high'].iloc[i]

    return temp

df_final = calculate_synchronized_signals(df_main, df_scan)
last_row = df_final.iloc[-1]

# ==========================================
# 3. 布局渲染 (保持原有 UI 面板)
# ==========================================
c1, c2 = st.columns(2)
with c1:
    st.subheader("🔥 资金热度 (Z-Score)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_z = df_final['Heat_Z'].tail(100)
    ax1.fill_between(p_z.index, p_z, 0, where=(p_z>=0), color='red', alpha=0.3)
    ax1.fill_between(p_z.index, p_z, 0, where=(p_z<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    st.pyplot(fig1)

with c2:
    st.subheader("📊 市场广度趋势")
    fig2, axl = plt.subplots(figsize=(10, 5))
    axl.plot(df_scan.index, df_scan['ma20_ratio'], color='tab:blue', label='MA20%')
    axr = axl.twinx()
    axr.bar(df_scan.index, df_scan['new_high_ratio'], color='tab:orange', alpha=0.3)
    st.pyplot(fig2)

st.divider()
st.subheader("🛡️ 动态逻辑诊断报告")
m1, m2, m3 = st.columns(3)
is_bull = last_row['close'] > last_row['MA60']
m1.metric("市场模式", "📈 多头" if is_bull else "📉 空头")
m2.metric("资金热度 (Z)", f"{last_row['Heat_Z']:.2f}")
m3.metric("市场宽度", f"{last_row['breadth']:.1f}%")

st.write("🔥 **全市场量能共振监测**")
def get_t(lbl):
    if not df_summary.empty:
        v = df_summary[df_summary['Index_Label'] == lbl]['ETF_Turnover'].values
        if len(v)>0: return v[0] if v[0]>1 else v[0]*100
    return 0.0
t1, t2, t3, t4 = st.columns(4)
t1.metric("上证50", f"{get_t('SSE50'):.2f}%")
t2.metric("沪深300", f"{get_t('CSI300'):.2f}%")
t3.metric("中证500", f"{last_row['Turnover_Pct']:.2f}%")
t4.metric("中证1000", f"{get_t('CSI1000'):.2f}%")

# ==========================================
# 4. K线标注与结论 (同步回测显示所有点)
# ==========================================
st.divider()
st.subheader("💡 最终操作建议与走势标注")

if last_row['signal'] == 1:
    st.success(f"🚀 **操作建议：买入 ({last_row['logic_type']})**")
elif last_row['signal'] == -1:
    st.error("🚨 **操作建议：清仓/减仓**")
else:
    st.info("✅ **操作建议：持股/观望**")

st.markdown("#### 📅 中证500 (sh000905) 走势与信号标注 (2024至今)")
df_plot = df_final.loc["2024-01-01":]
fig3, ax3 = plt.subplots(figsize=(16, 8))
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.5, label='Close Price')
ax3.plot(df_plot.index, df_plot['MA_Filter'], color='blue', linestyle='--', label='MA30 Filter')

# 标注所有买入点 (对齐代码2的 scatter 逻辑)
buys = df_plot[df_plot['signal'] == 1]
ax3.scatter(buys.index, buys['close'], color='red', marker='^', s=120, zorder=5, label='Buy Signal')
# 标注所有卖出点
sells = df_plot[df_plot['signal'] == -1]
ax3.scatter(sells.index, sells['close'], color='green', marker='v', s=120, zorder=5, label='Sell Signal')
# 标注升级点
upgrades = df_plot[df_plot['marker'] == "升级"]
ax3.scatter(upgrades.index, upgrades['close'], color='orange', marker='o', s=80, alpha=0.6, label='Identity Upgrade')

ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.2)
st.pyplot(fig3)

# ==========================================
# 5. 新增：市场广度波动环境图 (代码2核心图表)
# ==========================================
st.markdown("#### 🌊 市场广度波动环境 (持仓区间同步)")
fig4, ax4 = plt.subplots(figsize=(16, 4))
ax4.plot(df_plot.index, df_plot['breadth'], color='orange', label='市场广度 (breadth)', alpha=0.8)
ax4.axhline(y=16, color='red', linestyle='--', alpha=0.6, label='战略抄底区 (16%)')
ax4.axhline(y=79, color='green', linestyle='--', alpha=0.6, label='宏观风险区 (79%)')

# 用淡蓝色背景显示持仓区间 (核心对齐代码2)
ax4.fill_between(df_plot.index, 0, 100, where=(df_plot['pos']==1), color='blue', alpha=0.1, label='策略持仓中')

ax4.set_ylim(0, 100)
ax4.legend(loc='upper left', ncol=4)
ax4.grid(True, alpha=0.2)
st.pyplot(fig4)

# ==========================================
# 6. 决策逻辑详情
# ==========================================
with st.expander("查看【回测同步版】决策逻辑判定详情", expanded=True):
    st.markdown("""
    ### ⚔️ 核心策略逻辑 (已与 Backtest 脚本完全对齐)
    
    1. **战略买入 (Composite)**：广度 < 16%。此单为战略底仓，止损极度宽松。
    2. **战术买入 (FirstNeg)**：
        - 必须处于 **MA30** 趋势线上方。
        - 满足 10日线上 + 5日线上 + 3连阳后首阴 + 换手率 > 1%。
    3. **身份升级**：若持有战术单期间，市场广度跌破 16%，该单自动“升级”为战略单，不再执行战术止损。
    4. **复合止损 (仅针对战术单)**：
        - 宏观过热 (广度 > 79% 且 资金热度 Z < 1.5)。
        - **或者** 价格跌破 MA30 且 (今日收阴线 或 5日不创新高)。
    """)
