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
st.set_page_config(page_title="量化大师-逻辑同步版", layout="wide")
st.title("🛡️ 量化大师：MA30过滤旗舰进化版 (生产/回测完全同步)")

def set_chinese_font():
    # 尝试设置中文字体，兼容多系统
    font_list = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif']
    plt.rcParams['font.sans-serif'] = font_list + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# ==========================================
# 1. 核心数据加载
# ==========================================
@st.cache_data(ttl=60)
def load_all_data():
    # 1. 加载指数 (用于显示)
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    
    # 2. 加载 CSV 文件 (生产环境数据源)
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
# 2. 仿真引擎：完全同步回测代码逻辑 (Core)
# ==========================================
def calculate_synchronized_signals(df_p, df_b):
    """
    此函数逻辑已完全对齐回测脚本。
    df_p: 包含价格、成交量/额、换手率的主数据
    df_b: 包含市场广度的数据
    """
    temp = df_p.copy()
    
    # --- 1. 特征计算 (对齐回测口径) ---
    # 映射广度列名
    temp = temp.join(df_b[['ma20_ratio']], how='left').ffill()
    temp.rename(columns={'ma20_ratio': 'breadth'}, inplace=True) 

    # 移动平均线
    temp['MA_Filter'] = temp['close'].rolling(30).mean()   # MA30 趋势过滤
    temp['MA_Support'] = temp['close'].rolling(5).mean()
    temp['MA_Trend'] = temp['close'].rolling(10).mean()
    temp['MA60'] = temp['close'].rolling(60).mean()       # 用于多空模式判断
    
    # 资金热度 Z-Score (使用 amount 且周期为 20)
    target_col = 'amount' if 'amount' in temp.columns else 'volume'
    temp['Heat_Z'] = (temp[target_col] - temp[target_col].rolling(20).mean()) / temp[target_col].rolling(20).std()
    
    # 连阳逻辑
    temp['Is_Up'] = (temp['close'] > temp['close'].shift(1)).astype(int)
    temp['Streak'] = temp['Is_Up'].groupby((temp['Is_Up'] != temp['Is_Up'].shift()).cumsum()).cumcount() + 1
    temp['Consec_Gains'] = np.where(temp['Is_Up'] == 1, temp['Streak'], 0)
    
    # 换手率标准化
    temp['Turnover_Pct'] = np.where(temp['ETF_Turnover'] > 1, temp['ETF_Turnover'], temp['ETF_Turnover'] * 100)

    # --- 2. 预计算买入条件 (向量化计算提高效率) ---
    cond_comp_b = (temp['breadth'] < 16)
    cond_comp_s = (temp['breadth'] > 79) & (temp['Heat_Z'] < 1.5)
    
    # 战术买入基准条件：10日线上 + 3连阳后首阴 + 换手率 > 1% + 5日线上
    cond_fn_b_base = (temp['close'] > temp['MA_Trend']) & \
                     (temp['Consec_Gains'].shift(1) >= 3) & \
                     (temp['close'] < temp['close'].shift(1)) & \
                     (temp['Turnover_Pct'] > 1.0) & \
                     (temp['close'] > temp['MA_Support'])

    # --- 3. 状态机循环 (逻辑完全同步 backtest_engine) ---
    temp['pos'] = 0
    temp['signal'] = 0
    temp['logic_type'] = ""
    temp['marker'] = ""
    
    in_pos = False
    logic_state = "" # "Composite" (战略) 或 "FirstNeg" (战术)
    entry_idx, entry_high = 0, 0

    for i in range(len(temp)):
        if i == 0: continue
        
        current_close = temp['close'].iloc[i]
        prev_close = temp['close'].iloc[i-1]
        current_ma30 = temp['MA_Filter'].iloc[i]
        
        if in_pos:
            # 身份升级逻辑：持仓过程中如果触及战略抄底区，自动升级
            if logic_state == "FirstNeg" and cond_comp_b.iloc[i]:
                logic_state = "Composite"
                temp.iloc[i, temp.columns.get_loc('marker')] = "升级"

            # 卖出判定逻辑
            exit_flag = False
            is_1d = current_close < prev_close
            is_5d = (i - entry_idx >= 5) and not (temp['close'].iloc[entry_idx:i+1] > entry_high).any()
            is_below_ma = current_close < current_ma30

            if logic_state == "Composite":
                if cond_comp_s.iloc[i]: exit_flag = True
            else: # FirstNeg (战术止损)
                if cond_comp_s.iloc[i]: exit_flag = True
                elif is_below_ma and (is_1d or is_5d): exit_flag = True
            
            if exit_flag:
                temp.iloc[i, temp.columns.get_loc('signal')] = -1
                temp.iloc[i, temp.columns.get_loc('pos')] = 0
                in_pos, logic_state = False, ""
            else:
                temp.iloc[i, temp.columns.get_loc('pos')] = 1
        
        else: # 未持仓，判定买入
            buy_triggered = False
            if cond_comp_b.iloc[i]: # 优先触发战略买入
                temp.iloc[i, temp.columns.get_loc('logic_type')] = "Strategic"
                logic_state = "Composite"
                buy_triggered = True
            elif cond_fn_b_base.iloc[i] and (current_close > current_ma30): # 战术买入需加 MA30 过滤
                temp.iloc[i, temp.columns.get_loc('logic_type')] = "Tactical"
                logic_state = "FirstNeg"
                buy_triggered = True
            
            if buy_triggered:
                temp.iloc[i, temp.columns.get_loc('signal')] = 1
                temp.iloc[i, temp.columns.get_loc('pos')] = 1
                in_pos = True
                entry_idx = i
                entry_high = temp['high'].iloc[i]

    return temp

df_final = calculate_synchronized_signals(df_main, df_scan)
last_row = df_final.iloc[-1]

# ==========================================
# 3. 布局渲染：仪表盘
# ==========================================
c1, c2 = st.columns(2)
with c1:
    st.subheader("🔥 资金热度 (Z-Score)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_z = df_final['Heat_Z'].tail(100)
    ax1.fill_between(p_z.index, p_z, 0, where=(p_z>=0), color='red', alpha=0.3)
    ax1.fill_between(p_z.index, p_z, 0, where=(p_z<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    ax1.set_title("近期资金共振强度 (20日窗口)")
    st.pyplot(fig1)

with c2:
    st.subheader("📊 市场广度趋势")
    fig2, axl = plt.subplots(figsize=(10, 5))
    axl.plot(df_scan.index, df_scan['ma20_ratio'], color='tab:blue', label='MA20% (广度)')
    axr = axl.twinx()
    axr.bar(df_scan.index, df_scan['new_high_ratio'], color='tab:orange', alpha=0.3, label='新高比例')
    axl.set_title("广度指标与新高共振")
    st.pyplot(fig2)

st.divider()
st.subheader("🛡️ 动态逻辑诊断报告")
m1, m2, m3 = st.columns(3)
is_bull = last_row['close'] > last_row['MA60']
m1.metric("市场模式", "📈 多头" if is_bull else "📉 空头")
m2.metric("资金热度 (Z)", f"{last_row['Heat_Z']:.2f}")
m3.metric("市场宽度 (Breadth)", f"{last_row['breadth']:.1f}%")

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
# 4. K线标注与走势
# ==========================================
st.divider()
st.subheader("💡 信号走势标注 (2024至今)")

if last_row['signal'] == 1:
    st.success(f"🚀 **操作建议：买入 ({last_row['logic_type']})**")
elif last_row['signal'] == -1:
    st.error("🚨 **操作建议：清仓/减仓**")
else:
    if last_row['pos'] == 1:
        st.info("✅ **操作建议：继续持仓**")
    else:
        st.info("✅ **操作建议：空仓观望**")

df_plot = df_final.loc["2024-01-01":]
fig3, ax3 = plt.subplots(figsize=(16, 7))
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.4, label='CSI500 Close')
ax3.plot(df_plot.index, df_plot['MA_Filter'], color='blue', linestyle='--', alpha=0.6, label='MA30趋势过滤线')

# 标注买卖点
buys = df_plot[df_plot['signal'] == 1]
ax3.scatter(buys.index, buys['close'], color='red', marker='^', s=150, zorder=5, label='买入信号')
sells = df_plot[df_plot['signal'] == -1]
ax3.scatter(sells.index, sells['close'], color='green', marker='v', s=150, zorder=5, label='卖出信号')
upgrades = df_plot[df_plot['marker'] == "升级"]
ax3.scatter(upgrades.index, upgrades['close'], color='orange', marker='o', s=100, alpha=0.8, label='身份升级点')

ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.1)
st.pyplot(fig3)

# ==========================================
# 5. 新增：市场广度波动环境图 (完全对齐回测)
# ==========================================
st.subheader("🌊 市场广度波动环境 (持仓区间监测)")
fig4, ax4 = plt.subplots(figsize=(16, 4))
ax4.plot(df_plot.index, df_plot['breadth'], color='orange', label='市场广度 (breadth)', alpha=0.8)
ax4.axhline(y=16, color='red', linestyle='--', alpha=0.6, label='战略抄底区 (16%)')
ax4.axhline(y=79, color='green', linestyle='--', alpha=0.6, label='宏观风险区 (79%)')

# 用淡蓝色背景显示持仓区间 (pos == 1)
ax4.fill_between(df_plot.index, 0, 100, where=(df_plot['pos']==1), color='blue', alpha=0.1, label='策略持仓中')

ax4.set_ylim(0, 100)
ax4.set_ylabel("广度百分比 (%)")
ax4.legend(loc='upper left', ncol=4)
ax4.grid(True, alpha=0.2, axis='y')
st.pyplot(fig4)

# ==========================================
# 6. 决策逻辑说明
# ==========================================
with st.expander("查看核心策略决策逻辑 (已与 Backtest 同步)", expanded=False):
    st.markdown("""
    ### ⚔️ 核心逻辑细节
    1. **战略买入**：市场广度跌破 **16%**。视为市场进入“绝望区”，战略性建仓。
    2. **战术买入**：
        - 价格必须在 **MA30** 过滤线上方（确保不接坠落的飞刀）。
        - 满足“连阳后首阴”：过去3天上涨，今日收跌。
        - 配合量能：换手率 > 1%。
    3. **动态升级**：若以战术买入，持仓期内广度跌破 16%，该头寸自动转为“战略持有”，过滤掉战术级别的止损动作。
    4. **复合卖出**：
        - **过热止盈**：广度 > 79% 且资金热度出现衰减（Z < 1.5）。
        - **破位止损**：价格跌破 MA30，且满足（今日收阴 或 5日不创新高）。
    """)
