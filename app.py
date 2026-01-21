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
st.title("🛡️ 量化大师：MA30过滤旗舰进化版 (回测逻辑同步)")

def set_chinese_font():
    font_list = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif']
    plt.rcParams['font.sans-serif'] = font_list + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# ==========================================
# 1. 核心数据加载 (确保包含 amount)
# ==========================================
@st.cache_data(ttl=60)
def load_all_data():
    # 1. 加载指数 (用于看板热度显示)
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    
    # 2. 加载 CSV 文件
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
# 2. 仿真引擎：完全同步回测逻辑 (backtest_engine 移植)
# ==========================================
def calculate_synchronized_signals(df_p, df_b):
    temp = df_p.copy()
    # --- 1. 特征计算 (对齐回测口径) ---
    temp['MA_Filter'] = temp['close'].rolling(30).mean()
    temp['MA_Support'] = temp['close'].rolling(5).mean()
    temp['MA_Trend'] = temp['close'].rolling(10).mean()
    temp['MA60'] = temp['close'].rolling(60).mean() # 用于多头模式判断
    
    # 关键：Heat_Z 使用 amount 且 周期为 20
    if 'amount' in temp.columns:
        amt = temp['amount']
    else:
        # 如果 CSV 里没有 amount，回退到 volume 但保持 20 周期
        amt = temp['volume']
    temp['Heat_Z'] = (amt - amt.rolling(20).mean()) / amt.rolling(20).std()
    
    # 合并广度
    temp = temp.join(df_b[['ma20_ratio']], how='left').ffill()
    
    # 连阳逻辑
    temp['Is_Up'] = (temp['close'] > temp['close'].shift(1)).astype(int)
    temp['Streak'] = temp['Is_Up'].groupby((temp['Is_Up'] != temp['Is_Up'].shift()).cumsum()).cumcount() + 1
    temp['Consec_Gains'] = np.where(temp['Is_Up'] == 1, temp['Streak'], 0)
    
    # 换手率归一化
    temp['Turnover_Pct'] = np.where(temp['ETF_Turnover'] > 1, temp['ETF_Turnover'], temp['ETF_Turnover'] * 100)

    # --- 2. 信号仿真 (逐日循环) ---
    temp['signal'] = 0; temp['logic_type'] = ""; temp['marker'] = ""
    in_pos = False; logic_state = ""; entry_idx = 0; entry_high = 0

    for i in range(len(temp)):
        if i == 0: continue
        
        # 判定条件
        cond_comp_b = temp['ma20_ratio'].iloc[i] < 16
        cond_comp_s = (temp['ma20_ratio'].iloc[i] > 79) and (temp['Heat_Z'].iloc[i] < 1.5)
        
        # 战术买入基准条件
        cond_fn_b_base = (temp['close'].iloc[i] > temp['MA_Trend'].iloc[i]) and \
                         (temp['Consec_Gains'].iloc[i-1] >= 3) and \
                         (temp['close'].iloc[i] < temp['close'].iloc[i-1]) and \
                         (temp['Turnover_Pct'].iloc[i] > 1.0) and \
                         (temp['close'].iloc[i] > temp['MA_Support'].iloc[i])

        if in_pos:
            # 身份升级逻辑：战术单 -> 战略单 (Fusion)
            if logic_state == "FirstNeg" and cond_comp_b:
                logic_state = "Composite"
                temp.iloc[i, temp.columns.get_loc('marker')] = "升级"

            # 卖出判定逻辑
            exit_flag = False
            is_1d = temp['close'].iloc[i] < temp['close'].iloc[i-1]
            # 5天不创新高判定 (对齐回测 [entry_idx:i+1])
            is_5d = (i - entry_idx >= 5) and not (temp['close'].iloc[entry_idx:i+1] > entry_high).any()
            is_below_ma = temp['close'].iloc[i] < temp['MA_Filter'].iloc[i]

            if logic_state == "Composite":
                if cond_comp_s: exit_flag = True
            else: # FirstNeg
                if cond_comp_s: exit_flag = True
                elif is_below_ma and (is_1d or is_5d): exit_flag = True
            
            if exit_flag:
                temp.iloc[i, temp.columns.get_loc('signal')] = -1
                in_pos = False; logic_state = ""
        
        else: # 未持仓，判定买入
            if cond_comp_b:
                temp.iloc[i, temp.columns.get_loc('signal')] = 1
                temp.iloc[i, temp.columns.get_loc('logic_type')] = "Strategic"
                in_pos = True; logic_state = "Composite"; entry_idx = i; entry_high = temp['high'].iloc[i]
            else:
                # 战术买入增加 MA30 过滤
                if cond_fn_b_base and (temp['close'].iloc[i] > temp['MA_Filter'].iloc[i]):
                    temp.iloc[i, temp.columns.get_loc('signal')] = 1
                    temp.iloc[i, temp.columns.get_loc('logic_type')] = "Tactical"
                    in_pos = True; logic_state = "FirstNeg"; entry_idx = i; entry_high = temp['high'].iloc[i]

    return temp

df_final = calculate_synchronized_signals(df_main, df_scan)
last_row = df_final.iloc[-1]

# ==========================================
# 3. 布局渲染 (保持原面板不改动)
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
m3.metric("市场宽度", f"{last_row['ma20_ratio']:.1f}%")

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

# 标注所有买入点
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
# 5. 决策逻辑详情
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
