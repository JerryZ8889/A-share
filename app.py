import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 环境配置
# ==========================================
st.set_page_config(page_title="量化大师-100%信号还原版", layout="wide")
st.title("🛡️ 量化大师：信号 1:1 还原版 (解决冷启动问题)")

def set_chinese_font():
    font_list = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif']
    plt.rcParams['font.sans-serif'] = font_list + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
set_chinese_font()

# ==========================================
# 1. 核心数据加载 (增加预热期)
# ==========================================
@st.cache_data(ttl=60)
def load_all_data():
    # 注意：生产环境 CSV 建议包含至少 2023 年末的数据以供预热
    df_scan = pd.read_csv("scan_results.csv", index_col='date', parse_dates=True).sort_index()
    df_main = pd.read_csv("CSI500_Master_Strategy.csv", index_col='date', parse_dates=True).sort_index()
    df_summary = pd.read_csv("master_summary.csv") if os.path.exists("master_summary.csv") else pd.DataFrame()
    
    # 强制将两个主表的日期索引对齐，避免 join 产生 NaN
    combined = df_main.join(df_scan[['ma20_ratio']], how='inner')
    combined.rename(columns={'ma20_ratio': 'breadth'}, inplace=True)
    
    return combined, df_summary

combined_df, df_summary = load_all_data()

# ==========================================
# 2. 仿真引擎 (强化逻辑鲁棒性)
# ==========================================
def backtest_engine_final(df):
    temp = df.copy()
    
    # --- 指标计算 (必须在切片前计算，确保预热) ---
    temp['MA_Filter'] = temp['close'].rolling(30).mean()
    temp['MA_Support'] = temp['close'].rolling(5).mean()
    temp['MA_Trend'] = temp['close'].rolling(10).mean()
    temp['MA60'] = temp['close'].rolling(60).mean()
    
    # 连阳逻辑
    temp['Is_Up'] = (temp['close'] > temp['close'].shift(1)).astype(int)
    temp['Streak'] = temp['Is_Up'].groupby((temp['Is_Up'] != temp['Is_Up'].shift()).cumsum()).cumcount() + 1
    temp['Consec_Gains'] = np.where(temp['Is_Up'] == 1, temp['Streak'], 0)
    
    # 热度与换手
    target_col = 'amount' if 'amount' in temp.columns else 'volume'
    temp['Heat_Z'] = (temp[target_col] - temp[target_col].rolling(20).mean()) / temp[target_col].rolling(20).std()
    temp['Turnover_Pct'] = np.where(temp['ETF_Turnover'] > 1, temp['ETF_Turnover'], temp['ETF_Turnover'] * 100)

    # --- 信号循环 ---
    temp['pos'] = 0; temp['signal'] = 0; temp['logic_type'] = ""; temp['marker'] = ""
    in_pos = False; logic_state = ""; entry_idx = 0; entry_high = 0

    # 预判定向量
    cond_comp_b = (temp['breadth'] < 16)
    cond_comp_s = (temp['breadth'] > 79) & (temp['Heat_Z'] < 1.5)
    # 这里的 shift(1) 是导致早期信号丢失的关键，必须确保 i=1 时能取到 i=0 的值
    cond_fn_b_base = (temp['close'] > temp['MA_Trend']) & \
                     (temp['Consec_Gains'].shift(1) >= 3) & \
                     (temp['close'] < temp['close'].shift(1)) & \
                     (temp['Turnover_Pct'] > 1.0) & \
                     (temp['close'] > temp['MA_Support'])

    for i in range(len(temp)):
        if i < 30: continue # 略过预热期，但不切断数据
        
        curr_c = temp['close'].iloc[i]; prev_c = temp['close'].iloc[i-1]
        curr_ma30 = temp['MA_Filter'].iloc[i]
        
        if in_pos:
            if logic_state == "FirstNeg" and cond_comp_b.iloc[i]:
                logic_state = "Composite"; temp.iloc[i, temp.columns.get_loc('marker')] = "升级"

            exit_flag = False
            is_1d = curr_c < prev_c
            is_5d = (i - entry_idx >= 5) and not (temp['close'].iloc[entry_idx:i+1] > entry_high).any()
            is_below_ma = curr_c < curr_ma30

            if logic_state == "Composite":
                if cond_comp_s.iloc[i]: exit_flag = True
            else: 
                if cond_comp_s.iloc[i]: exit_flag = True
                elif is_below_ma and (is_1d or is_5d): exit_flag = True
            
            if exit_flag:
                temp.iloc[i, temp.columns.get_loc('signal')] = -1
                in_pos = False; logic_state = ""
            else:
                temp.iloc[i, temp.columns.get_loc('pos')] = 1
        
        else:
            triggered = False
            if cond_comp_b.iloc[i]:
                logic_state = "Composite"; temp.iloc[i, temp.columns.get_loc('logic_type')] = "Strategic"
                triggered = True
            elif cond_fn_b_base.iloc[i] and (curr_c > curr_ma30):
                logic_state = "FirstNeg"; temp.iloc[i, temp.columns.get_loc('logic_type')] = "Tactical"
                triggered = True
            
            if triggered:
                temp.iloc[i, temp.columns.get_loc('signal')] = 1
                temp.iloc[i, temp.columns.get_loc('pos')] = 1
                in_pos = True; entry_idx = i; entry_high = temp['high'].iloc[i]

    return temp

df_final = backtest_engine_final(combined_df)
# 只展示 2024 年以后的图表，但计算是从最早开始的
df_plot = df_final.loc["2024-01-01":]
last_row = df_plot.iloc[-1]

# ==========================================
# 3. 布局渲染 (原有指标部分)
# ==========================================
m1, m2, m3 = st.columns(3)
m1.metric("市场模式", "📈 多头" if last_row['close'] > last_row['MA60'] else "📉 空头")
m2.metric("资金热度 (Z)", f"{last_row['Heat_Z']:.2f}")
m3.metric("市场宽度", f"{last_row['breadth']:.1f}%")

st.divider()

# ==========================================
# 4. 关键：K线标注 (确保显示所有信号点)
# ==========================================
st.subheader("💡 信号 1:1 还原分布图")
fig3, ax3 = plt.subplots(figsize=(16, 8))
ax3.plot(df_plot.index, df_plot['close'], color='silver', alpha=0.6, label='Price')
ax3.plot(df_plot.index, df_plot['MA_Filter'], color='blue', linestyle='--', alpha=0.8, label='MA30 Filter')

# 标注买点
buys = df_plot[df_plot['signal'] == 1]
ax3.scatter(buys.index, buys['close'], color='red', marker='^', s=150, zorder=10, label='Buy')
# 标注卖点
sells = df_plot[df_plot['signal'] == -1]
ax3.scatter(sells.index, sells['close'], color='green', marker='v', s=150, zorder=10, label='Sell')
# 标注升级点
upgrades = df_plot[df_plot['marker'] == "升级"]
ax3.scatter(upgrades.index, upgrades['close'], color='orange', marker='o', s=100, alpha=0.7, label='Upgrade')

ax3.legend(loc='upper left'); ax3.grid(True, alpha=0.1)
st.pyplot(fig3)

# ==========================================
# 5. 广度遮罩图 (确认 pos 连续性)
# ==========================================
st.markdown("#### 🌊 策略持仓区间分布 (同步校验)")
fig4, ax4 = plt.subplots(figsize=(16, 3))
ax4.plot(df_plot.index, df_plot['breadth'], color='orange', alpha=0.8)
ax4.fill_between(df_plot.index, 0, 100, where=(df_plot['pos']==1), color='blue', alpha=0.15)
ax4.axhline(y=16, color='red', linestyle=':', alpha=0.5)
ax4.axhline(y=79, color='green', linestyle=':', alpha=0.5)
ax4.set_ylim(0, 100); ax4.grid(True, alpha=0.1)
st.pyplot(fig4)
