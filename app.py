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
st.set_page_config(page_title="量化大师-旗舰进化版", layout="wide")
st.title("🛡️ 量化大师：MA30过滤旗舰进化版综合看板")

# --- ⚡ 字体兼容性修复 ---
def set_matplotlib_font():
    # 尝试多种常用中文字体
    fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False # 修复负号显示
set_matplotlib_font()

# ==========================================
# 1. 核心数据加载 (保持原结构)
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
# 2. 旗舰进化版计算引擎 (修复 nan 问题)
# ==========================================
def calculate_flagship_signals(df_price, df_breadth):
    df = df_price.copy()
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean() # 补上矩阵需要的列
    df['MA30'] = df['close'].rolling(30).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 广度合并
    df = df.join(df_breadth[['ma20_ratio', 'new_high_ratio']], how='left').ffill()
    
    # --- ⚡ 修复 Z-Score nan 问题 ---
    vol = df['volume']
    # 确保窗口内有值，并向前填充
    df['Heat_Z'] = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    df['Heat_Z'] = df['Heat_Z'].ffill().fillna(0) # 填充最后的空值
    
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Consec_Gains'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Consec_Gains'], 0)
    
    df['Turnover_Pct'] = np.where(df['ETF_Turnover'] > 1, df['ETF_Turnover'], df['ETF_Turnover'] * 100)
    
    # 信号循环
    df['signal'] = 0
    df['logic_type'] = ""
    in_pos, logic_state, entry_high, hold_days = False, "", 0, 0

    for i in range(1, len(df)):
        curr, prev = df.iloc[i], df.iloc[i-1]
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
                  prev['Consec_Gains'] >= 3 and curr['close'] < prev['close'] and curr['Turnover_Pct'] > 1.0):
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Tactical"
                in_pos, logic_state, hold_days, entry_high = True, "Tactical", 0, curr['high']
    return df

df_final = calculate_flagship_signals(df_main, df_scan)
last_row = df_final.iloc[-1]

# ==========================================
# 3. 页面布局与看板 (保持原有逻辑)
# ==========================================
# [此处代码与之前相同：col1/col2、诊断报告、换手率监测矩阵等]

# ==========================================
# 5. 结论输出与走势标注 (修复图表中文)
# ==========================================
st.divider()
st.subheader("💡 最终操作建议与走势标注")

# ... [结论判定逻辑保持不变] ...

st.markdown("#### 📅 中证500 (sh000905) 走势与信号标注 (2024至今)")
df_plot = df_final.loc["2024-01-01":]
fig3, ax3 = plt.subplots(figsize=(16, 8))
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.5, label='收盘价')
ax3.plot(df_plot.index, df_plot['MA30'], color='blue', linestyle='--', alpha=0.4, label='MA30趋势线')

# 标注买卖点
buys = df_plot[df_plot['signal'] == 1]
ax3.scatter(buys.index, buys['close'], color='red', marker='^', s=120, zorder=5, label='买入(战略/战术)')
sells = df_plot[df_plot['signal'] == -1]
ax3.scatter(sells.index, sells['close'], color='green', marker='v', s=120, zorder=5, label='卖出(复合止损)')

# --- ⚡ 显式设置图例，防止乱码 ---
ax3.legend(loc='upper left', prop={'size': 12})
ax3.grid(True, alpha=0.2)
st.pyplot(fig3)

# ==========================================
# 6. 决策逻辑判定详情 (保持原样)
# ==========================================
# [此处代码与之前相同：st.expander 部分]
