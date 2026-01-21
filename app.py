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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心数据加载 (加速版)
# ==========================================
@st.cache_data(ttl=60)  # 缓存60秒，平衡实时性与速度
def load_data():
    # 1. 加载扫描结果（广度数据）
    df_scan = pd.read_csv("scan_results.csv", index_col='date', parse_dates=True)
    
    # 2. 加载中证500底表
    df_main = pd.read_csv("CSI500_Master_Strategy.csv", index_col='date', parse_dates=True)
    
    # 3. 加载汇总换手率 (读取 master_summary.csv)
    other_turnovers = {}
    default_etfs = ["SSE50", "CSI300", "CSI1000"]
    for etf in default_etfs: other_turnovers[etf] = 0.0
        
    try:
        if os.path.exists("master_summary.csv"):
            df_sum = pd.read_csv("master_summary.csv")
            for _, row in df_sum.iterrows():
                label = row['Index_Label']
                val = row['ETF_Turnover']
                other_turnovers[label] = val if val > 1 else val * 100
    except Exception as e:
        st.error(f"读取汇总表失败: {e}")

    return df_scan, df_main, other_turnovers

# --- ⚡ 核心修复：执行数据加载 ---
try:
    df_scan, df_main, other_turnovers = load_data()
except Exception as e:
    st.error(f"❌ 数据加载失败，请检查 CSV 文件是否在 GitHub 根目录: {e}")
    st.stop()

# ==========================================
# 2. 旗舰进化版逻辑计算引擎
# ==========================================
def calculate_signals(df, df_breadth):
    df = df.copy()
    # 基础指标
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA30'] = df['close'].rolling(30).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 广度指标合并 (确保索引对齐)
    df = df.join(df_breadth[['ma20_ratio', 'new_high_ratio']], how='left')
    
    # 资金热度 Z-Score
    df['Vol_MA60'] = df['volume'].rolling(60).mean()
    df['Vol_STD60'] = df['volume'].rolling(60).std()
    df['Heat_Z'] = (df['volume'] - df['Vol_MA60']) / df['Vol_STD60']
    
    # 连阳特征
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Consec_Gains'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Consec_Gains'], 0)
    
    # 信号生成逻辑
    df['signal'] = 0
    df['logic_type'] = ""
    
    in_pos = False
    logic_state = ""
    entry_high = 0
    hold_days = 0

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # 买入逻辑判定
        if not in_pos:
            # 战略：极度超跌
            if curr['ma20_ratio'] < 16:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Strategic"
                in_pos, logic_state, hold_days = True, "Strategic", 0
            # 战术：旗舰进化首阴
            elif (curr['close'] > curr['MA30'] and curr['close'] > curr['MA10'] and 
                  prev['Consec_Gains'] >= 3 and curr['close'] < prev['close'] and 
                  (curr['ETF_Turnover'] if curr['ETF_Turnover']>1 else curr['ETF_Turnover']*100) > 1.0 and 
                  curr['close'] > curr['MA5']):
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Tactical"
                in_pos, logic_state, hold_days = True, "Tactical", 0
                entry_high = curr['high']
        
        # 退出逻辑判定
        else:
            hold_days += 1
            is_overheat = curr['ma20_ratio'] > 79 and curr['Heat_Z'] < 1.5
            exit_flag = False
            
            if logic_state == "Strategic":
                if is_overheat: exit_flag = True
            else: # Tactical 复合止损
                is_below_ma30 = curr['close'] < curr['MA30']
                is_1d_drop = curr['close'] < prev['close']
                is_5d_no_high = (hold_days >= 5 and curr['close'] < entry_high)
                if is_overheat or (is_below_ma30 and (is_1d_drop or is_5d_no_high)):
                    exit_flag = True
            
            if exit_flag:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                in_pos, logic_state = False, ""
                
    return df

# 计算并获取最新状态
df_final = calculate_signals(df_main, df_scan)
last_data = df_final.iloc[-1]

# ==========================================
# 3. 页面渲染逻辑 (省略部分重复布局代码以保持简洁)
# ==========================================
# [此处保留你原有的布局 col1, col2, 诊断报告, 结论输出等代码]
# 注意：t3 指标请使用: f"{last_data['ETF_Turnover'] if last_data['ETF_Turnover']>1 else last_data['ETF_Turnover']*100:.2f}%"

# ==========================================
# 4. K线标注增强
# ==========================================
st.subheader("💡 最终操作建议与走势标注")
# ... (保留你的结论判定逻辑)

fig3, ax3 = plt.subplots(figsize=(16, 8))
df_plot = df_final.loc["2024-01-01":]
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.4, label='CSI500 Close')
ax3.plot(df_plot.index, df_plot['MA30'], color='blue', linestyle='--', alpha=0.5, label='MA30')

# 标注信号
buys = df_plot[df_plot['signal'] == 1]
sells = df_plot[df_plot['signal'] == -1]
ax3.scatter(buys.index, buys['close'], color='red', marker='^', s=100, label='Buy Signal')
ax3.scatter(sells.index, sells['close'], color='green', marker='v', s=100, label='Sell Signal')

ax3.legend()
st.pyplot(fig3)
