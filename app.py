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
# 1. 核心数据加载
# ==========================================
@st.cache_data(ttl=0)
def load_data():
    # 加载扫描结果（广度）
    df_scan = pd.read_csv("scan_results.csv", index_col='date', parse_dates=True)
    # 加载中证500底表
    df_500 = pd.read_csv("CSI500_Master_Strategy.csv", index_col='date', parse_dates=True)
    # 加载其他指数换手率用于共振分析
    etf_files = {
        "SSE50": "SSE50_Master_Strategy.csv",
        "CSI300": "CSI300_Master_Strategy.csv",
        "CSI1000": "CSI1000_Master_Strategy.csv"
    }
    other_turnovers = {}
    for k, v in etf_files.items():
        if os.path.exists(v):
            tdf = pd.read_csv(v)
            val = tdf['ETF_Turnover'].iloc[-1]
            other_turnovers[k] = val if val > 1 else val * 100
    return df_scan, df_500, other_turnovers

try:
    df_scan, df_main, other_turnovers = load_data()
    st.success(f"✅ 数据同步成功！最新数据日期：{df_main.index[-1].strftime('%Y-%m-%d')}")
except Exception as e:
    st.error(f"❌ 数据同步失败，请检查GitHub文件是否齐全: {e}")
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
    
    # 广度指标合并
    df = df.join(df_breadth[['ma20_ratio', 'new_high_ratio']], how='left')
    
    # 资金热度 Z-Score
    df['Vol_MA60'] = df['volume'].rolling(60).mean()
    df['Vol_STD60'] = df['volume'].rolling(60).std()
    df['Heat_Z'] = (df['volume'] - df['Vol_MA60']) / df['Vol_STD60']
    
    # 首阴特征
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Consec_Gains'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Consec_Gains'], 0)
    
    # 仿真买卖点（用于画图）
    df['signal'] = 0  # 1: 买入, -1: 卖出
    df['logic_type'] = "" # Strategic 或 Tactical
    
    in_pos = False
    logic_state = "" # "Strategic" 或 "Tactical"
    entry_high = 0
    hold_days = 0

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # 1. 宏观/战略买入信号 (广度冰点)
        cond_strategic_buy = curr['ma20_ratio'] < 16
        
        # 2. 战术/首阴买入信号 (旗舰进化版)
        cond_tactical_buy = (
            curr['close'] > curr['MA30'] and 
            curr['close'] > curr['MA10'] and 
            prev['Consec_Gains'] >= 3 and 
            curr['close'] < prev['close'] and 
            (curr['ETF_Turnover'] if curr['ETF_Turnover']>1 else curr['ETF_Turnover']*100) > 1.0 and 
            curr['close'] > curr['MA5']
        )
        
        # 卖出逻辑判断
        if in_pos:
            hold_days += 1
            is_overheat = curr['ma20_ratio'] > 79 and curr['Heat_Z'] < 1.5
            exit_flag = False
            
            if logic_state == "Strategic":
                if is_overheat: exit_flag = True
            else: # Tactical
                is_below_ma30 = curr['close'] < curr['MA30']
                is_1d_drop = curr['close'] < prev['close']
                is_5d_no_high = (hold_days >= 5 and curr['close'] < entry_high)
                if is_overheat or (is_below_ma30 and (is_1d_drop or is_5d_no_high)):
                    exit_flag = True
            
            if exit_flag:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                in_pos = False
                logic_state = ""
        
        # 买入执行
        else:
            if cond_strategic_buy:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Strategic"
                in_pos, logic_state, hold_days = True, "Strategic", 0
            elif cond_tactical_buy:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Tactical"
                in_pos, logic_state, hold_days = True, "Tactical", 0
                entry_high = curr['high']
                
    return df

df_final = calculate_signals(df_main, df_scan)
last_data = df_final.iloc[-1]

# ==========================================
# 3. 页面布局：资金热度与广度面板 (保持原样)
# ==========================================
col1, col2 = st.columns(2)
with col1:
    st.subheader("🔥 资金热度 (Z-Score)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_data = df_final['Heat_Z'].tail(100)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data>=0), color='red', alpha=0.3)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    st.pyplot(fig1)

with col2:
    st.subheader("📊 市场广度趋势")
    fig2, ax_l = plt.subplots(figsize=(10, 5))
    ax_l.plot(df_final.index[-100:], df_final['ma20_ratio'].tail(100), color='tab:blue', marker='o', label='MA20 %')
    ax_l.set_ylim(0, 100)
    ax_r = ax_l.twinx()
    ax_r.bar(df_final.index[-100:], df_final['new_high_ratio'].tail(100), color='tab:orange', alpha=0.3)
    st.pyplot(fig2)

# ==========================================
# 4. 动态逻辑诊断报告
# ==========================================
st.divider()
st.subheader("🛡️ 动态逻辑诊断报告")
m1, m2, m3 = st.columns(3)
is_bull = last_data['MA20'] > last_data['MA60']
m1.metric("市场模式", "📈 多头趋势" if is_bull else "📉 空头趋势")
m2.metric("资金热度 (Z)", f"{last_data['Heat_Z']:.2f}")
m3.metric("市场宽度 (MA20%)", f"{last_data['ma20_ratio']:.1f}%")

st.write("🔥 **全市场量能监测**")
t1, t2, t3, t4 = st.columns(4)
t1.metric("上证50", f"{other_turnovers.get('SSE50',0):.2f}%")
t2.metric("沪深300", f"{other_turnovers.get('CSI300',0):.2f}%")
t3.metric("中证500", f"{last_data['ETF_Turnover'] if last_data['ETF_Turnover']>1 else last_data['ETF_Turnover']*100:.2f}%")
t4.metric("中证1000", f"{other_turnovers.get('CSI1000',0):.2f}%")

# ==========================================
# 5. 最终结论与日K线标注
# ==========================================
st.divider()
st.subheader("💡 最终操作建议与走势标注")

# 逻辑判定
curr_buy_signal = last_data['signal'] == 1
curr_logic = last_data['logic_type']

if curr_buy_signal:
    if curr_logic == "Strategic":
        st.success("🚀 **综合结论：战略级买入！** 全市场进入广度冰点区域，宏观赔率极高。")
    else:
        st.success("🔥 **综合结论：战术级加仓！** 满足MA30过滤+首阴回踩，短期爆发力强。")
elif last_data['signal'] == -1:
    st.error("🚨 **综合结论：立刻减仓！** 触发旗舰版复合止损逻辑，保护利润/规避风险。")
else:
    st.info("✅ **综合结论：观望或持股。** 目前未触发新的买卖信号。")

# 中证500 K线图
st.markdown("#### 📅 中证500 走势与信号回顾 (2024至今)")
df_plot = df_final.loc["2024-01-01":]
fig3, ax3 = plt.subplots(figsize=(16, 8))
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.6, label='中证500收盘价')
ax3.plot(df_plot.index, df_plot['MA30'], color='blue', linestyle='--', alpha=0.4, label='MA30趋势线')

# 标注买入
buys = df_plot[df_plot['signal'] == 1]
ax3.scatter(buys.index, buys['close'], color='red', marker='^', s=100, label='买入点')
# 标注卖出
sells = df_plot[df_plot['signal'] == -1]
ax3.scatter(sells.index, sells['close'], color='green', marker='v', s=100, label='卖出点')

ax3.legend()
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

# ==========================================
# 6. 决策逻辑判定详情
# ==========================================
with st.expander("查看【MA30过滤版旗舰进化】决策逻辑详情"):
    st.write(f"""
    **1. 战略买入 (Strategic Buy)**:
    - 核心条件：市场广度 (MA20 Ratio) < 16% (当前: {last_data['ma20_ratio']:.1f}%)
    - 逻辑：全市场极度超跌，属于宏观底部的左侧博弈。

    **2. 战术买入 (Tactical Buy - 旗舰进化)**:
    - MA30过滤器：价格 > MA30 (当前: {'满足' if last_data['close']>last_data['MA30'] else '不满足'})
    - 首阴形态：此前连阳 >= 3天，今日收阴。
    - 活跃度：ETF换手率 > 1.0% (当前: {last_data['ETF_Turnover'] if last_data['ETF_Turnover']>1 else last_data['ETF_Turnover']*100:.2f}%)
    - 防御位：价格 > MA5 且 > MA10。

    **3. 复合止损 (Composite Exit)**:
    - 战略单：仅在宏观过热 (广度>79% 且 资金热度衰减) 时退出。
    - 战术单：若价格在 MA30 下方，满足 (今日下跌) 或 (5日不创新高) 即刻退出。
    """)
