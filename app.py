import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================
# 0. 页面配置与字体修复 (暴力适配版)
# ==========================================
st.set_page_config(page_title="量化大师-旗舰进化版", layout="wide")
st.title("🛡️ 量化大师：MA30过滤旗舰进化版综合看板")

def set_chinese_font():
    font_list = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif']
    plt.rcParams['font.sans-serif'] = font_list + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# ==========================================
# 1. 数据加载逻辑
# ==========================================
@st.cache_data(ttl=60)
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
    st.error(f"⚠️ 数据加载失败，请确保已运行 runscan.py 并成功上传至 GitHub: {e}")
    st.stop()

# ==========================================
# 2. 旗舰进化逻辑计算引擎（完全对齐回测 fusion_ma 版本）
# ==========================================
def calculate_flagship_signals(df_price, df_breadth):
    df = df_price.copy()
    # 基础均线
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA30'] = df['close'].rolling(30).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 合并广度数据并填充
    df = df.join(df_breadth[['ma20_ratio', 'new_high_ratio']], how='left').ffill()
    
    # 【对齐回测】Heat_Z 使用 amount rolling(20)
    amt = df['amount']
    df['Heat_Z'] = ((amt - amt.rolling(20).mean()) / amt.rolling(20).std()).ffill().fillna(0)
    
    # 计算连阳特征
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Consec_Gains'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Consec_Gains'], 0)
    
    # 换手率格式统一
    df['Turnover_Pct'] = np.where(df['ETF_Turnover'] > 1, df['ETF_Turnover'], df['ETF_Turnover'] * 100)
    
    # 信号与标记列
    df['signal'] = 0          # 1=买入, -1=卖出
    df['logic_type'] = ""     # 显示用：Strategic / Tactical
    df['upgrade'] = 0         # 1=战术单身份升级为战略风控（标记青色圆圈）
    
    in_pos = False
    logic_state = ""          # 内部风控状态： "Strategic" (Composite，只宏观卖) / "Tactical" (可复合止损)
    entry_high = 0
    hold_days = 0
    max_close_since_entry = 0

    for i in range(1, len(df)):
        curr, prev = df.iloc[i], df.iloc[i-1]
        
        # 卖出逻辑
        if in_pos:
            hold_days += 1
            max_close_since_entry = max(max_close_since_entry, curr['close'])
            
            # 【关键对齐】身份升级：战术单遇到冰点条件时升级为战略风控（只接受宏观过热卖出）
            if logic_state == "Tactical" and curr['ma20_ratio'] < 16:
                logic_state = "Strategic"
                df.iloc[i, df.columns.get_loc('upgrade')] = 1  # 标记升级点
            
            # 宏观过热条件
            is_macro_exit = (curr['ma20_ratio'] > 79) and (curr['Heat_Z'] < 1.5)
            
            exit_flag = False
            if logic_state == "Strategic":
                # 战略/升级后：仅宏观过热卖出
                if is_macro_exit:
                    exit_flag = True
            else:
                # 战术：宏观过热 OR (破MA30 + (收阴 OR 5日无新高))
                is_trend_broken = curr['close'] < curr['MA30']
                is_yin = curr['close'] < prev['close']
                is_time_stop = (hold_days >= 5) and (max_close_since_entry <= entry_high)
                if is_macro_exit or (is_trend_broken and (is_yin or is_time_stop)):
                    exit_flag = True
            
            if exit_flag:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                in_pos = False
                logic_state = ""
                hold_days = 0
        
        # 买入逻辑
        else:
            # 战略买入
            if curr['ma20_ratio'] < 16:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Strategic"
                in_pos = True
                logic_state = "Strategic"
                hold_days = 0
                entry_high = curr['high']
                max_close_since_entry = curr['close']
            # 战术买入（完全对齐 fusion_ma）
            elif (curr['close'] > curr['MA30'] and   # MA30趋势过滤
                  curr['close'] > curr['MA10'] and   # 短期支撑
                  curr['close'] > curr['MA5'] and    # 攻击形态
                  prev['Consec_Gains'] >= 3 and      # 此前连阳≥3天
                  curr['close'] < prev['close'] and  # 今日首阴
                  curr['Turnover_Pct'] > 1.0):       # 量能活跃
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Tactical"
                in_pos = True
                logic_state = "Tactical"
                hold_days = 0
                entry_high = curr['high']
                max_close_since_entry = curr['close']
    
    return df

df_final = calculate_flagship_signals(df_main, df_scan)
last_row = df_final.iloc[-1]

# ==========================================
# 3. 页面布局与看板
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
m1.metric("市场模式", "📈 多头" if last_row['MA20'] > last_row['MA60'] else "📉 空头")
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
# 4. 结论与K线标注（所有买卖点均标注 + 身份升级标记）
# ==========================================
st.divider()
st.subheader("💡 最终操作建议与走势标注")

sig = last_row['signal']
log_type = last_row['logic_type']
if sig == 1:
    if log_type == "Strategic":
        st.success("🚀 **操作建议：买入 (战略单)** | 触发宏观冰点极值，胜率极高。")
    else:
        st.success("🔥 **操作建议：买入 (战术单)** | 触发趋势中继首阴回踩，爆发力强。")
elif sig == -1:
    st.error("🚨 **操作建议：清仓/减仓** | 触发复合止损逻辑（宏观过热或趋势破位）。")
else:
    st.info("✅ **操作建议：持股/观望** | 当前无新信号触发，按原有策略持有。")

st.markdown("#### 📅 中证500 (sh000905) 走势与信号标注 (2024至今)")
df_plot = df_final.loc["2024-01-01":]
fig3, ax3 = plt.subplots(figsize=(16, 8))
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.5, label='Close Price')
ax3.plot(df_plot.index, df_plot['MA30'], color='blue', linestyle='--', label='MA30 Trend')

# 所有买入点（红↑）
b_pts = df_plot[df_plot['signal'] == 1]
ax3.scatter(b_pts.index, b_pts['close'], color='red', marker='^', s=120, zorder=5, label='Buy Signal')

# 所有卖出点（绿↓）
s_pts = df_plot[df_plot['signal'] == -1]
ax3.scatter(s_pts.index, s_pts['close'], color='green', marker='v', s=120, zorder=5, label='Sell Signal')

# 身份升级点（青色圆圈）
u_pts = df_plot[df_plot['upgrade'] == 1]
ax3.scatter(u_pts.index, u_pts['close'], color='cyan', marker='o', s=100, edgecolors='black', zorder=6, label='Upgrade to Strategic')

ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.2)
st.pyplot(fig3)

# ==========================================
# 5. 决策逻辑详情 (详细版，同步更新说明)
# ==========================================
with st.expander("查看【MA30过滤版 旗舰进化】决策逻辑判定详情", expanded=True):
    st.markdown("""
    ### ⚔️ 核心策略体系详解（已完全对齐回测 fusion_ma 版本）

    本策略采用**“战略 (Strategic) + 战术 (Tactical) + 身份升级融合”**机制。

    ---

    #### ✅ 一、买入逻辑

    **1. 战略买入**：广度 < 16%（冰点抄底），风控最严格（仅宏观过热卖出）。

    **2. 战术买入**：连阳≥3天后首阴回踩 + 多均线支撑（>MA5/10/30）+ 换手>1%。

    ---

    #### 🔄 身份升级（融合机制）
    * 战术单持仓中若再次触发广度 < 16%，自动升级为**战略风控**（之后仅接受宏观过热卖出，不再接受趋势破位止损）。

    ---

    #### 🛑 二、卖出逻辑（复合止损）

    **1. 宏观过热退出**（适用于所有仓位）：
    * 广度 > 79% 且 Heat_Z < 1.5。

    **2. 趋势破位退出**（**仅战术单**，升级后失效）：
    * 跌破 MA30 且（今日收阴 或 持仓≥5天且期间收盘未创新高（未超过买入日最高价））。

    """)
