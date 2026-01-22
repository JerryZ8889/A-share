import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 页面配置与字体修复
# ==========================================
st.set_page_config(page_title="量化大师-100%全功能对齐版", layout="wide")
st.title("🛡️ 量化大师：MA30过滤旗舰 (信号+收益全量同步版)")

def set_chinese_font():
    font_list = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif']
    plt.rcParams['font.sans-serif'] = font_list + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# ==========================================
# 1. 核心数据加载 (ttl=0 确保调试时数据实时)
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
# 2. 仿真引擎：逻辑 100% 对齐回测脚本
# ==========================================
def calculate_synchronized_signals(df_p, df_b):
    temp = df_p.copy()
    if 'ma20_ratio' in df_b.columns:
        temp = temp.join(df_b[['ma20_ratio']], how='left').ffill()
        temp.rename(columns={'ma20_ratio': 'breadth'}, inplace=True)
    
    # 指标计算
    temp['MA_Filter'] = temp['close'].rolling(30).mean() 
    temp['MA_Support'] = temp['close'].rolling(5).mean()
    temp['MA_Trend'] = temp['close'].rolling(10).mean()
    temp['MA60'] = temp['close'].rolling(60).mean()
    
    temp['Is_Up'] = (temp['close'] > temp['close'].shift(1)).astype(int)
    temp['Streak'] = temp['Is_Up'].groupby((temp['Is_Up'] != temp['Is_Up'].shift()).cumsum()).cumcount() + 1
    temp['Consec_Gains'] = np.where(temp['Is_Up'] == 1, temp['Streak'], 0)
    
    target_col = 'amount' if 'amount' in temp.columns else 'volume'
    temp['Heat_Z'] = (temp[target_col] - temp[target_col].rolling(20).mean()) / temp[target_col].rolling(20).std()
    
    t_raw = temp['ETF_Turnover']
    temp['Turnover_Pct'] = np.where(t_raw.max() > 1, t_raw, t_raw * 100)

    # 判定条件
    cond_comp_b = (temp['breadth'] < 16)
    cond_comp_s = (temp['breadth'] > 79) & (temp['Heat_Z'] < 1.5)
    cond_fn_b_base = (temp['close'] > temp['MA_Trend']) & \
                     (temp['Consec_Gains'].shift(1) >= 3) & \
                     (temp['close'] < temp['close'].shift(1)) & \
                     (temp['Turnover_Pct'] > 1.0) & \
                     (temp['close'] > temp['MA_Support'])

    # 循环仿真
    temp['pos'] = 0; temp['signal'] = 0; temp['logic_type'] = ""; temp['marker'] = ""
    in_pos = False; logic_state = ""; entry_idx = 0; entry_high = 0

    for i in range(len(temp)):
        if i == 0: continue
        current_close = temp['close'].iloc[i]
        prev_close = temp['close'].iloc[i-1]
        current_ma30 = temp['MA_Filter'].iloc[i]
        
        if in_pos:
            if logic_state == "FirstNeg" and cond_comp_b.iloc[i]:
                logic_state = "Composite"; temp.iloc[i, temp.columns.get_loc('marker')] = "升级"

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
        else:
            buy_triggered = False
            if cond_comp_b.iloc[i]: 
                temp.iloc[i, temp.columns.get_loc('logic_type')] = "Strategic"
                logic_state = "Composite"; buy_triggered = True
            elif cond_fn_b_base.iloc[i] and (current_close > current_ma30):
                temp.iloc[i, temp.columns.get_loc('logic_type')] = "Tactical"
                logic_state = "FirstNeg"; buy_triggered = True
            
            if buy_triggered:
                temp.iloc[i, temp.columns.get_loc('signal')] = 1
                temp.iloc[i, temp.columns.get_loc('pos')] = 1
                in_pos = True; entry_idx = i; entry_high = temp['high'].iloc[i]

    # --- 净值收益计算逻辑 (新增) ---
    actual_pos = temp['pos'].shift(1).fillna(0) # 交易发生在信号次日
    ret = temp['close'].pct_change().fillna(0)
    # 费率 0.1%，仅在仓位变化时收取
    fee = np.where(actual_pos.diff() != 0, 0.001, 0)
    temp['cum_ret'] = (1 + actual_pos * ret - fee).cumprod()
    temp['benchmark_cum'] = (1 + ret).cumprod()

    return temp

df_final = calculate_synchronized_signals(df_main, df_scan)
last_row = df_final.iloc[-1]

# ==========================================
# 3. UI 渲染：顶层看板
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
    axl.plot(df_scan.index, df_scan['ma20_ratio'], color='tab:blue')
    axr = axl.twinx()
    axr.bar(df_scan.index, df_scan['new_high_ratio'], color='tab:orange', alpha=0.3)
    st.pyplot(fig2)

st.divider()
st.subheader("🛡️ 动态逻辑诊断报告")
m1, m2, m3 = st.columns(3)
m1.metric("市场模式", "📈 多头" if last_row['close'] > last_row['MA60'] else "📉 空头")
m2.metric("资金热度 (Z)", f"{last_row['Heat_Z']:.2f}")
m3.metric("净值涨幅", f"{(last_row['cum_ret']-1)*100:.2f}%")

# ==========================================
# 4. 收益曲线图 (新增：1:1 对齐代码 2)
# ==========================================
st.markdown("#### 📈 策略收益曲线与基准对比 (MA30 同步版策略)")
df_plot = df_final.loc["2024-01-01":]
# 重新归一化绘图区起点为 1.0
df_plot_norm = df_plot.copy()
df_plot_norm['cum_ret'] = df_plot_norm['cum_ret'] / df_plot_norm['cum_ret'].iloc[0]
df_plot_norm['benchmark_cum'] = df_plot_norm['benchmark_cum'] / df_plot_norm['benchmark_cum'].iloc[0]

fig_ret, ax_ret = plt.subplots(figsize=(16, 6))
ax_ret.plot(df_plot_norm.index, df_plot_norm['benchmark_cum'], label='中证500基准', color='gray', alpha=0.4, linestyle='--')
ax_ret.plot(df_plot_norm.index, df_plot_norm['cum_ret'], label='MA30同步版策略', color='crimson', linewidth=2.5)

# 在收益曲线上同步标注买卖信号点
buys_ret = df_plot_norm[df_plot_norm['signal'] == 1]
ax_ret.scatter(buys_ret.index, df_plot_norm.loc[buys_ret.index, 'cum_ret'], color='red', marker='^', s=120, zorder=5)
sells_ret = df_plot_norm[df_plot_norm['signal'] == -1]
ax_ret.scatter(sells_ret.index, df_plot_norm.loc[sells_ret.index, 'cum_ret'], color='green', marker='v', s=120, zorder=5)

ax_ret.legend(loc='upper left'); ax_ret.grid(True, alpha=0.1)
st.pyplot(fig_ret)

# ==========================================
# 5. K线标注图
# ==========================================
st.divider()
st.markdown("#### 📅 中证500 (sh000905) 价格走势与信号")
fig3, ax3 = plt.subplots(figsize=(16, 7))
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.5, label='Price')
ax3.plot(df_plot.index, df_plot['MA_Filter'], color='blue', linestyle='--', label='MA30 Filter')

buys = df_plot[df_plot['signal'] == 1]
ax3.scatter(buys.index, buys['close'], color='red', marker='^', s=150, zorder=10, label='Buy')
sells = df_plot[df_plot['signal'] == -1]
ax3.scatter(sells.index, sells['close'], color='green', marker='v', s=150, zorder=10, label='Sell')
upgrades = df_plot[df_plot['marker'] == "升级"]
ax3.scatter(upgrades.index, upgrades['close'], color='orange', marker='o', s=80, alpha=0.6, label='Upgrade')

ax3.legend(loc='upper left'); ax3.grid(True, alpha=0.1)
st.pyplot(fig3)

# ==========================================
# 6. 市场广度遮罩图
# ==========================================
st.markdown("#### 🌊 市场广度波动环境与策略持仓状态")
fig4, ax4 = plt.subplots(figsize=(16, 3))
ax4.plot(df_plot.index, df_plot['breadth'], color='orange', label='Market Breadth')
ax4.fill_between(df_plot.index, 0, 100, where=(df_plot['pos']==1), color='blue', alpha=0.1, label='Holding')
ax4.axhline(y=16, color='red', linestyle='--', alpha=0.5); ax4.axhline(y=79, color='green', linestyle='--', alpha=0.5)
ax4.set_ylim(0, 100); ax4.legend(loc='upper left')
st.pyplot(fig4)
