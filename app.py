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
# 1. 核心数据加载模块
# ==========================================

@st.cache_data(ttl=0) # 生产环境下建议设为60秒
def load_all_data():
    """集中加载所有必要数据"""
    # 1. 指数日线 (用于热度计算)
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    
    # 2. 市场广度结果
    df_scan = pd.read_csv("scan_results.csv", index_col='date', parse_dates=True).sort_index()
    
    # 3. 中证500主策略表 (包含换手率)
    df_main = pd.read_csv("CSI500_Master_Strategy.csv", index_col='date', parse_dates=True).sort_index()
    
    # 4. 全市场汇总表 (用于看板)
    df_summary = pd.read_csv("master_summary.csv") if os.path.exists("master_summary.csv") else pd.DataFrame()
    
    return df_idx, df_scan, df_main, df_summary

try:
    df_idx, df_scan, df_main, df_summary = load_all_data()
    st.success(f"✅ 数据同步成功！最新数据日期：{df_main.index[-1].strftime('%Y-%m-%d')}")
except Exception as e:
    st.error(f"⚠️ 数据加载失败: {e}")
    st.stop()

# ==========================================
# 2. 旗舰进化逻辑计算引擎
# ==========================================
def calculate_flagship_signals(df_price, df_breadth):
    df = df_price.copy()
    # 基础指标计算
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA30'] = df['close'].rolling(30).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # 合并广度
    df = df.join(df_breadth[['ma20_ratio', 'new_high_ratio']], how='left').fillna(method='ffill')
    
    # 计算连阳
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Consec_Gains'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Consec_Gains'], 0)
    
    # 换手率纠偏
    df['Turnover_Pct'] = np.where(df['ETF_Turnover'] > 1, df['ETF_Turnover'], df['ETF_Turnover'] * 100)
    
    # 计算热度 Z-Score
    vol = df['volume']
    df['Heat_Z'] = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    
    # --- 信号仿真循环 ---
    df['signal'] = 0  # 1: 买入, -1: 卖出
    df['logic'] = ""
    in_pos = False
    logic_state = ""
    entry_high = 0
    hold_days = 0

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # 卖出判定
        if in_pos:
            hold_days += 1
            is_macro_s = (curr['ma20_ratio'] > 79) and (curr['Heat_Z'] < 1.5)
            exit_flag = False
            
            if logic_state == "Strategic":
                if is_macro_s: exit_flag = True
            else: # Tactical 战术退出
                is_below_ma30 = curr['close'] < curr['MA30']
                is_drop = curr['close'] < prev['close']
                is_5d = (hold_days >= 5) and (curr['close'] < entry_high)
                if is_macro_s or (is_below_ma30 and (is_drop or is_5d)): 
                    exit_flag = True
            
            if exit_flag:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                in_pos, logic_state = False, ""
        
        # 买入判定
        else:
            # 1. 战略买入 (冰点)
            if curr['ma20_ratio'] < 16:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic')] = "Strategic"
                in_pos, logic_state, hold_days = True, "Strategic", 0
            # 2. 战术买入 (MA30过滤+首阴进化)
            elif (curr['close'] > curr['MA30'] and curr['close'] > curr['MA10'] and 
                  curr['close'] > curr['MA5'] and prev['Consec_Gains'] >= 3 and 
                  curr['close'] < prev['close'] and curr['Turnover_Pct'] > 1.0):
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic')] = "Tactical"
                in_pos, logic_state, hold_days = True, "Tactical", 0
                entry_high = curr['high']
                
    return df

df_final = calculate_flagship_signals(df_main, df_scan)
last_row = df_final.iloc[-1]

# ==========================================
# 3. 布局：左右双图 (面板布局不改动)
# ==========================================
col_heat, col_breadth = st.columns(2)

with col_heat:
    st.subheader("🔥 资金热度 (Z-Score)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_data = df_final['Heat_Z'].tail(100)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data>=0), color='red', alpha=0.3)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col_breadth:
    st.subheader("📊 市场广度 (全量历史趋势)")
    fig2, ax_l = plt.subplots(figsize=(10, 5))
    ax_l.plot(df_scan.index, df_scan['ma20_ratio'], color='tab:blue', marker='o', linewidth=2, label='MA20 %')
    ax_l.set_ylim(0, 100)
    ax_l.set_ylabel('Above MA20 (%)', color='tab:blue')
    ax_r = ax_l.twinx()
    ax_r.bar(df_scan.index, df_scan['new_high_ratio'], color='tab:orange', alpha=0.4)
    ax_r.set_ylabel('New High (%)', color='tab:orange')
    plt.xticks(rotation=45)
    fig2.tight_layout()
    st.pyplot(fig2)

# ==========================================
# 4. 诊断报告看板 (不改动内容)
# ==========================================
st.divider()
st.subheader("🛡️ 动态逻辑诊断报告")

# 4.1 核心指标矩阵
m1, m2, m3 = st.columns(3)
is_bull = last_row['MA20'] > last_row['MA60']
m1.metric("市场模式", "📈 多头 (Bull)" if is_bull else "📉 空头 (Bear)")
m2.metric("资金热度 (Z)", f"{last_row['Heat_Z']:.2f}")
m3.metric("市场宽度 (MA20%)", f"{last_row['ma20_ratio']:.1f}%")

# 4.2 全市场换手率监测
st.write("🔥 **全市场量能共振监测 (实时换手率)**")
t1, t2, t3, t4 = st.columns(4)
# 从 summary 获取
def get_t(label):
    if not df_summary.empty:
        v = df_summary[df_summary['Index_Label'] == label]['ETF_Turnover'].values
        if len(v) > 0: return v[0] if v[0] > 1 else v[0] * 100
    return 0.0

t1.metric("上证50", f"{get_t('SSE50'):.2f}%")
t2.metric("沪深300", f"{get_t('CSI300'):.2f}%")
t3.metric("中证500", f"{last_row['Turnover_Pct']:.2f}%")
t4.metric("中证1000", f"{get_t('CSI1000'):.2f}%")

st.info(f"**模式分析**：{'📈 当前处于中长期上涨趋势中，策略容错率较高' if is_bull else '📉 当前处于空头或调整环境，战术操作需严控止损'}")

# ==========================================
# 5. 最终结论输出 (替换原策略A/B)
# ==========================================
st.divider()
st.subheader("💡 最终操作建议 (旗舰进化版)")

# 获取当前信号
curr_sig = last_row['signal']
curr_logic = last_row['logic']

if curr_sig == 1:
    if curr_logic == "Strategic":
        st.warning("🚀 **综合结论：战略单触发！** 市场进入全量广度冰点区域。这是宏观维度的左侧建仓信号，胜率极高，建议重仓布局。")
    else:
        st.success("🔥 **综合结论：战术单触发！** 趋势向好（MA30之上）且满足首阴回踩。这是一个典型的上涨中继买点，建议积极参与。")
elif curr_sig == -1:
    st.error("🚨 **综合结论：立刻清仓/减仓！** 满足旗舰版复合退出逻辑（宏观过热或趋势走弱），请严格执行纪律，规避回撤。")
else:
    # 维持现状
    if last_row['ma20_ratio'] > 70:
        st.info("⌛ **当前状态：持股观望。** 广度进入高位区，不宜新开仓，密切关注卖点信号。")
    elif last_row['ma20_ratio'] < 30:
        st.info("⌛ **当前状态：空仓等待。** 市场仍处于弱势寻底阶段，等待冰点或趋势反转。")
    else:
        st.write("✅ **综合结论：目前市场处于平稳期**。建议按原有比例持仓，不触发逻辑不操作。")

# ==========================================
# 6. K线可视化 (2024至今)
# ==========================================
st.markdown("#### 📅 中证500 (sh000905) 走势与信号标注")
plot_start = "2024-01-01"
df_plot = df_final.loc[plot_start:]

fig3, ax3 = plt.subplots(figsize=(16, 8))
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.6, label='中证500收盘价')
ax3.plot(df_plot.index, df_plot['MA30'], color='blue', linestyle='--', alpha=0.4, label='MA30趋势过滤器')

# 标注买点
buys = df_plot[df_plot['signal'] == 1]
ax3.scatter(buys.index, buys['close'], color='red', marker='^', s=130, label='买入点 (战略/战术)')
# 标注卖点
sells = df_plot[df_plot['signal'] == -1]
ax3.scatter(sells.index, sells['close'], color='green', marker='v', s=130, label='卖出点 (复合止损)')

ax3.set_title(f"中证500策略回顾 ({plot_start} 至今)", fontsize=15)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.2)
st.pyplot(fig3)

# ==========================================
# 7. 逻辑详情说明
# ==========================================
with st.expander("查看【MA30过滤版 旗舰进化】决策逻辑判定详情"):
    st.markdown(f"""
    ### 1. 买入逻辑 (双轨制)
    * **战略买入 (Strategic)**：
        * 当 **市场广度 (Above MA20%) < 16%** 时触发。
        * *逻辑*：宏观冰点，此时全市场极度超跌，属于高胜率左侧机会。
    * **战术买入 (Tactical - 旗舰版)**：
        * **趋势过滤**：价格必须在 **MA30** 之上。
        * **形态要求**：价格在 MA10 和 MA5 之上，且经历过连续 3 日及以上上涨后，今日首次收阴（收盘价 < 昨收）。
        * **量能配合**：ETF 最新换手率 > 1.0%。
    
    ### 2. 卖出逻辑 (复合止损)
    * **战略单退出**：仅在宏观过热（广度 > 79% 且 Z-Score < 1.5）时触发。
    * **战术单退出**：
        * 触发宏观过热。
        * **或** 价格跌破 MA30 的同时，满足“今日下跌”或“5日不创新高”。
    
    ---
    **当前核心参数状态：**
    - 价格 vs MA30：{'上方(多头)' if last_row['close'] > last_row['MA30'] else '下方(空头)'}
    - 广度位置：{last_row['ma20_ratio']:.1f}%
    - 换手率：{last_row['Turnover_Pct']:.2f}%
    """)
