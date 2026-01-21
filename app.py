import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from datetime import datetime

# ==========================================
# 0. 页面配置与字体修复 (暴力适配版)
# ==========================================
st.set_page_config(page_title="量化大师-旗舰进化版", layout="wide")
st.title("🛡️ 量化大师：MA30过滤旗舰进化版综合看板")

def set_chinese_font():
    """
    设置 Matplotlib 中文字体。
    尝试使用多个常见的中文字体族，以适应不同的操作系统环境（Windows/Linux/Mac）。
    """
    # 字体优先级列表：优先尝试 SimHei, 然后是微软雅黑, 苹果字体, Linux字体, 最后是通用无衬线字体
    font_list = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif']
    
    # 将这些字体加入到 Matplotlib 的首选字体列表中
    # Matplotlib 会自动尝试列表中的字体，直到找到系统安装了的那一个
    plt.rcParams['font.sans-serif'] = font_list + plt.rcParams['font.sans-serif']
    
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

# 执行字体设置
set_chinese_font()

# ==========================================
# 1. 数据加载逻辑
# ==========================================
@st.cache_data(ttl=60)
def load_all_data():
    # 1. 指数数据
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    
    # 2. 广度/主策略/汇总数据
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
# 2. 旗舰进化逻辑计算引擎
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
    
    # 计算热度 Z-Score并填充nan
    vol = df['volume']
    df['Heat_Z'] = ((vol - vol.rolling(60).mean()) / vol.rolling(60).std()).ffill().fillna(0)
    
    # 计算连阳特征
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Consec_Gains'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Consec_Gains'], 0)
    
    # 换手率格式统一
    df['Turnover_Pct'] = np.where(df['ETF_Turnover'] > 1, df['ETF_Turnover'], df['ETF_Turnover'] * 100)
    
    # 信号循环生成
    df['signal'] = 0; df['logic_type'] = ""
    in_pos, logic_state, entry_high, hold_days = False, "", 0, 0

    for i in range(1, len(df)):
        curr, prev = df.iloc[i], df.iloc[i-1]
        
        # 卖出逻辑
        if in_pos:
            hold_days += 1
            # 宏观过热条件
            is_macro_exit = (curr['ma20_ratio'] > 79) and (curr['Heat_Z'] < 1.5)
            exit_flag = False
            
            if logic_state == "Strategic":
                if is_macro_exit: exit_flag = True
            else: # Tactical 战术单复合止损
                is_trend_broken = curr['close'] < curr['MA30']
                # 破位后只要收阴或时间失效就走
                if is_macro_exit or (is_trend_broken and (curr['close'] < prev['close'] or (hold_days >= 5 and curr['close'] < entry_high))):
                    exit_flag = True
            
            if exit_flag:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                in_pos, logic_state = False, ""
        
        # 买入逻辑
        else:
            # 战略买入
            if curr['ma20_ratio'] < 16:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Strategic"
                in_pos, logic_state, hold_days = True, "Strategic", 0
            # 战术买入
            elif (curr['close'] > curr['MA30'] and  # 趋势过滤
                  curr['close'] > curr['MA10'] and  # 短期支撑
                  curr['close'] > curr['MA5'] and   # 攻击形态
                  prev['Consec_Gains'] >= 3 and     # 此前连阳
                  curr['close'] < prev['close'] and # 今日首阴
                  curr['Turnover_Pct'] > 1.0):      # 量能活跃
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('logic_type')] = "Tactical"
                in_pos, logic_state, hold_days, entry_high = True, "Tactical", 0, curr['high']
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
# 4. 结论与K线标注
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
# 使用英文标签以防万一字体仍有问题
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.5, label='Close Price')
ax3.plot(df_plot.index, df_plot['MA30'], color='blue', linestyle='--', label='MA30 Trend')
b_pts = df_plot[df_plot['signal'] == 1]
s_pts = df_plot[df_plot['signal'] == -1]
ax3.scatter(b_pts.index, b_pts['close'], color='red', marker='^', s=120, zorder=5, label='Buy Signal')
ax3.scatter(s_pts.index, s_pts['close'], color='green', marker='v', s=120, zorder=5, label='Sell Signal')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.2)
st.pyplot(fig3)

# ==========================================
# 5. 决策逻辑详情 (详细版)
# ==========================================
with st.expander("查看【MA30过滤版 旗舰进化】决策逻辑判定详情", expanded=True):
    st.markdown("""
    ### ⚔️ 核心策略体系详解

    本策略采用**“战略 (Strategic) + 战术 (Tactical)”**双轨制驱动，旨在结合宏观择时的高胜率与微观形态的高爆发力。

    ---

    #### ✅ 一、买入逻辑 (Entry Rules)

    **1. 战略买入 (Strategic Entry)**
    * **核心理念**：人弃我取，博弈市场极度恐慌后的宏观修复。
    * **触发条件**：
        * **市场广度 (MA20 Ratio) < 16%**：全市场只有不到 16% 的股票在 20 日均线上方，代表市场进入冰点超跌区。
    
    **2. 战术买入 (Tactical Entry - 旗舰进化版)**
    * **核心理念**：在明确的上升趋势中，捕捉主力洗盘后的“首阴”回踩机会。
    * **触发条件（必须全部满足）**：
        * **【趋势过滤】价格 > MA30**：确保大方向向上，不做空头反弹。
        * **【支撑确认】价格 > MA10 且 价格 > MA5**：确保短期强势结构未被破坏。
        * **【形态特征】此前连阳 ≥ 3天，且 今日收阴**：确认是强势上涨后的首次回调。
        * **【量能门槛】ETF换手率 > 1.0%**：确保市场活跃度足够支撑反弹。

    ---

    #### 🛑 二、卖出逻辑 (Exit Rules - 复合止损)

    卖出采用“宏观过热”与“趋势破位”双重保险。

    **1. 宏观过热退出 (Macro Overheat Exit)**
    * **适用对象**：战略单 & 战术单
    * **触发条件**：
        * **广度 > 79% 且 资金热度 (Z-Score) < 1.5**：市场情绪极度亢奋但增量资金开始跟不上，预示顶部临近。

    **2. 趋势破位退出 (Trend Breakdown Exit)**
    * **适用对象**：仅战术单
    * **触发条件（满足其一即可）**：
        * **条件 A (破位止损)**：价格跌破 **MA30**，且今日收阴线。
        * **条件 B (时间止损)**：价格跌破 **MA30**，且持仓超过 5 天仍未创出买入后的新高（证明上涨动力消失）。
    """)
