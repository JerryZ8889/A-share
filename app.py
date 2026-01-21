import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import akshare as ak

# ==========================================
# 0. 环境与字体修复
# ==========================================
st.set_page_config(page_title="量化大师-旗舰进化版", layout="wide")
st.title("🛡️ 量化大师：MA30过滤旗舰进化版综合看板")

def set_matplotlib_font():
    fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False
set_matplotlib_font()

# ==========================================
# 1. 数据加载逻辑
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
    st.error(f"⚠️ 加载失败: {e}")
    st.stop()

# ==========================================
# 2. 旗舰进化计算引擎
# ==========================================
def calculate_signals(df_price, df_breadth):
    df = df_price.copy()
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA30'] = df['close'].rolling(30).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    df = df.join(df_breadth[['ma20_ratio', 'new_high_ratio']], how='left').ffill()
    
    # 修复 nan Z-Score 问题
    vol = df['volume']
    df['Heat_Z'] = ((vol - vol.rolling(60).mean()) / vol.rolling(60).std()).ffill().fillna(0)
    
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Consec_Gains'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Consec_Gains'], 0)
    
    df['Turnover_Pct'] = np.where(df['ETF_Turnover'] > 1, df['ETF_Turnover'], df['ETF_Turnover'] * 100)
    
    df['signal'], df['logic_type'] = 0, ""
    in_pos, logic_state, entry_high, hold_days = False, "", 0, 0

    for i in range(1, len(df)):
        curr, prev = df.iloc[i], df.iloc[i-1]
        if in_pos:
            hold_days += 1
            is_overheat = (curr['ma20_ratio'] > 79) and (curr['Heat_Z'] < 1.5)
            exit_flag = False
            if logic_state == "Strategic":
                if is_overheat: exit_flag = True
            else:
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

df_final = calculate_signals(df_main, df_scan)
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
# 4. K线标注与建议
# ==========================================
st.divider()
st.subheader("💡 最终操作建议与走势标注")
if last_row['signal'] == 1: st.success(f"🚀 **操作建议：买入 ({last_row['logic_type']})**")
elif last_row['signal'] == -1: st.error("🚨 **操作建议：清仓/减仓**")
else: st.info("✅ **操作建议：持股/观望**")

df_plot = df_final.loc["2024-01-01":]
fig3, ax3 = plt.subplots(figsize=(16, 8))
ax3.plot(df_plot.index, df_plot['close'], color='gray', alpha=0.5, label='收盘价')
ax3.plot(df_plot.index, df_plot['MA30'], color='blue', linestyle='--', label='MA30')
b_pts = df_plot[df_plot['signal'] == 1]
s_pts = df_plot[df_plot['signal'] == -1]
ax3.scatter(b_pts.index, b_pts['close'], color='red', marker='^', s=120, label='买入')
ax3.scatter(s_pts.index, s_pts['close'], color='green', marker='v', s=120, label='卖出')
ax3.legend()
st.pyplot(fig3)

with st.expander("查看决策逻辑判定详情"):
    st.write("战略：广度<16%。战术：MA30线上+首阴+换手>1%。退出：过热或趋势走弱。")
