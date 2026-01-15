import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 页面配置与基础环境
# ==========================================
st.set_page_config(page_title="量化大师-策略融合版", layout="wide")
st.title("🛡️ 量化大师：全量扫描与首阴战法综合看板")

# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心数据加载模块
# ==========================================

@st.cache_data(ttl=3600)
def load_index_data():
    """1. 加载指数日线数据 ( sh000905 )"""
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    return df_idx

@st.cache_data(ttl=0) # 强制实时同步
def load_scan_results():
    """2. 加载 A策略 市场广度结果 (scan_results.csv)"""
    file_name = "scan_results.csv"
    if not os.path.exists(file_name):
        st.error(f"❌ 未找到 {file_name}")
        st.stop()
    df = pd.read_csv(file_name)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df.set_index('date', inplace=True)
    return df

@st.cache_data(ttl=3600)
def load_master_data():
    """3. 加载 B策略 首阴战法数据 (CSI500_Master_Strategy.csv)"""
    file_name = 'CSI500_Master_Strategy.csv'
    if not os.path.exists(file_name):
        st.error(f"❌ 找不到文件 {file_name}")
        st.stop()
    df = pd.read_csv(file_name, index_col='date', parse_dates=True)
    return df.sort_index()

# 执行加载
try:
    df_idx = load_index_data()
    history_df = load_scan_results()
    df_b = load_master_data()
    
    # 获取最新数据用于顶部看板
    last_row_a = history_df.iloc[-1]
    curr_ma20 = last_row_a['ma20_ratio']
    curr_nh = last_row_a['new_high_ratio']
    scan_date = history_df.index[-1].strftime('%Y-%m-%d')
    update_time = f" | 扫描时间：{last_row_a['update_time']}" if 'update_time' in last_row_a else ""
    
    # 顶部成功提示框 (显示精确时间)
    st.success(f"✅ 数据同步成功！ 数据日期：{scan_date}{update_time}")
except Exception as e:
    st.error(f"⚠️ 数据同步失败: {e}")
    st.stop()

# ==========================================
# 2. 布局：左右双图 (维持原 A代码 布局)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 资金热度 (Z-Score)")
    vol = df_idx['volume']
    z_series = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    curr_z = z_series.iloc[-1]
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_data = z_series.tail(100)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data>=0), color='red', alpha=0.3)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader("📊 市场广度 (全量历史趋势)")
    fig2, ax_l = plt.subplots(figsize=(10, 5))
    # 绘制站上 MA20 比例 (左轴)
    ax_l.plot(history_df.index, history_df['ma20_ratio'], color='tab:blue', marker='o', linewidth=2, label='MA20 %')
    ax_l.set_ylim(0, 100)
    ax_l.set_ylabel('Above MA20 (%)', color='tab:blue')
    # 绘制新高比例 (右轴)
    ax_r = ax_l.twinx()
    ax_r.bar(history_df.index, history_df['new_high_ratio'], color='tab:orange', alpha=0.4)
    ax_r.set_ylabel('New High (%)', color='tab:orange')
    plt.xticks(rotation=45)
    fig2.tight_layout()
    st.pyplot(fig2)

# ==========================================
# 3. 核心计算中心 (融合逻辑)
# ==========================================

# 3.1 A策略逻辑环境 (牛熊判定)
idx_close = df_idx['close']
ma20_idx = idx_close.rolling(20).mean().iloc[-1]
ma60_idx = idx_close.rolling(60).mean().iloc[-1]
is_bull = ma20_idx > ma60_idx

# 3.2 B策略逻辑计算 (首阴战法判定)
df_b['MA5'] = df_b['close'].rolling(5).mean()
df_b['MA10'] = df_b['close'].rolling(10).mean()
df_b['Is_Up'] = (df_b['close'] > df_b['close'].shift(1)).astype(int)
df_b['Streak'] = df_b['Is_Up'].groupby((df_b['Is_Up'] != df_b['Is_Up'].shift()).cumsum()).cumcount() + 1
df_b['Consec_Gains'] = np.where(df_b['Is_Up'] == 1, df_b['Streak'], 0)

last_b = df_b.iloc[-1]
prev_b = df_b.iloc[-2]

# B-买入/加仓 条件
b_cond1 = last_b['close'] > last_b['MA10']
b_cond2 = prev_b['Consec_Gains'] >= 2
b_cond3 = last_b['close'] < prev_b['close']
# 换手率判定
t_val = last_b['ETF_Turnover'] if last_b['ETF_Turnover'] > 1 else last_b['ETF_Turnover'] * 100
b_cond4 = t_val > 1.5
b_cond5 = last_b['close'] > last_b['MA5']

b_add_signal = b_cond1 and b_cond2 and b_cond3 and b_cond4 and b_cond5

# B-卖出/平仓 条件
recent_rets = df_b['close'].pct_change().tail(3)
b_sell_signal = (recent_rets < 0).all()

# ==========================================
# 4. 诊断报告看板 (融合诊断)
# ==========================================
st.divider()
st.subheader("🛡️ 动态逻辑诊断报告")

# 4.1 指标矩阵
c1, c2, c3, c4 = st.columns(4)
c1.metric("市场模式", "📈 多头 (Bull)" if is_bull else "📉 空头 (Bear)")
c2.metric("资金热度 (Z)", f"{curr_z:.2f}")
c3.metric("市场宽度 (MA20%)", f"{curr_ma20:.1f}%")
c4.metric("中证500换手", f"{t_val:.2f}%")

# 4.2 模式分析文本
mode_text = "📈 当前为：多头趋势环境 (MA20 > MA60)" if is_bull else "📉 当前为：空头趋势环境 (MA20 < MA60)"
st.info(f"**模式分析**：{mode_text}")

# 4.3 策略分项建议
st.write("---")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 🟢 策略A：宽度择时")
    buy_a = curr_ma20 < 16
    if is_bull:
        sell_a = (curr_ma20 > 79) and (curr_z < 1.5) and (curr_nh < 10)
        s_reason = "宽度过热且创新高动能枯竭"
    else:
        sell_a = (curr_ma20 > 40) and (curr_z < 1.0) and (curr_nh < 25)
        s_reason = "反抽遇阻"

    if buy_a: st.success("🎯 **A建议：【买入/补仓】** (冰点触发)")
    elif sell_a: st.error(f"🚨 **A建议：【卖出/清仓】** ({s_reason})")
    else: st.warning("💎 **A状态：持股待涨**") if is_bull else st.info("⌛ **A状态：空仓观望**")

with col_b:
    st.markdown("#### 🔴 策略B：首阴战法")
    if b_add_signal:
        st.success("🔥 **B建议：【加仓】** —— 满足首阴回踩逻辑")
    elif b_sell_signal:
        st.error("🚨 **B建议：【减仓】** —— 满足重心下移止损")
    else:
        st.info("⌛ **B状态：无需操作**")

# ==========================================
# 5. 最终结论输出 (综合结论)
# ==========================================
st.divider()
st.subheader("💡 最终操作建议")

if buy_a and b_add_signal:
    st.warning("🚀 **综合结论：重仓共振！** 大盘冰点与500指数首阴回踩同时出现，胜率极高。")
elif b_add_signal:
    st.info("🔎 **综合结论：局部加仓。** 虽然大盘宽度一般，但中证500提供了高性价比的回踩加仓点。")
elif sell_a or b_sell_signal:
    reason = "A策略风险预警" if sell_a else "B策略趋势走坏"
    st.error(f"🚨 **综合结论：防御减仓。** 满足【{reason}】，建议收缩头寸。")
else:
    st.write("✅ **综合结论：目前市场处于平稳期**。建议按原有比例持仓，等待信号。")

# 逻辑详情参考
with st.expander("查看决策逻辑判定详情"):
    st.write(f"""
    - **A策略买入标准**：宽度 < 16% (当前: {curr_ma20:.1f}%)
    - **A策略卖出标准 ({'多头' if is_bull else '空头'})**：宽度 > {'79%' if is_bull else '40%'}, Z < {'1.5' if is_bull else '1.0'}
    - **B策略买入逻辑**：10日线上 + 连阳后首阴 + 换手>1.5% + 5日线不破
    - **B策略止损逻辑**：价格重心连续 3 日下移
    """)
