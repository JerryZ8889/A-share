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

# 设置绘图字体 (Streamlit云端通常自带支持，若显示乱码可改回默认)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载模块 (全部改为根目录读取)
# ==========================================

@st.cache_data(ttl=3600)
def load_index_data():
    """加载指数日线数据 (用于计算 Z-Score 和 趋势)"""
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    return df_idx

@st.cache_data(ttl=0)
def load_scan_results():
    """加载 A策略 市场广度结果 (根目录直读)"""
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
    """加载 B策略 首阴战法数据 (根目录直读)"""
    # 直接读取根目录下的文件
    file_name = 'CSI500_Master_Strategy.csv'
    if not os.path.exists(file_name):
        st.error(f"❌ 找不到文件 {file_name}，请检查是否已上传到 GitHub 根目录")
        st.stop()
    df = pd.read_csv(file_name, index_col='date', parse_dates=True)
    return df.sort_index()

# 执行加载过程
try:
    df_idx = load_index_data()
    history_df = load_scan_results()
    df_b = load_master_data()
    
    # 提取 A策略 最新数据
    last_row_a = history_df.iloc[-1]
    curr_ma20 = last_row_a['ma20_ratio']
    curr_nh = last_row_a['new_high_ratio']
    scan_date = history_df.index[-1].strftime('%Y-%m-%d')
    st.success(f"✅ 数据全量同步成功！ 数据日期：{scan_date}")
except Exception as e:
    st.error(f"⚠️ 核心数据载入失败: {e}")
    st.stop()

# ==========================================
# 2. 逻辑计算中心
# ==========================================

# --- [A策略计算：广度与热度] ---
vol = df_idx['volume']
z_series = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
curr_z = z_series.iloc[-1]
# 判断牛熊环境
idx_close = df_idx['close']
ma20_idx = idx_close.rolling(20).mean().iloc[-1]
ma60_idx = idx_close.rolling(60).mean().iloc[-1]
is_bull = ma20_idx > ma60_idx

# --- [B策略计算：首阴战法实时判定] ---
df_b['MA5'] = df_b['close'].rolling(5).mean()
df_b['MA10'] = df_b['close'].rolling(10).mean()
df_b['Is_Up'] = (df_b['close'] > df_b['close'].shift(1)).astype(int)
df_b['Streak'] = df_b['Is_Up'].groupby((df_b['Is_Up'] != df_b['Is_Up'].shift()).cumsum()).cumcount() + 1
df_b['Consec_Gains'] = np.where(df_b['Is_Up'] == 1, df_b['Streak'], 0)

last_b = df_b.iloc[-1]
prev_b = df_b.iloc[-2]

# B-买入/加仓 判定
b_cond1 = last_b['close'] > last_b['MA10']        # 1. 趋势线之上
b_cond2 = prev_b['Consec_Gains'] >= 2             # 2. 之前有连阳
b_cond3 = last_b['close'] < prev_b['close']       # 3. 今日首阴
# 换手率单位处理
t_val = last_b['ETF_Turnover'] if last_b['ETF_Turnover'] > 1 else last_b['ETF_Turnover'] * 100
b_cond4 = t_val > 1.5                             # 4. 放量 > 1.5%
b_cond5 = last_b['close'] > last_b['MA5']         # 5. 支撑位之上

b_add_signal = b_cond1 and b_cond2 and b_cond3 and b_cond4 and b_cond5

# B-卖出/平仓 判定 (规则6：连跌3天)
recent_rets = df_b['close'].pct_change().tail(3)
b_sell_signal = (recent_rets < 0).all()

# ==========================================
# 3. 结果看板展示
# ==========================================
st.divider()
st.subheader("🛡️ 动态决策综合报告")

# 3.1 核心指标矩阵
m1, m2, m3, m4 = st.columns(4)
m1.metric("市场模式", "📈 多头 (Bull)" if is_bull else "📉 空头 (Bear)")
m2.metric("资金热度 (Z)", f"{curr_z:.2f}")
m3.metric("市场宽度 (MA20%)", f"{curr_ma20:.1f}%")
m4.metric("500ETF 换手率", f"{t_val:.2f}%")

# 3.2 分项策略详情
st.write("---")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 🟢 策略A：宽度/热度择时")
    buy_a = curr_ma20 < 16
    if is_bull:
        sell_a = (curr_ma20 > 79) and (curr_z < 1.5) and (curr_nh < 10)
        s_reason = "宽度过热且动能耗尽"
    else:
        sell_a = (curr_ma20 > 40) and (curr_z < 1.0) and (curr_nh < 25)
        s_reason = "熊市反抽遇阻"

    if buy_a: st.success("🎯 **操作建议：买入/补仓** (冰点放量)")
    elif sell_a: st.error(f"🚨 **操作建议：减仓/清仓** ({s_reason})")
    else: st.info("⌛ **当前状态：持仓观望** (未达临界点)")

with col_b:
    st.markdown("#### 🔴 策略B：首阴回踩战法")
    if b_add_signal:
        st.success("🔥 **操作建议：【加仓】**")
        st.caption("理由：满足10日趋势向上、连阳后首阴回踩、且守住5日支撑。")
    elif b_sell_signal:
        st.error("🚨 **操作建议：【平仓】**")
        st.caption("理由：指数重心连续3日下移，短期趋势走坏。")
    else:
        st.info("⌛ **当前状态：无需操作**")

# 3.3 最终操作综合结论
st.divider()
st.subheader("💡 最终操作建议")
if buy_a and b_add_signal:
    st.warning("🚀 **综合结论：重仓共振！** 大盘处于冰点且中证500触发强力首阴回踩，胜率极高。")
elif b_add_signal:
    st.info("🔎 **综合结论：局部加仓。** 总体宽度一般，但500指数提供了高性价比的回踩加仓点。")
elif sell_a or b_sell_signal:
    reason = "A策略风险预警" if sell_a else "B策略趋势保护"
    st.error(f"⚠️ **综合结论：防御减仓。** 满足【{reason}】，建议收缩头寸。")
else:
    st.write("✅ **综合结论：保持现状。** 市场处于平稳博弈区，按既定比例持仓。")

# --- 4. 底部图表区 ---
with st.expander("查看实时趋势图表"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # 宽度趋势图
    ax1.plot(history_df.index, history_df['ma20_ratio'], label='MA20 %')
    ax1.set_title("市场宽度趋势")
    # 热度趋势图
    p_data = z_series.tail(100)
    ax2.fill_between(p_data.index, p_data, 0, alpha=0.3, color='red')
    ax2.set_title("资金热度 (Z-Score)")
    plt.tight_layout()
    st.pyplot(fig)
