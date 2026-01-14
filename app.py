import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(page_title="量化大师-专业版", layout="wide")
st.title("🛡️ 量化大师：全量扫描结果看板 (C方案)")

# --- 1. 基础数据加载 (指数走势：每小时更新一次) ---
@st.cache_data(ttl=3600)
def load_index_data():
    df_idx = ak.stock_zh_index_daily(symbol="sh000905")
    df_idx['date'] = pd.to_datetime(df_idx['date'])
    df_idx.set_index('date', inplace=True)
    return df_idx

df_idx = load_index_data()

# --- 2. 读取累积的扫描结果 (核心加固区：强制实时同步) ---
@st.cache_data(ttl=0)  # 🚩 关键：设置缓存为 0，确保每次刷新都读最新的 GitHub 文件
def load_scan_results():
    # 读取你手动补全或自动生成的 CSV
    df = pd.read_csv("scan_results.csv")
    
    # 你的核心清洗逻辑：强制转日期并扔掉空行
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df.set_index('date', inplace=True)
    return df

try:
    # 调用加固后的函数
    history_df = load_scan_results()
    
    # 获取最新数据用于展示
    last_row = history_df.iloc[-1]
    curr_ma20 = last_row['ma20_ratio']
    curr_nh = last_row['new_high_ratio']
    scan_date = history_df.index[-1].strftime('%Y-%m-%d')
    
    # 尝试读取时间，如果没有这个列就显示为空
    update_time = f" | 更新时间：{last_row['update_time']}" if 'update_time' in last_row else ""
    
    # --- 顶部的成功提示框 (确保对齐) ---
    st.success(f"✅ 深度扫描数据同步成功！ 数据日期：{scan_date}{update_time}")
    
except Exception as e:
    st.error(f"⚠️ 数据同步中或格式有误。 详情: {e}")
    st.stop()

# --- 3. 布局：左右双图 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 资金热度 (Z-Score)")
    vol = df_idx['volume']
    z = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    p_data = z.tail(100)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data>=0), color='red', alpha=0.3)
    ax1.fill_between(p_data.index, p_data, 0, where=(p_data<0), color='blue', alpha=0.3)
    ax1.axhline(y=1.5, color='orange', linestyle='--')
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader("📊 市场广度 (全量历史趋势)")
    fig2, ax_l = plt.subplots(figsize=(10, 5))
    # 绘制站上 MA20 比例
    ax_l.plot(history_df.index, history_df['ma20_ratio'], color='tab:blue', marker='o', linewidth=2, label='MA20 %')
    ax_l.set_ylim(0, 100)
    ax_l.set_ylabel('Above MA20 (%)', color='tab:blue')
    # 绘制新高比例
    ax_r = ax_l.twinx()
    ax_r.bar(history_df.index, history_df['new_high_ratio'], color='tab:orange', alpha=0.4)
    ax_r.set_ylabel('New High (%)', color='tab:orange')
    plt.xticks(rotation=45)
    fig2.tight_layout()
    st.pyplot(fig2)

# --- 4. 核心逻辑计算：动态模式识别 ---
st.divider()

# 4.1 准备计算环境
# 获取指数均线，用于判断当前是“多头环境”还是“空头环境”
idx_close = df_idx['close']
ma20_idx = idx_close.rolling(20).mean().iloc[-1]
ma60_idx = idx_close.rolling(60).mean().iloc[-1]
is_bull = ma20_idx > ma60_idx

# 获取当前各项因子数值
# curr_ma20 和 curr_nh 在前面 load_scan_results 已经提取了
curr_z = z.iloc[-1]  # 当前资金热度 Z-Score

# 4.2 逻辑判定
# 【买入逻辑】：冰点抄底 (不分牛熊)
buy_signal = curr_ma20 < 16

# 【卖出逻辑】：动态双轨制
if is_bull:
    # 多头环境：宽容持仓。满足：过热(>79) & 缩量(<1.5) & 新高减少(<10%) 才建议卖
    mode_text = "📈 当前为：多头趋势环境 (MA20 > MA60)"
    sell_signal = (curr_ma20 > 79) and (curr_z < 1.5) and (curr_nh < 10)
    sell_reason = "宽度过热且创新高动能枯竭"
else:
    # 空头环境：严苛防御。满足：宽度回升(>40) & 缩量(<1.0) & 新高不足(<25%) 就要卖
    mode_text = "📉 当前为：空头趋势环境 (MA20 < MA60)"
    sell_signal = (curr_ma20 > 40) and (curr_z < 1.0) and (curr_nh < 25)
    sell_reason = "熊市反抽遇阻，动能不足以支撑继续上涨"

# --- 5. 结果看板展示 ---
st.subheader("🛡️ 动态逻辑诊断报告")
c1, c2, c3, c4 = st.columns(4)
c1.metric("市场模式", "多头" if is_bull else "空头")
c2.metric("资金热度 (Z)", f"{curr_z:.2f}")
c3.metric("市场宽度 (MA20%)", f"{curr_ma20:.1f}%")
c4.metric("新高比例 (NH%)", f"{curr_nh:.1f}%")

st.info(f"**模式分析**：{mode_text}")

# 最终结论输出
if buy_signal:
    st.success("🎯 **操作建议：【买入/补仓】** —— 市场进入冰点区域，胜率极高。")
elif sell_signal:
    st.error(f"🚨 **操作建议：【卖出/清仓】** —— 满足{sell_reason}，风险集聚。")
else:
    if is_bull:
        st.warning("💎 **操作建议：【持股待涨】** —— 虽然有所波动或过热，但新高保护/趋势仍在，建议让利润奔跑。")
    else:
        st.info("⌛ **操作建议：【空仓观望】** —— 趋势未反转且未达冰点，耐心等待机会。")

# 逻辑详情参考 (展开查看)
with st.expander("查看当前决策逻辑详情"):
    st.write(f"""
    - **买入标准**：宽度 < 16% (当前: {curr_ma20:.1f}%)
    - **卖出标准 ({'多头' if is_bull else '空头'}模式)**：
        - 宽度门槛: {'> 79%' if is_bull else '> 40%'} (当前: {curr_ma20:.1f}%)
        - 热度门槛: {'< 1.5' if is_bull else '< 1.0'} (当前: {curr_z:.2f})
        - 新高保护: {'< 10%' if is_bull else '< 25%'} (当前: {curr_nh:.1f}%)
    """)
