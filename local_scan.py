import akshare as ak
import pandas as pd
from tqdm import tqdm
import time
import os
from datetime import datetime

def run_local_scan():
    print(f"🚀 启动每日深度扫描 | 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 获取中证500成份股清单
    try:
        index_stock_df = ak.index_stock_cons(symbol="000905")
        stock_list = index_stock_df['品种代码'].tolist()
    except Exception as e:
        print(f"❌ 获取清单失败: {e}")
        return

    results = []
    
    # --- 核心设置：往前多取点数据，确保能算出 60日新高 ---
    start_search_date = "20250601" 

    # 2. 开始扫描 500 只股票
    for i, code in enumerate(tqdm(stock_list, desc="扫描中证500成份股")):
        try:
            # 抓取历史数据
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_search_date, adjust="qfq")
            
            # 数据健壮性检查：至少要有 20 天才能算 MA20
            if df is not None and len(df) >= 20:
                latest_close = df['收盘'].iloc[-1]
                
                # 指标 1: 站上 MA20 (短线趋势)
                ma20 = df['收盘'].rolling(20).mean().iloc[-1]
                is_above_ma20 = 1 if latest_close > ma20 else 0
                
                # 指标 2: 创 60日新高 (长线动能)
                # 如果新上市不满 60 天，则取当前所有交易日的最高
                window_size = min(len(df), 60)
                high_60 = df['最高'].tail(window_size).max()
                is_new_high = 1 if latest_close >= high_60 else 0
                
                results.append({
                    'ma20_ok': is_above_ma20, 
                    'new_high_ok': is_new_high
                })
            
            # --- 频率保护：每抓 100 只歇 2 秒，防止被新浪封 IP ---
            if i % 100 == 0 and i > 0:
                time.sleep(2)
            else:
                time.sleep(0.05)
                
        except Exception:
            # 个别股票报错则跳过，保证大盘数据能算出来
            continue
    
    if not results:
        print("❌ 扫描失败：未收集到有效数据，请检查网络。")
        return

    # 3. 计算今日百分比
    res_df = pd.DataFrame(results)
    today_str = datetime.now().strftime('%Y-%m-%d')
    new_data = {
        'date': [today_str],
        'ma20_ratio': [round(res_df['ma20_ok'].mean() * 100, 2)],
        'new_high_ratio': [round(res_df['new_high_ok'].mean() * 100, 2)]
    }
    new_df = pd.DataFrame(new_data)

    # --- 核心逻辑：智能追加与覆盖 ---
    file_name = "scan_results.csv"
    
    if os.path.exists(file_name):
        # 1. 读取旧数据
        old_df = pd.read_csv(file_name)
        # 2. 合并新旧数据
        # 使用 drop_duplicates 时，如果日期相同，keep='last' 会让下午的数据覆盖中午的
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
        # 3. 按日期排序确保图表正确
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values('date')
        combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d')
    else:
        # 如果文件不存在，则直接使用当前数据
        combined_df = new_df

    # 4. 保存结果
    combined_df.to_csv(file_name, index=False)
    
    print("\n" + "="*30)
    print(f"✅ 扫描任务圆满完成！")
    print(f"📊 今日({today_str})占比结果：")
    print(f"   - 站上 MA20 比例: {new_data['ma20_ratio'][0]}%")
    print(f"   - 创 60日新高比例: {new_data['new_high_ratio'][0]}%")
    print(f"📁 结果已存入 {file_name}，当前数据库共积累 {len(combined_df)} 天数据。")
    print("="*30)

if __name__ == "__main__":
    run_local_scan()