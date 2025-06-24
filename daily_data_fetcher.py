# -*- encoding: UTF-8 -*-

import akshare as ak
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import inspect
import os
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import tushare as ts

# Import helpers from data_fetcher, which now holds all tooling and helper functions
from data_fetcher import (
    load_config,
    process_stock,
    write_one_stock_daily,
)

# --- Data Fetcher Registry ---
class _DataFetcherRegistry:
    def __init__(self):
        self.update_tasks = []

    def register(self, func):
        """Decorator to register a function as a data update task."""
        logging.debug(f"Registering data update task: {func.__name__}")
        self.update_tasks.append(func)
        return func

registry = _DataFetcherRegistry()

def run_all_updates():
    """
    Runs all registered data update tasks in sequence.
    Fetches the master stock list once and passes it to tasks that need it.
    """
    print(f"[{datetime.now()}] 开始执行所有数据更新任务...")
    logging.info("Starting all registered data update tasks...")
    
    print(f"[{datetime.now()}] 正在获取股票列表和实时数据...")
    logging.info("Fetching master stock list and spot data...")
    try:
        spot_df = ak.stock_zh_a_spot_em()
        if spot_df is None or spot_df.empty:
            logging.error("Failed to get spot data, aborting all updates.")
            print(f"[{datetime.now()}] 获取实时数据失败，终止所有更新任务")
            return
        stock_codes = spot_df[~spot_df['收盘'].isna()]['代码'].tolist()
        print(f"[{datetime.now()}] 成功获取 {len(stock_codes)} 只股票的实时数据")
        logging.info(f"Successfully fetched spot data for {len(stock_codes)} stocks.")
    except Exception as e:
        logging.error(f"Failed to fetch master stock list: {e}. Aborting all updates.", exc_info=True)
        print(f"[{datetime.now()}] 获取股票列表失败: {e}，终止所有更新任务")
        return

    for task_func in registry.update_tasks:
        task_name = task_func.__name__
        print(f"[{datetime.now()}] 开始执行任务: {task_name}")
        logging.info(f"--- Running update task: {task_name} ---")
        sig = inspect.signature(task_func)
        params = sig.parameters
        
        try:
            if 'spot_df' in params:
                task_func(spot_df=spot_df)
            elif 'stock_codes' in params:
                task_func(stock_codes=stock_codes)
            else:
                task_func()
            print(f"[{datetime.now()}] 任务完成: {task_name}")
            logging.info(f"--- Finished update task: {task_name} ---")
        except Exception as e:
            print(f"[{datetime.now()}] 任务执行失败: {task_name}, 错误: {e}")
            logging.error(f"--- Error executing task '{task_name}': {e} ---", exc_info=True)
    
    print(f"[{datetime.now()}] 所有数据更新任务执行完毕！")


@registry.register
def update_minute_data(stock_codes):
    """更新传入股票列表的分钟数据"""
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    minute_dir = os.path.join(data_dir, 'minute')
    
    if not os.path.exists(minute_dir):
        os.makedirs(minute_dir)
    
    try:
        # 使用线程池并行处理，减少并发数
        results = {}
        failed_stocks = []
        
        # 创建进度条
        pbar = tqdm(total=len(stock_codes), desc="更新分钟数据", unit="只")
        
        with ThreadPoolExecutor(max_workers=3) as executor:  # 减少并发数
            future_to_stock = {executor.submit(process_stock, code, minute_dir): code for code in stock_codes}
            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[stock] = result
                    else:
                        failed_stocks.append(stock)
                except Exception as exc:
                    logging.error(f'[分钟] 股票 {stock} 处理失败: {exc}')
                    failed_stocks.append(stock)
                finally:
                    pbar.update(1)
                    # 更新进度条描述
                    pbar.set_postfix({
                        '成功': len(results),
                        '失败': len(failed_stocks),
                        '总记录': sum(r['total'] for r in results.values()) if results else 0,
                        '新增': sum(r['new'] for r in results.values()) if results else 0
                    })
        
        pbar.close()
        
        # 打印汇总信息
        total_stocks = len(results)
        total_records = sum(r['total'] for r in results.values())
        total_new = sum(r['new'] for r in results.values())
        logging.info(f"[分钟] 更新完成，成功处理 {total_stocks} 只股票，总记录数: {total_records}，新增记录数: {total_new}")
        if failed_stocks:
            logging.warning(f"[分钟] 以下股票处理失败: {', '.join(failed_stocks)}")
            
    except Exception as e:
        logging.error(f"[分钟] 更新数据失败: {str(e)}")


@registry.register
def update_daily_data_snapshot(spot_df):
    """高效批量更新当日日线数据"""
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    daily_dir = os.path.join(data_dir, 'daily')
    if not os.path.exists(daily_dir):
        os.makedirs(daily_dir)

    if spot_df is None or spot_df.empty:
        logging.error("[日线快照] 传入的A股快照为空")
        return

    # 2. 字段映射，整理为daily parquet格式
    today = datetime.now().strftime('%Y-%m-%d')
    daily_cols = ['日期','股票代码','开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌幅','涨跌额','换手率','代码']
    rename_map = {
        '代码': '代码', '名称': '名称', '今开': '开盘', '最新价': '收盘', '最高': '最高',
        '最低': '最低', '成交量': '成交量', '成交额': '成交额', '振幅': '振幅',
        '涨跌幅': '涨跌幅', '涨跌额': '涨跌额', '换手率': '换手率',
    }
    daily_df = pd.DataFrame()
    for src, tgt in rename_map.items():
        if src in spot_df.columns:
            daily_df[tgt] = spot_df[src]
        else:
            daily_df[tgt] = None
    daily_df['日期'] = today
    daily_df['股票代码'] = daily_df['代码']
    # 调整列顺序
    daily_df = daily_df[daily_cols]
    # 类型转换
    daily_df['日期'] = pd.to_datetime(daily_df['日期'])

    # 3. 并行写入/追加到各自股票的parquet
    max_workers = config.get('daily_snapshot_workers', 16)
    delisted_codes = []
    failed_codes = []
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(write_one_stock_daily, row, daily_dir) for idx, row in daily_df.iterrows()]
        
        pbar = tqdm(as_completed(futures), total=len(futures), desc="[日线快照] 批量更新")
        for f in pbar:
            status, code = f.result()
            if status == 'success':
                success_count += 1
            elif status == 'delisted':
                delisted_codes.append(code)
            elif status == 'failed':
                failed_codes.append(code)
            pbar.set_postfix(success=success_count, delisted=len(delisted_codes), failed=len(failed_codes))

    logging.info(f"[日线快照] 批量更新完成，成功写入 {success_count} 只股票，退市/无最新价 {len(delisted_codes)} 只，写入失败 {len(failed_codes)} 只")
    if delisted_codes:
        logging.warning(f"[日线快照] 跳过退市/无最新价股票: {delisted_codes}")
    if failed_codes:
        logging.error(f"[日线快照] 写入失败股票: {failed_codes}")


@registry.register
def update_hsgt_top10_data():
    """
    增量更新沪深股通十大成交股数据.
    - 如果文件存在, 从最新日期开始更新.
    - 如果文件不存在, 获取最近30天的数据作为初始数据.
    """
    config = load_config()
    token = config.get('tushare_token')
    if not token or 'your_tushare_pro_token' in token:
        logging.warning("[HSGT Update] Tushare token not configured in config.yaml, skipping.")
        return

    try:
        pro = ts.pro_api(token)
    except Exception as e:
        logging.error(f"[HSGT Update] Failed to initialize Tushare API: {e}")
        return
    
    data_dir = config.get('data_dir', 'E:/data')
    other_dir = os.path.join(data_dir, 'other')
    os.makedirs(other_dir, exist_ok=True)
    file_path = os.path.join(other_dir, 'hsgt_top10.parquet')

    start_date = None
    existing_df = pd.DataFrame()
    n_existing = 0

    if os.path.exists(file_path):
        try:
            existing_df = pd.read_parquet(file_path)
            n_existing = len(existing_df)
            if not existing_df.empty:
                latest_date_str = existing_df['trade_date'].max()
                latest_date = pd.to_datetime(latest_date_str, format='%Y%m%d')
                start_date = (latest_date).strftime('%Y%m%d')
                logging.info(f"[HSGT Update] 增量更新模式，从 {start_date} 开始获取.")
            else: # file exists but is empty
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                logging.info(f"[HSGT Update] 数据文件为空，获取最近30天数据. 从 {start_date} 开始.")
        except Exception as e:
            logging.error(f"[HSGT Update] 读取数据文件失败: {e}. 将尝试获取最近30天数据.")
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    else: # file does not exist
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        logging.info(f"[HSGT Update] 数据文件不存在，将获取最近30天数据. 建议先运行 `initialize_hsgt_top10_data`.")
        
    end_date = datetime.now().strftime('%Y%m%d')
    
    if start_date > end_date:
        logging.info("[HSGT Update] 数据已是最新.")
        return

    # Fetch new data
    logging.info(f"[HSGT Update] 开始获取 {start_date} 到 {end_date} 的数据...")
    new_df = pd.DataFrame()
    try:
        new_df = pro.hsgt_top10(start_date=start_date, end_date=end_date)
    except Exception as e:
        logging.error(f"[HSGT Update] 获取 {start_date}-{end_date} 数据失败: {e}")
        return

    if new_df is None or new_df.empty:
        logging.info("[HSGT Update] 在指定时间段内未获取到新的十大成交股数据.")
        return
        
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset=['trade_date', 'ts_code', 'market_type'], keep='last', inplace=True)
    combined_df.sort_values(by=['trade_date', 'market_type', 'rank'], inplace=True)
    
    n_added = len(combined_df) - n_existing

    try:
        combined_df.to_parquet(file_path, index=False)
        logging.info(f"[HSGT Update] 数据更新成功. 文件路径: {file_path}, 总行数: {len(combined_df)}, 新增: {n_added} 行.")
    except Exception as e:
        logging.error(f"[HSGT Update] 保存数据到 {file_path} 失败: {e}")

