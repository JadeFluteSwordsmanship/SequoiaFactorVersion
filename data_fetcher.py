# -*- encoding: UTF-8 -*-

import akshare as ak
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import time
import yaml
import concurrent.futures
from tqdm import tqdm
import tushare as ts
import talib as tl

from utils import setup_logging


def fetch(code_name):
    stock = code_name[0]
    data = ak.stock_zh_a_hist(symbol=stock, period="daily", start_date="20220101", adjust="qfq")

    if data is None or data.empty:
        logging.debug("股票："+stock+" 没有数据，略过...")
        return

    data['p_change'] = tl.ROC(data['收盘'], 1)

    return data


def run(stocks):
    stocks_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_stock = {executor.submit(fetch, stock): stock for stock in stocks}
        for future in concurrent.futures.as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                data = future.result()
                if data is not None:
                    data = data.astype({'成交量': 'double'})
                    stocks_data[stock] = data
            except Exception as exc:
                print('%s(%r) generated an exception: %s' % (stock[1], stock[0], exc))

    return stocks_data


def load_config():
    """加载配置文件"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_latest_date(parquet_path):
    """获取某只股票的最新数据日期"""
    if not os.path.exists(parquet_path):
        return None
    try:
        df = pd.read_parquet(parquet_path)
        return pd.to_datetime(df['时间'].max())
    except Exception as e:
        logging.error(f"[分钟] 读取文件 {parquet_path} 失败: {str(e)}")
        return None


def get_latest_date_daily(parquet_path):
    """获取某只股票的最新日线数据日期"""
    if not os.path.exists(parquet_path):
        return None
    try:
        df = pd.read_parquet(parquet_path)
        return pd.to_datetime(df['日期'].max())
    except Exception as e:
        logging.error(f"[日线] 读取文件 {parquet_path} 失败: {str(e)}")
        return None


def fetch_minute_data(stock_code, start_date=None, end_date=None):
    """获取单只股票的分钟数据"""
    try:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).replace(hour=9, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        time.sleep(0.45)
            
        df = ak.stock_zh_a_hist_min_em(
            symbol=stock_code,
            start_date=start_date,
            end_date=end_date,
            period='1',
            adjust='qfq'
        )
        
        if df is not None and not df.empty:
            df['时间'] = pd.to_datetime(df['时间'])
            df['代码'] = stock_code
            return df
        return None
        
    except Exception as e:
        logging.error(f"[分钟] 获取股票 {stock_code} 分钟数据失败: {str(e)}")
        return None


def fetch_daily_data(stock_code, start_date=None, end_date=None):
    """获取单个股票的日线数据"""
    try:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
        else:
            start_date = pd.to_datetime(start_date).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        time.sleep(0.6)
        
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if df is not None and not df.empty:
            df['日期'] = pd.to_datetime(df['日期'])
            df['代码'] = stock_code
            return df
        return None
        
    except Exception as e:
        logging.error(f"获取股票 {stock_code} 日线数据失败: {str(e)}")
        return None


def process_stock(stock_code, minute_dir):
    """处理单只股票的分钟数据更新"""
    try:
        parquet_path = os.path.join(minute_dir, f'{stock_code}.parquet')
        
        latest_date = get_latest_date(parquet_path)
        
        df = fetch_minute_data(stock_code, 
                            start_date=latest_date.strftime('%Y-%m-%d %H:%M:%S') if latest_date else None)
        if df is None:
            return None
            
        if latest_date is not None and os.path.exists(parquet_path):
            try:
                old_df = pd.read_parquet(parquet_path)
                old_count = len(old_df)
                df = pd.concat([old_df, df]).drop_duplicates(subset=['时间'], keep='last').sort_values('时间')
                new_count = len(df) - old_count
            except Exception as e:
                logging.error(f"[分钟] 合并股票 {stock_code} 分钟数据失败: {str(e)}")
                return None
        else:
            new_count = len(df)
        
        if not df.empty:
            try:
                df.to_parquet(parquet_path, index=False)
                logging.info(f"[分钟] 更新股票 {stock_code} 分钟数据成功，总记录数: {len(df)}，新增记录数: {new_count}")
                return {'code': stock_code, 'total': len(df), 'new': new_count}
            except Exception as e:
                logging.error(f"[分钟] 保存股票 {stock_code} 分钟数据失败: {str(e)}")
                return None
            
    except Exception as e:
        logging.error(f"[分钟] 处理股票 {stock_code} 分钟数据失败: {str(e)}")
        return None


def process_stock_daily(stock_code, daily_dir):
    """处理单只股票的日线数据更新"""
    try:
        parquet_path = os.path.join(daily_dir, f'{stock_code}.parquet')
        latest_date = get_latest_date_daily(parquet_path)
        df = fetch_daily_data(stock_code, start_date=latest_date + timedelta(days=1) if latest_date else None)
        if df is None:
            return None
        if latest_date is not None and os.path.exists(parquet_path):
            try:
                old_df = pd.read_parquet(parquet_path)
                old_count = len(old_df)
                df = pd.concat([old_df, df]).drop_duplicates(subset=['日期'], keep='last').sort_values('日期')
                new_count = len(df) - old_count
            except Exception as e:
                logging.error(f"合并股票 {stock_code} 日线数据失败: {str(e)}")
                return None
        else:
            new_count = len(df)
        if not df.empty:
            try:
                df.to_parquet(parquet_path, index=False)
                logging.info(f"[日线] 更新股票 {stock_code} 数据成功，总记录数: {len(df)}，新增记录数: {new_count}")
                return {'code': stock_code, 'total': len(df), 'new': new_count}
            except Exception as e:
                logging.error(f"保存股票 {stock_code} 日线数据失败: {str(e)}")
                return None
    except Exception as e:
        logging.error(f"处理股票 {stock_code} 日线失败: {str(e)}")
        return None


def update_daily_data():
    """更新所有股票的日线数据"""
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    daily_dir = os.path.join(data_dir, 'daily')
    if not os.path.exists(daily_dir):
        os.makedirs(daily_dir)
    try:
        stock_list = ak.stock_zh_a_spot_em()
        if stock_list is None or stock_list.empty:
            logging.error("获取股票列表失败")
            return
            
        stock_codes = stock_list['代码'].tolist()
        results = {}
        failed_stocks = []
        
        pbar = tqdm(total=len(stock_codes), desc="更新日线数据", unit="只")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_stock = {executor.submit(process_stock_daily, code, daily_dir): code for code in stock_codes}
            for future in concurrent.futures.as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[stock] = result
                    else:
                        failed_stocks.append(stock)
                except Exception as exc:
                    logging.error(f'[日线] 股票 {stock} 处理失败: {exc}')
                    failed_stocks.append(stock)
                finally:
                    pbar.update(1)
                    pbar.set_postfix({
                        '成功': len(results),
                        '失败': len(failed_stocks),
                        '总记录': sum(r['total'] for r in results.values()) if results else 0,
                        '新增': sum(r['new'] for r in results.values()) if results else 0
                    })
        
        pbar.close()
        
        total_stocks = len(results)
        total_records = sum(r['total'] for r in results.values())
        total_new = sum(r['new'] for r in results.values())
        logging.info(f"[日线] 更新完成，成功处理 {total_stocks} 只股票，总记录数: {total_records}，新增记录数: {total_new}")
        if failed_stocks:
            logging.warning(f"[日线] 以下股票处理失败: {', '.join(failed_stocks)}")
    except Exception as e:
        logging.error(f"[日线] 更新数据失败: {str(e)}")


def retry_failed_stocks(failed_stocks, mode='daily'):
    """
    重试失败的股票数据获取，支持日线和分钟线
    Args:
        failed_stocks: 股票代码列表
        mode: 'daily' 或 'minute'
    """
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    if mode == 'minute':
        data_subdir = 'minute'
        process_func = process_stock
        desc = "重试失败股票(分钟)"
    else:
        data_subdir = 'daily'
        process_func = process_stock_daily
        desc = "重试失败股票(日线)"

    target_dir = os.path.join(data_dir, data_subdir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    results = {}
    still_failed = []
    pbar = tqdm(total=len(failed_stocks), desc=desc, unit="只")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_stock = {executor.submit(process_func, code, target_dir): code for code in failed_stocks}
        for future in concurrent.futures.as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                result = future.result()
                if result is not None:
                    results[stock] = result
                else:
                    still_failed.append(stock)
            except Exception as exc:
                logging.error(f'[{mode}] 股票 {stock} 处理失败: {exc}')
                still_failed.append(stock)
            finally:
                pbar.update(1)
                pbar.set_postfix({
                    '成功': len(results),
                    '失败': len(still_failed)
                })
    pbar.close()
    
    successful_count = len(results)
    logging.info(f"[{mode}] 重试完成，成功处理 {successful_count} 只股票.")
    if still_failed:
        logging.warning(f"[{mode}] 以下股票仍然处理失败: {', '.join(still_failed)}")
    return still_failed


def write_one_stock_daily(row, daily_dir):
    if pd.isna(row['收盘']):
        return 'delisted', row['代码']
    code = row['代码']
    parquet_path = os.path.join(daily_dir, f'{code}.parquet')
    row_df = pd.DataFrame([row])
    try:
        if os.path.exists(parquet_path):
            old_df = pd.read_parquet(parquet_path)
            combined = pd.concat([old_df, row_df]).drop_duplicates(subset=['日期'], keep='last')
            combined = combined.sort_values('日期')
            combined.to_parquet(parquet_path, index=False)
        else:
            row_df.to_parquet(parquet_path, index=False)
        return 'success', code
    except Exception as e:
        logging.error(f"[日线快照] 写入 {code} 失败: {e}")
        return 'failed', code


def initialize_hsgt_top10_data(force_rerun=False):
    """
    初始化沪深股通十大成交股数据, 获取过去10年的历史数据.
    由于API单次返回有行数限制，此函数会分批获取.
    Args:
        force_rerun (bool): 如果为 True, 将会删除现有文件并重新获取. 默认为 False.
    """
    config = load_config()
    token = config.get('tushare_token')
    if not token or 'your_tushare_pro_token' in token:
        logging.warning("[HSGT Init] Tushare token not configured in config.yaml, skipping.")
        return

    try:
        pro = ts.pro_api(token)
    except Exception as e:
        logging.error(f"[HSGT Init] Failed to initialize Tushare API: {e}")
        return
    
    data_dir = config.get('data_dir', 'E:/data')
    other_dir = os.path.join(data_dir, 'other')
    os.makedirs(other_dir, exist_ok=True)
    file_path = os.path.join(other_dir, 'hsgt_top10.parquet')

    if os.path.exists(file_path) and not force_rerun:
        logging.warning(f"[HSGT Init] 数据文件 {file_path} 已存在. 如需重新初始化, 请设置 force_rerun=True 或手动删除文件.")
        return

    total_start_date = datetime.now() - timedelta(days=11*365)
    date_ranges = []
    current_start = total_start_date
    while current_start <= datetime.now():
        current_end = current_start + timedelta(days=19)
        date_ranges.append((current_start.strftime('%Y%m%d'), current_end.strftime('%Y%m%d')))
        current_start = current_end + timedelta(days=1)

    all_dfs = []
    logging.info(f"[HSGT Init] 开始分批获取11年历史数据，共 {len(date_ranges)} 批...")
    pbar = tqdm(date_ranges, desc="[HSGT Init] 分批获取历史数据")
    for start_dt, end_dt in pbar:
        pbar.set_postfix_str(f"{start_dt} to {end_dt}")
        try:
            time.sleep(1)
            df = pro.hsgt_top10(start_date=start_dt, end_date=end_dt)
            if df is not None and not df.empty:
                all_dfs.append(df)
        except Exception as e:
            logging.error(f"[HSGT Init] 获取 {start_dt}-{end_dt} 数据失败: {e}")
            time.sleep(1)
            continue
            
    if not all_dfs:
        logging.warning("[HSGT Init] 未获取到任何历史数据.")
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.drop_duplicates(subset=['trade_date', 'ts_code', 'market_type'], keep='last', inplace=True)
    combined_df.sort_values(by=['trade_date', 'market_type', 'rank'], inplace=True)
    
    try:
        combined_df.to_parquet(file_path, index=False)
        logging.info(f"[HSGT Init] 初始化完成. 文件路径: {file_path}, 总行数: {len(combined_df)}.")
    except Exception as e:
        logging.error(f"[HSGT Init] 保存数据到 {file_path} 失败: {e}")


if __name__ == "__main__":
    setup_logging('data_fetcher')
    logging.info("运行数据初始化/工具脚本...")
    
    # --- One-off Initializations ---
    # print("Initializing HSGT Top 10 data...")
    # initialize_hsgt_top10_data(force_rerun=False) 
    # print("HSGT Top 10 data initialization complete.")

    # --- Utility Functions ---
    # raw = "301012, 301266, 002275"
    # failed_stocks = [code.strip() for code in raw.split(",") if code.strip()]
    # if failed_stocks:
    #     retry_failed_stocks(failed_stocks, mode='minute')
