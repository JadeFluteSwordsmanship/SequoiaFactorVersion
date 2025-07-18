# -*- encoding: UTF-8 -*-

import akshare as ak
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import time
import yaml
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tushare as ts
import talib as tl

from utils import setup_logging


# def fetch(code_name):
#     stock = code_name[0]
#     data = ak.stock_zh_a_hist(symbol=stock, period="daily", start_date="20220101", adjust="")

#     if data is None or data.empty:
#         logging.debug("股票："+stock+" 没有数据，略过...")
#         return

#     data['p_change'] = tl.ROC(data['收盘'], 1)

#     return data


# def run(stocks):
#     stocks_data = {}
#     with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
#         future_to_stock = {executor.submit(fetch, stock): stock for stock in stocks}
#         for future in concurrent.futures.as_completed(future_to_stock):
#             stock = future_to_stock[future]
#             try:
#                 data = future.result()
#                 if data is not None:
#                     data = data.astype({'成交量': 'double'})
#                     stocks_data[stock] = data
#             except Exception as exc:
#                 print('%s(%r) generated an exception: %s' % (stock[1], stock[0], exc))

#     return stocks_data


from settings import config


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
            
        time.sleep(0.5)
            
        df = ak.stock_zh_a_hist_min_em(
            symbol=stock_code,
            start_date=start_date,
            end_date=end_date,
            period='1',
            adjust=''
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
            start_date = 19700101
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
            adjust=""
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
        df = fetch_daily_data(stock_code, start_date=latest_date.strftime('%Y%m%d') if latest_date else None)
        if df is None:
            return None
        if os.path.exists(parquet_path):
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


def update_daily_qfq_data():
    """更新所有股票的日线数据"""
    data_dir = config.get('data_dir', 'E:/data')
    daily_dir = os.path.join(data_dir, 'daily_qfq')
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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
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


def retry_failed_stocks(failed_stocks, mode='daily_qfq'):
    """
    重试失败的股票数据获取，支持日线和分钟线
    Args:
        failed_stocks: 股票代码列表
        mode: 'daily_qfq' 或 'minute'
    """
    data_dir = config.get('data_dir', 'E:/data')
    if mode == 'minute':
        data_subdir = 'minute'
        process_func = process_stock
        desc = "重试失败股票(分钟)"
    else:
        data_subdir = 'daily_qfq'
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
    初始化沪深股通前十成交量数据, 获取过去10年的历史数据.
    由于API单次返回有行数限制，此函数会分批获取.
    Args:
        force_rerun (bool): 如果为 True, 将会删除现有文件并重新获取. 默认为 False.
    """
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
    logging.info(f"[HSGT Init] 开始分批获取11年沪深股通前十成交量历史数据，共 {len(date_ranges)} 批...")
    pbar = tqdm(date_ranges, desc="[HSGT Init] 分批获取沪深股通前十成交量历史数据")
    for start_dt, end_dt in pbar:
        pbar.set_postfix_str(f"{start_dt} to {end_dt}")
        try:
            time.sleep(1)
            df = pro.hsgt_top10(start_date=start_dt, end_date=end_dt)
            df['stock_code'] = df['ts_code'].apply(lambda x: x.split('.')[0] if '.' in x else x)
            if df is not None and not df.empty:
                all_dfs.append(df)
        except Exception as e:
            logging.error(f"[HSGT Init] 获取 {start_dt}-{end_dt} 数据失败: {e}")
            time.sleep(1)
            continue
            
    if not all_dfs:
        logging.warning("[HSGT Init] 未获取到任何沪深股通前十成交量历史数据.")
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.drop_duplicates(subset=['trade_date', 'ts_code', 'market_type'], keep='last', inplace=True)
    combined_df.sort_values(by=['trade_date', 'market_type', 'rank'], inplace=True)
    
    try:
        combined_df.to_parquet(file_path, index=False)
        logging.info(f"[HSGT Init] 沪深股通前十成交量数据初始化完成. 文件路径: {file_path}, 总行数: {len(combined_df)}.")
    except Exception as e:
        logging.error(f"[HSGT Init] 保存数据到 {file_path} 失败: {e}")


def initialize_daily_data(force_rerun=False):
    """
    初始化所有股票的不复权日线数据，获取完整历史数据。
    支持断点续传，按股票分批处理，优化并发性能。
    Args:
        force_rerun (bool): 如果为 True, 将会删除现有文件并重新获取. 默认为 False.
    """
    token = config.get('tushare_token')
    if not token or 'your_tushare_pro_token' in token:
        logging.warning("[Daily Init] Tushare token not configured in config.yaml, skipping.")
        return

    try:
        pro = ts.pro_api(token)
    except Exception as e:
        logging.error(f"[Daily Init] Failed to initialize Tushare API: {e}")
        return
    
    # 读取股票基本信息
    data_dir = config.get('data_dir', 'E:/data')
    stock_basic_path = os.path.join(data_dir, 'basics', 'stock_basic.parquet')
    if not os.path.exists(stock_basic_path):
        logging.error(f"[Daily Init] 股票基本信息文件不存在: {stock_basic_path}")
        return
    
    try:
        stock_df = pd.read_parquet(stock_basic_path)
        ts_codes = stock_df['ts_code'].tolist()  # ts_code格式：000001.SZ
        logging.info(f"[Daily Init] 读取到 {len(ts_codes)} 只股票")
    except Exception as e:
        logging.error(f"[Daily Init] 读取股票基本信息失败: {e}")
        return
    
    # 创建数据目录
    daily_dir = os.path.join(data_dir, 'daily')
    os.makedirs(daily_dir, exist_ok=True)
    
    # 断点续传：检查已处理的股票
    progress_file = os.path.join(daily_dir, 'init_progress.txt')
    processed_stocks = set()
    
    if os.path.exists(progress_file) and not force_rerun:
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                processed_stocks = set(line.strip() for line in f.readlines())
            logging.info(f"[Daily Init] 发现断点文件，已处理 {len(processed_stocks)} 只股票")
        except Exception as e:
            logging.warning(f"[Daily Init] 读取断点文件失败: {e}")
    
    # 过滤出未处理的股票
    remaining_ts_codes = [ts_code for ts_code in ts_codes if ts_code not in processed_stocks]
    logging.info(f"[Daily Init] 需要处理 {len(remaining_ts_codes)} 只股票")
    
    if not remaining_ts_codes and not force_rerun:
        logging.info("[Daily Init] 所有股票都已处理完成")
        return
    
    
    # 使用线程安全的计数器
    from threading import Lock
    progress_lock = Lock()  # 只保留进度文件写入的锁
    
    def process_single_stock(ts_code):
        try:
            # 第一次获取数据，不指定end_date
            time.sleep(0.075)
            df = pro.daily(ts_code=ts_code)
            
            if df is None or df.empty:
                logging.warning(f"[Daily Init] 股票 {ts_code} 无数据")
                return {'ts_code': ts_code, 'status': 'failed', 'error': 'no_data'}
            
            # 如果第一次获取的数据达到6000条，说明可能还有更早的数据
            if len(df) >= 6000:
                # 获取最早日期，继续获取更早的数据
                earliest_date = df['trade_date'].min()
                retry_count = 0
                max_retries = 3  # 最多重试3次
                
                while len(df) >= 6000 and retry_count < max_retries:
                    time.sleep(0.06)  # 频率控制
                    
                    # 获取更早的数据
                    earlier_df = pro.daily(ts_code=ts_code, end_date=earliest_date)
                    
                    # 如果返回空数据，可能是API问题，重试
                    if earlier_df is None or earlier_df.empty:
                        logging.warning(f"[Daily Init] 股票 {ts_code} 获取更早数据返回空，重试 {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        time.sleep(0.1)  # 重试前等待
                        continue
                    
                    # 合并数据
                    df = pd.concat([earlier_df, df], ignore_index=True)
                    df = df.drop_duplicates(subset=['ts_code','trade_date'], keep='last')
                    
                    # 更新最早日期
                    earliest_date = earlier_df['trade_date'].min()
                    
                    # 如果这次获取的数据少于6000条，说明已经到最早的数据了
                    if len(earlier_df) < 6000:
                        break
                    
                    retry_count = 0  # 重置重试计数
            
            # 最终排序 - 确保按日期从小到大排序
            df = df.sort_values(['trade_date'], ascending=True)
            
            # 添加计算列
            df['vwap'] = df['amount'] * 1000 / (df['vol'] * 100)
            df['stock_code'] = df['ts_code'].apply(lambda x: x.split('.')[0] if '.' in x else x)
            
            # 获取stock_code用于文件名（去掉.SZ/.SH后缀）
            stock_code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            
            # 保存到文件 - 使用stock_code作为文件名
            file_path = os.path.join(daily_dir, f'{stock_code}.parquet')
            df.to_parquet(file_path, index=False)
            
            # 更新进度 - 使用锁保护文件写入
            with progress_lock:
                processed_stocks.add(ts_code)
                # 写入进度文件
                with open(progress_file, 'a', encoding='utf-8') as f:
                    f.write(f'{ts_code}\n')
            
            logging.info(f"[Daily Init] 股票 {ts_code} (stock_code: {stock_code}) 处理完成，{len(df)} 条记录")
            return {'ts_code': ts_code, 'records': len(df), 'status': 'success'}
            
        except Exception as e:
            logging.error(f"[Daily Init] 处理股票 {ts_code} 失败: {e}")
            return {'ts_code': ts_code, 'status': 'failed', 'error': str(e)}
    
    pbar = tqdm(total=len(remaining_ts_codes), desc="[Daily Init] 获取股票日线数据", unit="只")
    
    # 统计变量
    success_count = 0
    failed_count = 0
    total_records = 0
    failed_stocks = []
        # 多线程处理股票 - 减少线程数以避免API限制
    max_workers = 5
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_ts_code = {executor.submit(process_single_stock, ts_code): ts_code for ts_code in remaining_ts_codes}
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_ts_code):
            ts_code = future_to_ts_code[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    success_count += 1
                    total_records += result['records']
                else:
                    failed_count += 1
                    failed_stocks.append(ts_code)
                
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': failed_count,
                    '总记录': total_records
                })
            except Exception as exc:
                logging.error(f"[Daily Init] 股票 {ts_code} 生成异常: {exc}")
                failed_count += 1
                failed_stocks.append(ts_code)
                pbar.update(1)
    
    pbar.close()
    
    # 输出统计信息
    logging.info(f"[Daily Init] 初始化完成:")
    logging.info(f"  成功处理: {success_count} 只股票")
    logging.info(f"  处理失败: {failed_count} 只股票")
    logging.info(f"  总记录数: {total_records}")
    
    if failed_stocks:
        logging.warning(f"[Daily Init] 失败的股票: {', '.join(failed_stocks)}")
        # 保存失败列表供后续重试
        failed_file = os.path.join(daily_dir, 'failed_stocks.txt')
        with open(failed_file, 'w', encoding='utf-8') as f:
            for stock in failed_stocks:
                f.write(f'{stock}\n')
        logging.info(f"[Daily Init] 失败股票列表已保存到: {failed_file}")


def initialize_daily_basic_data(force_rerun=False):
    """
    初始化所有股票的daily_basic数据，获取完整历史数据。
    支持断点续传，按股票分批处理，优化并发性能。
    Args:
        force_rerun (bool): 如果为 True, 将会删除现有文件并重新获取. 默认为 False.
    """
    token = config.get('tushare_token')
    if not token or 'your_tushare_pro_token' in token:
        logging.warning("[Daily Basic Init] Tushare token not configured in config.yaml, skipping.")
        return

    try:
        pro = ts.pro_api(token)
    except Exception as e:
        logging.error(f"[Daily Basic Init] Failed to initialize Tushare API: {e}")
        return
    
    # 读取股票基本信息
    data_dir = config.get('data_dir', 'E:/data')
    stock_basic_path = os.path.join(data_dir, 'basics', 'stock_basic.parquet')
    if not os.path.exists(stock_basic_path):
        logging.error(f"[Daily Basic Init] 股票基本信息文件不存在: {stock_basic_path}")
        return
    
    try:
        stock_df = pd.read_parquet(stock_basic_path)
        ts_codes = stock_df['ts_code'].tolist()  # ts_code格式：000001.SZ
        logging.info(f"[Daily Basic Init] 读取到 {len(ts_codes)} 只股票")
    except Exception as e:
        logging.error(f"[Daily Basic Init] 读取股票基本信息失败: {e}")
        return
    
    # 创建数据目录
    daily_basic_dir = os.path.join(data_dir, 'daily_basic')
    os.makedirs(daily_basic_dir, exist_ok=True)
    
    # 断点续传：检查已处理的股票
    progress_file = os.path.join(daily_basic_dir, 'init_progress.txt')
    processed_stocks = set()
    
    if os.path.exists(progress_file) and not force_rerun:
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                processed_stocks = set(line.strip() for line in f.readlines())
            logging.info(f"[Daily Basic Init] 发现断点文件，已处理 {len(processed_stocks)} 只股票")
        except Exception as e:
            logging.warning(f"[Daily Basic Init] 读取断点文件失败: {e}")
    
    # 过滤出未处理的股票
    remaining_ts_codes = [ts_code for ts_code in ts_codes if ts_code not in processed_stocks]
    logging.info(f"[Daily Basic Init] 需要处理 {len(remaining_ts_codes)} 只股票")
    
    if not remaining_ts_codes and not force_rerun:
        logging.info("[Daily Basic Init] 所有股票都已处理完成")
        return
    
    # 使用线程安全的计数器
    from threading import Lock
    progress_lock = Lock()  # 只保留进度文件写入的锁
    
    def process_single_stock(ts_code):
        try:
            # 第一次获取数据，不指定end_date
            time.sleep(0.33)
            df = pro.daily_basic(**{
                "ts_code": ts_code,
                "trade_date": "",
                "start_date": "",
                "end_date": "",
                "limit": "",
                "offset": ""
            }, fields=[
                "ts_code",
                "trade_date",
                "close",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe",
                "pe_ttm",
                "pb",
                "ps",
                "ps_ttm",
                "dv_ratio",
                "dv_ttm",
                "total_share",
                "float_share",
                "free_share",
                "total_mv",
                "circ_mv",
                "limit_status"
            ])
            
            if df is None or df.empty or len(df) == 0:
                logging.warning(f"[Daily Basic Init] 股票 {ts_code} 无数据")
                return {'ts_code': ts_code, 'status': 'failed', 'error': 'no_data'}
            
            # 如果第一次获取的数据达到6000条，说明可能还有更早的数据
            if len(df) >= 6000:
                # 获取最早日期，继续获取更早的数据
                earliest_date = df['trade_date'].min()
                retry_count = 0
                max_retries = 3  # 最多重试3次
                
                while len(df) >= 6000 and retry_count < max_retries:
                    time.sleep(0.32)  # 频率控制
                    
                    # 获取更早的数据
                    earlier_df = pro.daily_basic(**{
                        "ts_code": ts_code,
                        "trade_date": "",
                        "start_date": "",
                        "end_date": earliest_date,
                        "limit": "",
                        "offset": ""
                    }, fields=[
                        "ts_code",
                        "trade_date",
                        "close",
                        "turnover_rate",
                        "turnover_rate_f",
                        "volume_ratio",
                        "pe",
                        "pe_ttm",
                        "pb",
                        "ps",
                        "ps_ttm",
                        "dv_ratio",
                        "dv_ttm",
                        "total_share",
                        "float_share",
                        "free_share",
                        "total_mv",
                        "circ_mv",
                        "limit_status"
                    ])
                    
                    # 如果返回空数据，可能是API问题，重试
                    if earlier_df is None or earlier_df.empty or len(earlier_df) == 0:
                        logging.warning(f"[Daily Basic Init] 股票 {ts_code} 获取更早数据返回空，重试 {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        time.sleep(0.32)  # 重试前等待
                        continue
                    
                    # 合并数据
                    df = pd.concat([earlier_df, df], ignore_index=True)
                    df = df.drop_duplicates(subset=['ts_code','trade_date'], keep='last')
                    
                    # 更新最早日期
                    earliest_date = earlier_df['trade_date'].min()
                    
                    # 如果这次获取的数据少于6000条，说明已经到最早的数据了
                    if len(earlier_df) < 6000:
                        break
                    
                    retry_count = 0  # 重置重试计数
            
            # 最终排序 - 确保按日期从小到大排序
            df = df.sort_values(['trade_date'], ascending=True)
            
            # 添加stock_code列用于兼容性
            df['stock_code'] = df['ts_code'].apply(lambda x: x.split('.')[0] if '.' in x else x)
            
            # 获取stock_code用于文件名（去掉.SZ/.SH后缀）
            stock_code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            
            # 保存到文件 - 使用stock_code作为文件名
            file_path = os.path.join(daily_basic_dir, f'{stock_code}.parquet')
            df.to_parquet(file_path, index=False)
            
            # 更新进度 - 使用锁保护文件写入
            with progress_lock:
                processed_stocks.add(ts_code)
                # 写入进度文件
                with open(progress_file, 'a', encoding='utf-8') as f:
                    f.write(f'{ts_code}\n')
            
            logging.info(f"[Daily Basic Init] 股票 {ts_code} (stock_code: {stock_code}) 处理完成，{len(df)} 条记录")
            return {'ts_code': ts_code, 'records': len(df), 'status': 'success'}
            
        except Exception as e:
            logging.error(f"[Daily Basic Init] 处理股票 {ts_code} 失败: {e}")
            return {'ts_code': ts_code, 'status': 'failed', 'error': str(e)}
    
    pbar = tqdm(total=len(remaining_ts_codes), desc="[Daily Basic Init] 获取股票daily_basic数据", unit="只")
    
    # 统计变量
    success_count = 0
    failed_count = 0
    total_records = 0
    failed_stocks = []
    
    # 多线程处理股票 - 减少线程数以避免API限制
    max_workers = 2
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_ts_code = {executor.submit(process_single_stock, ts_code): ts_code for ts_code in remaining_ts_codes}
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_ts_code):
            ts_code = future_to_ts_code[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    success_count += 1
                    total_records += result['records']
                else:
                    failed_count += 1
                    failed_stocks.append(ts_code)
                
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': failed_count,
                    '总记录': total_records
                })
            except Exception as exc:
                logging.error(f"[Daily Basic Init] 股票 {ts_code} 生成异常: {exc}")
                failed_count += 1
                failed_stocks.append(ts_code)
                pbar.update(1)
    
    pbar.close()
    
    # 输出统计信息
    logging.info(f"[Daily Basic Init] 初始化完成:")
    logging.info(f"  成功处理: {success_count} 只股票")
    logging.info(f"  处理失败: {failed_count} 只股票")
    logging.info(f"  总记录数: {total_records}")
    
    if failed_stocks:
        logging.warning(f"[Daily Basic Init] 失败的股票: {', '.join(failed_stocks)}")
        # 保存失败列表供后续重试
        failed_file = os.path.join(daily_basic_dir, 'failed_stocks.txt')
        with open(failed_file, 'w', encoding='utf-8') as f:
            for stock in failed_stocks:
                f.write(f'{stock}\n')
        logging.info(f"[Daily Basic Init] 失败股票列表已保存到: {failed_file}")


def initialize_moneyflow_data(force_rerun=False):
    """
    初始化所有股票的moneyflow数据，获取完整历史数据。
    支持断点续传，按股票分批处理，优化并发性能。
    Args:
        force_rerun (bool): 如果为 True, 将会删除现有文件并重新获取. 默认为 False.
    """
    token = config.get('tushare_token')
    if not token or 'your_tushare_pro_token' in token:
        logging.warning("[Moneyflow Init] Tushare token not configured in config.yaml, skipping.")
        return

    try:
        pro = ts.pro_api(token)
    except Exception as e:
        logging.error(f"[Moneyflow Init] Failed to initialize Tushare API: {e}")
        return
    
    # 读取股票基本信息
    data_dir = config.get('data_dir', 'E:/data')
    stock_basic_path = os.path.join(data_dir, 'basics', 'stock_basic.parquet')
    if not os.path.exists(stock_basic_path):
        logging.error(f"[Moneyflow Init] 股票基本信息文件不存在: {stock_basic_path}")
        return
    
    try:
        stock_df = pd.read_parquet(stock_basic_path)
        ts_codes = stock_df['ts_code'].tolist()  # ts_code格式：000001.SZ
        logging.info(f"[Moneyflow Init] 读取到 {len(ts_codes)} 只股票")
    except Exception as e:
        logging.error(f"[Moneyflow Init] 读取股票基本信息失败: {e}")
        return
    
    # 创建数据目录
    moneyflow_dir = os.path.join(data_dir, 'moneyflow')
    os.makedirs(moneyflow_dir, exist_ok=True)
    
    # 断点续传：检查已处理的股票
    progress_file = os.path.join(moneyflow_dir, 'init_progress.txt')
    processed_stocks = set()
    
    if os.path.exists(progress_file) and not force_rerun:
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                processed_stocks = set(line.strip() for line in f.readlines())
            logging.info(f"[Moneyflow Init] 发现断点文件，已处理 {len(processed_stocks)} 只股票")
        except Exception as e:
            logging.warning(f"[Moneyflow Init] 读取断点文件失败: {e}")
    
    # 过滤出未处理的股票
    remaining_ts_codes = [ts_code for ts_code in ts_codes if ts_code not in processed_stocks]
    logging.info(f"[Moneyflow Init] 需要处理 {len(remaining_ts_codes)} 只股票")
    
    if not remaining_ts_codes and not force_rerun:
        logging.info("[Moneyflow Init] 所有股票都已处理完成")
        return
    
    # 使用线程安全的计数器
    from threading import Lock
    progress_lock = Lock()  # 只保留进度文件写入的锁
    
    def process_single_stock(ts_code):
        try:
            # 第一次获取数据，不指定end_date
            time.sleep(0.33)
            df = pro.moneyflow(**{
                "ts_code": ts_code,
                "trade_date": "",
                "start_date": "",
                "end_date": "",
                "limit": "",
                "offset": ""
            }, fields=[
                "ts_code",
                "trade_date",
                "buy_sm_vol",
                "buy_sm_amount",
                "sell_sm_vol",
                "sell_sm_amount",
                "buy_md_vol",
                "buy_md_amount",
                "sell_md_vol",
                "sell_md_amount",
                "buy_lg_vol",
                "buy_lg_amount",
                "sell_lg_vol",
                "sell_lg_amount",
                "buy_elg_vol",
                "buy_elg_amount",
                "sell_elg_vol",
                "sell_elg_amount",
                "net_mf_vol",
                "net_mf_amount",
                "trade_count"
            ])
            
            if df is None or df.empty or len(df) == 0:
                logging.warning(f"[Moneyflow Init] 股票 {ts_code} 无数据")
                return {'ts_code': ts_code, 'status': 'failed', 'error': 'no_data'}
            
            # 如果第一次获取的数据达到6000条，说明可能还有更早的数据
            if len(df) >= 6000:
                # 获取最早日期，继续获取更早的数据
                earliest_date = df['trade_date'].min()
                retry_count = 0
                max_retries = 3  # 最多重试3次
                
                while len(df) >= 6000 and retry_count < max_retries:
                    time.sleep(0.32)  # 频率控制
                    
                    # 获取更早的数据
                    earlier_df = pro.moneyflow(**{
                        "ts_code": ts_code,
                        "trade_date": "",
                        "start_date": "",
                        "end_date": earliest_date,
                        "limit": "",
                        "offset": ""
                    }, fields=[
                        "ts_code",
                        "trade_date",
                        "buy_sm_vol",
                        "buy_sm_amount",
                        "sell_sm_vol",
                        "sell_sm_amount",
                        "buy_md_vol",
                        "buy_md_amount",
                        "sell_md_vol",
                        "sell_md_amount",
                        "buy_lg_vol",
                        "buy_lg_amount",
                        "sell_lg_vol",
                        "sell_lg_amount",
                        "buy_elg_vol",
                        "buy_elg_amount",
                        "sell_elg_vol",
                        "sell_elg_amount",
                        "net_mf_vol",
                        "net_mf_amount",
                        "trade_count"
                    ])
                    
                    # 如果返回空数据，可能是API问题，重试
                    if earlier_df is None or earlier_df.empty or len(earlier_df) == 0:
                        logging.warning(f"[Moneyflow Init] 股票 {ts_code} 获取更早数据返回空，重试 {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        time.sleep(0.32)  # 重试前等待
                        continue
                    
                    # 合并数据
                    df = pd.concat([earlier_df, df], ignore_index=True)
                    df = df.drop_duplicates(subset=['ts_code','trade_date'], keep='last')
                    
                    # 更新最早日期
                    earliest_date = earlier_df['trade_date'].min()
                    
                    # 如果这次获取的数据少于6000条，说明已经到最早的数据了
                    if len(earlier_df) < 6000:
                        break
                    
                    retry_count = 0  # 重置重试计数
            
            # 最终排序 - 确保按日期从小到大排序
            df = df.sort_values(['trade_date'], ascending=True)
            
            # 添加stock_code列用于兼容性
            df['stock_code'] = df['ts_code'].apply(lambda x: x.split('.')[0] if '.' in x else x)
            
            # 获取stock_code用于文件名（去掉.SZ/.SH后缀）
            stock_code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            
            # 保存到文件 - 使用stock_code作为文件名
            file_path = os.path.join(moneyflow_dir, f'{stock_code}.parquet')
            df.to_parquet(file_path, index=False)
            
            # 更新进度 - 使用锁保护文件写入
            with progress_lock:
                processed_stocks.add(ts_code)
                # 写入进度文件
                with open(progress_file, 'a', encoding='utf-8') as f:
                    f.write(f'{ts_code}\n')
            
            logging.info(f"[Moneyflow Init] 股票 {ts_code} (stock_code: {stock_code}) 处理完成，{len(df)} 条记录")
            return {'ts_code': ts_code, 'records': len(df), 'status': 'success'}
            
        except Exception as e:
            logging.error(f"[Moneyflow Init] 处理股票 {ts_code} 失败: {e}")
            return {'ts_code': ts_code, 'status': 'failed', 'error': str(e)}
    
    pbar = tqdm(total=len(remaining_ts_codes), desc="[Moneyflow Init] 获取股票moneyflow数据", unit="只")
    
    # 统计变量
    success_count = 0
    failed_count = 0
    total_records = 0
    failed_stocks = []
    
    # 多线程处理股票 - 减少线程数以避免API限制
    max_workers = 2
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_ts_code = {executor.submit(process_single_stock, ts_code): ts_code for ts_code in remaining_ts_codes}
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_ts_code):
            ts_code = future_to_ts_code[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    success_count += 1
                    total_records += result['records']
                else:
                    failed_count += 1
                    failed_stocks.append(ts_code)
                
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': failed_count,
                    '总记录': total_records
                })
            except Exception as exc:
                logging.error(f"[Moneyflow Init] 股票 {ts_code} 生成异常: {exc}")
                failed_count += 1
                failed_stocks.append(ts_code)
                pbar.update(1)
    
    pbar.close()
    
    # 输出统计信息
    logging.info(f"[Moneyflow Init] 初始化完成:")
    logging.info(f"  成功处理: {success_count} 只股票")
    logging.info(f"  处理失败: {failed_count} 只股票")
    logging.info(f"  总记录数: {total_records}")
    
    if failed_stocks:
        logging.warning(f"[Moneyflow Init] 失败的股票: {', '.join(failed_stocks)}")
        # 保存失败列表供后续重试
        failed_file = os.path.join(moneyflow_dir, 'failed_stocks.txt')
        with open(failed_file, 'w', encoding='utf-8') as f:
            for stock in failed_stocks:
                f.write(f'{stock}\n')
        logging.info(f"[Moneyflow Init] 失败股票列表已保存到: {failed_file}")


def initialize_dividend_data(force_rerun=False):
    """
    初始化所有股票的分红数据，获取完整历史数据。
    支持断点续传，按股票分批处理，优化并发性能。
    Args:
        force_rerun (bool): 如果为 True, 将会删除现有文件并重新获取. 默认为 False.
    """
    token = config.get('tushare_token')
    if not token or 'your_tushare_pro_token' in token:
        logging.warning("[Dividend Init] Tushare token not configured in config.yaml, skipping.")
        return

    try:
        pro = ts.pro_api(token)
    except Exception as e:
        logging.error(f"[Dividend Init] Failed to initialize Tushare API: {e}")
        return
    
    # 读取股票基本信息
    data_dir = config.get('data_dir', 'E:/data')
    stock_basic_path = os.path.join(data_dir, 'basics', 'stock_basic.parquet')
    if not os.path.exists(stock_basic_path):
        logging.error(f"[Dividend Init] 股票基本信息文件不存在: {stock_basic_path}")
        return
    
    try:
        stock_df = pd.read_parquet(stock_basic_path)
        ts_codes = stock_df['ts_code'].tolist()  # ts_code格式：000001.SZ
        logging.info(f"[Dividend Init] 读取到 {len(ts_codes)} 只股票")
    except Exception as e:
        logging.error(f"[Dividend Init] 读取股票基本信息失败: {e}")
        return
    
    # 创建数据目录
    dividend_dir = os.path.join(data_dir, 'dividend')
    os.makedirs(dividend_dir, exist_ok=True)
    
    # 断点续传：检查已处理的股票
    progress_file = os.path.join(dividend_dir, 'init_progress.txt')
    processed_stocks = set()
    
    if os.path.exists(progress_file) and not force_rerun:
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                processed_stocks = set(line.strip() for line in f.readlines())
            logging.info(f"[Dividend Init] 发现断点文件，已处理 {len(processed_stocks)} 只股票")
        except Exception as e:
            logging.warning(f"[Dividend Init] 读取断点文件失败: {e}")
    
    # 过滤出未处理的股票
    remaining_ts_codes = [ts_code for ts_code in ts_codes if ts_code not in processed_stocks]
    logging.info(f"[Dividend Init] 需要处理 {len(remaining_ts_codes)} 只股票")
    
    if not remaining_ts_codes and not force_rerun:
        logging.info("[Dividend Init] 所有股票都已处理完成")
        return
    
    # 使用线程安全的计数器
    from threading import Lock
    progress_lock = Lock()  # 只保留进度文件写入的锁
    
    def process_single_stock(ts_code):
        try:
            time.sleep(0.35)
            df = pro.dividend(**{
                "ts_code": ts_code,
                "ann_date": "",
                "end_date": "",
                "record_date": "",
                "ex_date": "",
                "imp_ann_date": "",
                "limit": "",
                "offset": ""
            }, fields=[
                "ts_code",
                "end_date",
                "ann_date",
                "div_proc",
                "stk_div",
                "stk_bo_rate",
                "stk_co_rate",
                "cash_div",
                "cash_div_tax",
                "record_date",
                "ex_date",
                "pay_date",
                "div_listdate",
                "imp_ann_date",
                "base_date",
                "base_share",
                "update_flag"
            ])
            if df is None or df.empty or len(df) == 0:
                logging.warning(f"[Dividend Init] 股票 {ts_code} 无分红数据")
                return {'ts_code': ts_code, 'status': 'failed', 'error': 'no_data'}
            # 兼容性处理
            df['stock_code'] = df['ts_code'].apply(lambda x: x.split('.')[0] if '.' in x else x)
            # 去重
            df = df.drop_duplicates(subset=["ts_code", "end_date", "ann_date", "div_proc", "record_date"], keep='last')
            # 按公告日排序
            df = df.sort_values(["ann_date", "end_date", "record_date", "div_proc"], ascending=True)
            stock_code = ts_code.split('.')[0] if '.' in ts_code else ts_code
            file_path = os.path.join(dividend_dir, f'{stock_code}.parquet')
            df.to_parquet(file_path, index=False)
            with progress_lock:
                processed_stocks.add(ts_code)
                with open(progress_file, 'a', encoding='utf-8') as f:
                    f.write(f'{ts_code}\n')
            logging.info(f"[Dividend Init] 股票 {ts_code} (stock_code: {stock_code}) 处理完成，{len(df)} 条记录")
            return {'ts_code': ts_code, 'records': len(df), 'status': 'success'}
        except Exception as e:
            logging.error(f"[Dividend Init] 处理股票 {ts_code} 失败: {e}")
            return {'ts_code': ts_code, 'status': 'failed', 'error': str(e)}
    
    pbar = tqdm(total=len(remaining_ts_codes), desc="[Dividend Init] 获取股票分红数据", unit="只")
    success_count = 0
    failed_count = 0
    total_records = 0
    failed_stocks = []
    max_workers = 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ts_code = {executor.submit(process_single_stock, ts_code): ts_code for ts_code in remaining_ts_codes}
        for future in concurrent.futures.as_completed(future_to_ts_code):
            ts_code = future_to_ts_code[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    success_count += 1
                    total_records += result['records']
                else:
                    failed_count += 1
                    failed_stocks.append(ts_code)
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': failed_count,
                    '总记录': total_records
                })
            except Exception as exc:
                logging.error(f"[Dividend Init] 股票 {ts_code} 生成异常: {exc}")
                failed_count += 1
                failed_stocks.append(ts_code)
                pbar.update(1)
    pbar.close()
    logging.info(f"[Dividend Init] 初始化完成:")
    logging.info(f"  成功处理: {success_count} 只股票")
    logging.info(f"  处理失败: {failed_count} 只股票")
    logging.info(f"  总记录数: {total_records}")
    if failed_stocks:
        logging.warning(f"[Dividend Init] 失败的股票: {', '.join(failed_stocks)}")
        failed_file = os.path.join(dividend_dir, 'failed_stocks.txt')
        with open(failed_file, 'w', encoding='utf-8') as f:
            for stock in failed_stocks:
                f.write(f'{stock}\n')
        logging.info(f"[Dividend Init] 失败股票列表已保存到: {failed_file}")


if __name__ == "__main__":
    setup_logging('data_fetcher')
    logging.info("运行数据初始化/工具脚本...")
    
    # --- One-off Initializations ---
    # print("Initializing HSGT Top 10 data...")
    # initialize_hsgt_top10_data(force_rerun=False) 
    # print("HSGT Top 10 data initialization complete.")
    
    # print("Initializing Daily data...")
    # initialize_daily_data(force_rerun=False)
    # print("Daily data initialization complete.")
    
    # print("Initializing Daily Basic data...")
    # initialize_daily_basic_data(force_rerun=False)
    # print("Daily Basic data initialization complete.")
    
    # print("Initializing Moneyflow data...")
    # initialize_moneyflow_data(force_rerun=False)
    # print("Moneyflow data initialization complete.")
    
    # print("Initializing Dividend data...")
    # initialize_dividend_data(force_rerun=False)
    # print("Dividend data initialization complete.")
    
    # print("Retrying failed daily stocks...")
    # retry_failed_daily_stocks()
    # print("Daily retry complete.")
    # daily_data_fetcher.update_daily_data()

    # --- Utility Functions ---
    raw = "836871, 603722, 003007, 002657, 688435, 300391, 688615, 688095, 603226, 300421, 603697, 688767, 002723, 300541, 301283, 600396, 300525, 300972, 301092, 601077, 688489, 600675, 839729, 838837, 000031, 600051, 833394, 600981, 833171, 002592, 834770, 601069, 300521, 831768, 300268, 688410, 301127, 688776, 300658, 688328, 600360, 300779, 600989, 600399, 301266, 002956, 600743, 002197, 000526, 603766, 603893, 688633, 002295, 430564, 600894, 300190, 002734, 300043, 920108, 600770, 688226, 002620, 003040, 002316, 603661, 002364, 300748, 600988, 002915, 300910, 300813, 601965, 002890, 873833, 603321, 002161, 002929, 603488, 430090, 600547, 603538, 300266, 300839, 002170, 600744, 301261, 301227, 600658, 300942, 601886, 000711, 603201, 300856, 600128, 000890, 300872, 600228, 300931, 301488, 300167, 301004, 300198, 603268, 688411, 605255, 603585, 605162, 002891, 835184, 605151, 688501, 301033, 300348, 000965, 872351, 605286, 688627, 600323, 603350, 300778, 300211, 600350, 688058, 301056, 600604, 300984, 301259, 600606, 603045, 832023, 831370, 833873, 836961, 301192, 002721, 002225, 002852, 600841, 301168, 000010, 002908, 600173, 300763, 002623, 002961, 603112, 873132, 003003, 603811, 002235, 603171, 688717, 300694, 300650, 688480, 832662, 603855, 603130, 603955, 603980, 603081, 000514, 301345, 603778, 300512, 600152, 605196, 688583, 605336, 300950, 002380, 001896, 300662, 688656, 300436, 002774, 605136, 300472, 002899, 002602, 836807, 600200, 002727, 000903, 603388, 000972, 002581, 002700, 000627, 300195, 002978, 002951, 301265, 688118, 002645, 601512, 835305, 300326, 000545, 836892, 002209, 300280, 605259, 000803, 600930, 003037, 603176"
    failed_stocks = [code.strip() for code in raw.split(",") if code.strip()]
    if failed_stocks:
        retry_failed_stocks(failed_stocks, mode='minute')

