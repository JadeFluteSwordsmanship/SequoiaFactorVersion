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
import random
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
            
        time.sleep(0.9+random.random() * 0.1)
            
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
    import random
    """处理单只股票的分钟数据更新"""
    try:
        parquet_path = os.path.join(minute_dir, f'{stock_code}.parquet')
        
        latest_date = get_latest_date(parquet_path)
        
        df = fetch_minute_data(stock_code, 
                            start_date=latest_date.strftime('%Y-%m-%d %H:%M:%S') if latest_date else None)
        time.sleep(random.random() * 0.2)
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
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


def initialize_index_daily_data(force_rerun=False):
    """
    初始化所选指数的日线数据，获取完整历史数据。
    读取 basics/index_basic.parquet 中的 ts_code 列，逐个抓取并保存到 index 目录。
    支持断点续传，按指数分批处理，优化并发性能。
    Args:
        force_rerun (bool): 如果为 True, 将会删除现有进度并重新获取. 默认为 False.
    """
    token = config.get('tushare_token')
    if not token or 'your_tushare_pro_token' in token:
        logging.warning("[Index Daily Init] Tushare token not configured in config.yaml, skipping.")
        return

    try:
        pro = ts.pro_api(token)
    except Exception as e:
        logging.error(f"[Index Daily Init] Failed to initialize Tushare API: {e}")
        return
    
    # 读取指数基本信息（挑选过的指数列表）
    data_dir = config.get('data_dir', 'E:/data')
    index_basic_path_primary = os.path.join(data_dir, 'basics', 'index_basic.parquet')
    index_basic_path_fallback = os.path.join('D:/data', 'basics', 'index_basic.parquet')
    index_basic_path = index_basic_path_primary if os.path.exists(index_basic_path_primary) else index_basic_path_fallback
    if not os.path.exists(index_basic_path):
        logging.error(f"[Index Daily Init] 指数列表文件不存在: {index_basic_path_primary} 或 {index_basic_path_fallback}")
        return
    
    try:
        index_df = pd.read_parquet(index_basic_path)
        ts_codes = index_df['ts_code'].dropna().astype(str).tolist()
        logging.info(f"[Index Daily Init] 读取到 {len(ts_codes)} 个指数")
    except Exception as e:
        logging.error(f"[Index Daily Init] 读取指数列表失败: {e}")
        return
    
    # 创建数据目录
    index_dir = os.path.join(data_dir, 'index')
    os.makedirs(index_dir, exist_ok=True)
    
    # 断点续传：检查已处理的指数
    progress_file = os.path.join(index_dir, 'init_progress.txt')
    processed_indices = set()
    
    if os.path.exists(progress_file) and not force_rerun:
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                processed_indices = set(line.strip() for line in f.readlines())
            logging.info(f"[Index Daily Init] 发现断点文件，已处理 {len(processed_indices)} 个指数")
        except Exception as e:
            logging.warning(f"[Index Daily Init] 读取断点文件失败: {e}")
    
    # 过滤出未处理的指数
    remaining_ts_codes = [ts_code for ts_code in ts_codes if ts_code not in processed_indices]
    logging.info(f"[Index Daily Init] 需要处理 {len(remaining_ts_codes)} 个指数")
    
    if not remaining_ts_codes and not force_rerun:
        logging.info("[Index Daily Init] 所有指数都已处理完成")
        return
    
    # 仅用于进度文件写入的锁
    from threading import Lock
    progress_lock = Lock()
    
    def process_single_index(ts_code: str):
        try:
            # 第一次获取数据，不指定 end_date
            time.sleep(0.33)
            df = pro.index_daily(**{
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
                "open",
                "high",
                "low",
                "pre_close",
                "change",
                "pct_chg",
                "vol",
                "amount"
            ])
            
            if df is None or df.empty:
                logging.warning(f"[Index Daily Init] 指数 {ts_code} 无数据")
                return {'ts_code': ts_code, 'status': 'failed', 'error': 'no_data'}
            
            # 如果单次达到 6000 条，可能有更早数据，向前翻页
            if len(df) >= 6000:
                earliest_date = df['trade_date'].min()
                retry_count = 0
                max_retries = 3
                while len(df) >= 6000 and retry_count < max_retries:
                    time.sleep(0.32)
                    earlier_df = pro.index_daily(**{
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
                        "open",
                        "high",
                        "low",
                        "pre_close",
                        "change",
                        "pct_chg",
                        "vol",
                        "amount"
                    ])
                    if earlier_df is None or earlier_df.empty:
                        logging.warning(f"[Index Daily Init] 指数 {ts_code} 获取更早数据返回空，重试 {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        time.sleep(0.32)
                        continue
                    df = pd.concat([earlier_df, df], ignore_index=True)
                    df = df.drop_duplicates(subset=['ts_code','trade_date'], keep='last')
                    earliest_date = earlier_df['trade_date'].min()
                    if len(earlier_df) < 6000:
                        break
                    retry_count = 0
            
            # 排序
            df = df.sort_values(['trade_date'], ascending=True)
            
            # 保存文件：使用 ts_code 作为文件名（包含后缀），避免冲突
            file_name = f"{ts_code}.parquet"
            file_path = os.path.join(index_dir, file_name)
            df.to_parquet(file_path, index=False)
            
            # 更新断点进度
            with progress_lock:
                processed_indices.add(ts_code)
                with open(progress_file, 'a', encoding='utf-8') as f:
                    f.write(f'{ts_code}\n')
            
            logging.info(f"[Index Daily Init] 指数 {ts_code} 处理完成，{len(df)} 条记录")
            return {'ts_code': ts_code, 'records': len(df), 'status': 'success'}
        except Exception as e:
            logging.error(f"[Index Daily Init] 处理指数 {ts_code} 失败: {e}")
            return {'ts_code': ts_code, 'status': 'failed', 'error': str(e)}
    
    pbar = tqdm(total=len(remaining_ts_codes), desc="[Index Daily Init] 获取指数日线数据", unit="个")
    success_count = 0
    failed_count = 0
    total_records = 0
    failed_indices = []
    
    # 并发控制（指数接口限频谨慎设置）
    max_workers = 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ts_code = {executor.submit(process_single_index, ts_code): ts_code for ts_code in remaining_ts_codes}
        for future in concurrent.futures.as_completed(future_to_ts_code):
            ts_code = future_to_ts_code[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    success_count += 1
                    total_records += result['records']
                else:
                    failed_count += 1
                    failed_indices.append(ts_code)
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': failed_count,
                    '总记录': total_records
                })
            except Exception as exc:
                logging.error(f"[Index Daily Init] 指数 {ts_code} 生成异常: {exc}")
                failed_count += 1
                failed_indices.append(ts_code)
                pbar.update(1)
    pbar.close()
    
    logging.info(f"[Index Daily Init] 初始化完成:")
    logging.info(f"  成功处理: {success_count} 个指数")
    logging.info(f"  处理失败: {failed_count} 个指数")
    logging.info(f"  总记录数: {total_records}")
    
    if failed_indices:
        logging.warning(f"[Index Daily Init] 失败的指数: {', '.join(failed_indices)}")
        failed_file = os.path.join(index_dir, 'failed_indices.txt')
        with open(failed_file, 'w', encoding='utf-8') as f:
            for idx in failed_indices:
                f.write(f'{idx}\n')
        logging.info(f"[Index Daily Init] 失败指数列表已保存到: {failed_file}")

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

    # print("Initializing index daily data...")
    # initialize_index_daily_data(force_rerun=False)
    # print("index daily data initialization complete.")
    
    # print("Retrying failed daily stocks...")
    # retry_failed_daily_stocks()
    # print("Daily retry complete.")
    # daily_data_fetcher.update_daily_data()
    # from daily_data_fetcher import update_daily_basic_data, update_hsgt_top10_data
    # update_daily_basic_data(today='2025-07-18',today_ymd='20250718')
    # --- Utility Functions ---
    raw = "002500, 002086, 603155, 301210, 300453, 301568, 300755, 600983, 002830, 000768, 002573, 002343, 600248, 002022, 688505, 300752, 603811, 603711, 002469, 601038, 688633, 300848, 600239, 301585, 000928, 301281, 600673, 002300, 920641, 000026, 601019, 300470, 920810, 300534, 601598, 920260, 000430, 002939, 300913, 603528, 300996, 000078, 300498, 600874, 301031, 301246, 301109, 688520, 688293, 002746, 605358, 600729, 002125, 002373, 600367, 605020, 603075, 001266, 300170, 920946, 600862, 002673, 300981, 001299, 601222, 600639, 002546, 000876, 002771, 002923, 300305, 603880, 300520, 603737, 920931, 600540, 601990, 002440, 605259, 002138, 600873, 301267, 300654, 600356, 300519, 600273, 600979, 003022, 603357, 000056, 920578, 002677, 600051, 002081, 300601, 600936, 000789, 300545, 002051, 688100, 002587, 002458, 600339, 603038, 603107, 300743, 920670, 000539, 920833, 300155, 300775, 600959, 301230, 300321, 688247, 600649, 301361, 600694, 600180, 001313, 300326, 688420, 300160, 001219, 600982, 301177, 603786, 688223, 601599, 300527, 002003, 000862, 002221, 603708, 603377, 688553, 603350, 600883, 688029, 300759, 688019, 600826, 920753, 002912, 600373, 603517, 002116, 603501, 300235, 300142, 920118, 002993, 688726, 002802, 601375, 300863, 600721, 603172, 000027, 300951, 300762, 301033, 002797, 002211, 603139, 600061, 600858, 002186, 920866, 000096, 300883, 688612, 600300, 000766, 003039, 301225, 002056, 002367, 600409, 603628, 920675, 002174, 000543, 002732, 000757, 605001, 688403, 000999, 002015, 300595, 000030, 301567, 600006, 301633, 601136, 000501, 603277, 301314, 603055, 002986, 603877, 301165, 002198, 600287, 603056, 300633, 603059, 300030, 601696, 603018, 600668, 002841, 688636, 600645, 300977, 002524, 002791, 300598, 300482, 688670, 002350, 600210, 300619, 301589, 002073, 603599, 000965, 002719, 002701, 000852, 600864, 688184, 001258, 603214, 002668, 002162, 002187, 603698, 688003, 301009, 600148, 301419, 300210, 920099, 688475, 603833, 002597, 688350, 001388, 002404, 000100, 603801, 600499, 601216, 000682, 601233, 600021, 603117, 301276, 002414, 301309, 688133, 600346, 920001, 000155, 603970, 000978, 300851, 600987, 920056, 300982, 603267, 600865, 601677, 600847, 300663, 688085, 688033, 002364, 603337, 600158, 600853, 000157, 300500, 605100, 688189, 300444, 688399, 000404, 603365, 300616, 600168, 603533, 600108, 600288, 920275, 600955, 300473, 002558, 600072, 001239, 301006, 301058, 603860, 301349, 301089, 300176, 002905, 600335, 301032, 300332, 300903, 603370, 001328, 603948, 688087, 301613, 300215, 002309, 000695, 002984, 603108, 603559, 688002, 688072, 003020, 920242, 600790, 000546, 002428, 000503, 688248, 301098, 300929, 688310, 601568, 600789, 002322, 600359, 600753, 300025, 600844, 688418, 600681, 000032, 600773, 603418, 300613, 688655, 002738, 688570, 603126, 603717, 002452, 000605, 300625, 603799, 000731, 002532, 000504, 002008, 688287, 002893, 000767, 688617, 688016, 300872, 603310, 000159, 688569, 002833, 300259, 002861, 002121, 301149, 920305, 603633, 605399, 600720, 000762, 002630, 300994, 603355, 001360, 688479, 002091, 000686, 600152, 301317, 603281, 603318, 000506, 300608, 300838, 301198, 601021, 600893, 688219, 688603, 301182, 603477, 002672, 603920, 000921, 300788, 600191, 301373, 600502, 300347, 920010, 300393, 002995, 600714, 002119, 301286, 600482, 688200, 300351, 603311, 001391, 002267, 300122, 300445, 601339, 688798, 001260, 002372, 600155, 600320, 000591, 301397, 600820, 920346, 002693, 000903, 300311, 600298, 920239, 000822, 603060, 600166, 301298, 002556, 688646, 920418, 603022, 688062, 002519, 920510, 605009, 002623, 301102, 002025, 688221, 000407, 000913, 300294, 688015, 002988, 002739, 300546, 688375, 600302, 301321, 301068, 600567, 002687, 688575, 603262, 300866, 603676, 601156, 000528, 301265, 000541, 688435, 600741, 600216, 603379, 001324, 300447, 301175, 300172, 000563, 600266, 603119, 301280, 688143, 301036, 600289, 300680, 920857, 002840, 300917, 601500, 000636, 688232, 600830, 002383, 600621, 301335, 301287, 002495, 000681, 002749, 002785, 300858, 600189, 603806, 603221, 603266, 601002, 301578, 002955, 920057, 603103, 301088, 920566, 605369, 600328, 920007, 000967, 600099, 920077, 605166, 600052, 603073, 600506, 002391, 603356, 301199, 301512, 603779, 603900, 002292, 300939, 688229, 300422, 000995, 000710, 002845, 600505, 300551, 300480, 000711, 600877, 603156, 688468, 002826, 688472, 603013, 300962, 600184, 000949, 001306, 688819, 000531, 600397, 000059, 603129, 002601, 000415, 000016, 000633, 301081, 001279, 300990, 300246, 600707, 600517, 920008, 605198, 920080, 688345, 002931, 603681, 000720, 002908, 300283, 000892, 600345, 603029, 002999, 688193, 300605, 601669, 002832, 000550, 603893, 301331, 600821, 603132, 002686, 301237, 000878, 600271, 688289, 300147, 300834, 603341, 000069, 603703, 000663, 601155, 300144, 300856, 603803, 301608, 688057, 301580, 600459, 920089, 600927, 000516, 000009, 301153, 605099, 603682, 688356, 000987, 600418, 300850, 000868, 600785, 600396, 002408, 300164, 300538, 002763, 301195, 300735, 603798, 002063, 688659, 300782, 002405, 301260, 002606, 300343, 688586, 300647, 688053, 688618, 603260, 002128, 603936, 300232, 300875, 601865, 688351, 603322, 600977, 600973, 600419, 920575, 000599, 603206, 002801, 600365, 300811, 600643, 300879, 300795, 002929, 603881, 300365, 605599, 000428, 300615, 300610, 600227, 688377, 688526, 688368, 688321, 600975, 000690, 002059, 600833, 000705, 603396, 920553, 688580, 300201, 600327, 001387, 000721, 002446, 000012, 002129, 688252, 300531, 605186, 600073, 300251, 300103, 688681, 605300, 603220, 605122, 301658, 301100, 301218, 603012, 603041, 000514, 002707, 000639, 301113, 002538, 300074, 002496, 600528, 002285, 300563, 300264, 920058, 002956, 300423, 000697, 688508, 002876, 002254, 688398, 688733, 002429, 688001, 600637, 688334, 603693, 600601, 301018, 920116, 688566, 301012, 603538, 000779, 002662, 603072, 301001, 300844, 601117, 603300, 002569, 301370, 603193, 300622, 688349, 002060, 688429, 300384, 600638, 301313, 301488, 600526, 603136, 600611, 603005, 002878, 301037, 000670, 000948, 000750, 688533, 301092, 688121, 688093, 002836, 301602, 001323, 603727, 688695, 600685, 603955, 002682, 601099, 301186, 600988, 300862, 688317, 300684, 301132, 600390, 603990, 000917, 002555, 605162, 601162, 002122, 000603, 002335, 300193, 603909, 002901, 920519, 301303, 603068, 600805, 300824, 920395, 002218, 603316, 300441, 300456, 000993, 605069, 002850, 300835, 920030, 301059, 600768, 300231, 603926, 002377, 300265, 002627, 301301, 605336, 601311, 600556, 300094, 688026, 688217, 920304, 605218, 688511, 301395, 002815, 601238, 301282, 603585, 300676, 002822, 300218, 688592, 002561, 600135, 000557, 301152, 603212, 300813, 688296, 300333, 002873, 300407, 002389, 300298, 301219, 688105, 688571, 301025, 605118, 000728, 688565, 688045, 603409, 600763, 300091, 002698, 002757, 688084, 002463, 600019, 920198, 300682, 300674, 603298, 000400, 300533, 603085, 002387, 002103, 002795, 300532, 000958, 920896, 603519, 000002, 000055, 002926, 300931, 300761, 000525, 300079, 688329, 000715, 600734, 301238, 600187, 603086, 001259, 301206, 600439, 688137, 688262, 000783, 603823, 601717, 002234, 600340, 603153, 300926, 002853, 301192, 300375, 600965, 688114, 002856, 002753, 002230, 600391, 688576, 600699, 301158, 000707, 300508, 300134, 301322, 300561, 300949, 300387, 002835, 603768, 600765, 002659, 000957, 601890, 688710, 601168, 600743, 001326, 300119, 600477, 002607, 000652, 603578, 600056, 600871, 688291, 601828, 920627, 603186, 000911, 000635, 002925, 688699, 688098, 688179, 003017, 002705, 000593, 300805, 603520, 300827, 002613, 603332, 600491, 688162, 600481, 300098, 301631, 603327, 300154, 600010, 603598, 300711, 603912, 600815, 000065, 002829, 600845, 001367, 002328, 600882, 920768, 003019, 301101, 688484, 002253, 002847, 301588, 301368, 600609, 002467, 688239, 600284, 001356, 002576, 002688, 600120, 300894, 920703, 002250, 600619, 688652, 002722, 605138, 603328, 301603, 300417, 600280, 600319, 300256, 600861, 002310, 300988, 002605, 300278, 688212, 920699, 301263, 301353, 920212, 603105, 688182, 300420, 003025, 603106, 600576, 920523, 688152, 300562, 301555, 300226, 002863, 603790, 920682, 603729, 002068, 000818, 605258, 300928, 301156, 600171, 002712, 301190, 301159, 600237, 000668, 301586, 301151, 002093, 603488, 920261, 600218, 688432, 301127, 688378, 000049, 002972, 300335, 301277, 600303, 000839, 002340, 002421, 300724, 301051, 300288, 002582, 688662, 000727, 301155, 300987, 301548, 600736, 600133, 600822, 920029, 300659, 600202, 300636, 300802, 603182, 300412, 605158, 920068, 000601, 300575, 603158, 002193, 300245, 002362, 300947, 601588, 002989, 601886, 601016, 600657, 688530, 688488, 002315, 002049, 002239, 002862, 002622, 603416, 002580, 301508, 002890, 300732, 300244, 688276, 603617, 600495, 688561, 603882, 300550, 605337, 603766, 603051, 600718, 688265, 688286, 688616, 003042, 002642, 301359, 301592, 688246, 300050, 300696, 002579, 000547, 002902, 300253, 603657, 300908, 920230, 301628, 002236, 688119, 002395, 000926, 603170, 300585, 001216, 002809, 002697, 301552, 300783, 920139, 002123, 301083, 301061, 600192, 600895, 603016, 300334, 300797, 920665, 000509, 688202, 002635, 000533, 002194, 300461, 688330, 300299, 600293, 003007, 301170, 300537, 300592, 688679, 001319, 920832, 001338, 601106, 688385, 002843, 300228, 600308, 600679, 603421, 600727, 300694, 603076, 002641, 301557, 002631, 002442, 002494, 002297, 688231, 688096, 300803, 920415, 300889, 300614, 601678, 000712, 300397, 002165, 600250, 002134, 002061, 003003, 688315, 002676, 001359, 001208, 300460, 600603, 002241, 300853, 688188, 002248, 600617, 301332, 300906, 600757, 300837, 301212, 300635, 600101, 300279, 002413, 002572, 002438, 600198, 000629, 603508, 600742, 300377, 300075, 300792, 301525, 300034, 300032, 603976, 301072, 603637, 002095, 300092, 300383, 000068, 000402, 002306, 603728, 002319, 300833, 600560, 002456, 001230, 688601, 002598, 600810, 300438, 600249, 600222, 920284, 600467, 300188, 301328, 600383, 002139, 600606, 002516, 002489, 605222, 603816, 603819, 002483, 600816, 603258, 002800, 603185, 002695, 601899, 301529, 300960, 002431, 600739, 000703, 002545, 603917, 000661, 301041, 002277, 002879, 688161, 688325, 603368, 003015, 003031, 300923, 002151, 000420, 688522, 300641, 300396, 002666, 300402, 603700, 301136, 002715, 002416, 300560, 002700, 600338, 603688, 300427, 300163, 605183, 000935, 002127, 605151, 002449, 688678, 002903, 002482, 002930, 688039, 300594, 301028, 002694, 300695, 603286, 600149, 688793, 600110, 300860, 601600, 603062, 002933, 301160, 603082, 002870, 300008, 600732, 002470, 301052, 002152, 000980, 301029, 000975, 300769, 002316, 600448, 301372, 002144, 002172, 301396, 300557, 920685, 000813, 300938, 300374, 688285, 300275, 603908, 002296, 300248, 603050, 300169, 300118, 300077, 600487, 001368, 603239, 002400, 002889, 601133, 688728, 300566, 300449, 300425, 688606, 002852, 601996, 300907, 000785, 688581, 300501, 300010, 301339, 600325, 000812, 600569, 002748, 002816, 003027, 601608, 002439, 002491, 300292, 000829, 688573, 301278, 301519, 300230, 001231, 603697, 001914, 300212, 002970, 688032, 002299, 600697, 688529, 002824, 301315, 002337, 301320, 000821, 301297, 003030, 300080, 688531, 688595, 002398, 920640, 300900, 688278, 300710, 688981, 920402, 301203, 002599, 300669, 300148, 688767, 001366, 002949, 600818, 600160, 000953, 000830, 688079, 920953, 603949, 000778, 301509, 300587, 000985, 688355, 603201, 300733, 000672, 300489, 300936, 002577, 920122, 688013, 300995, 300371, 301096, 002609, 300291, 603210, 920098, 300902, 002615, 002448, 301336, 603325, 002009, 688148, 301022, 301138, 920378, 002105, 301024, 920195, 002946, 000815, 301566, 300691, 603331, 603897, 002272, 600157, 301505, 603788, 301556, 000537, 301167, 600476, 301157, 688281, 300496, 300861, 002006, 300418, 600633, 688409, 002247, 002266, 688538, 300678, 688560, 603738, 688178, 603344, 002540, 603039, 301076, 001269, 603529, 600589, 301636, 603818, 002875, 300085, 002224, 000565, 300130, 002985, 300776, 300350, 603638, 300236, 300084, 002486, 688439, 002692, 688361, 920146, 605296, 300697, 603033, 920128, 688667, 300479, 300845, 002891, 301362, 300307, 603535, 301049, 601010, 688191, 300871, 300013, 300046, 300722, 300763, 002268, 002460, 002363, 301226, 688227, 300671, 688459, 688332, 300133, 003037, 002271, 002245, 688267, 600722, 300781, 300217, 300514, 688787, 300640, 920642, 300190, 300197, 603506, 301139, 603095, 300567, 003018, 600516, 920717, 603871, 603330, 301629, 600452, 600232, 300739, 300992, 000923, 000408, 300785, 003040, 002783, 600486, 603668, 300123, 605598, 601567, 920370, 300751, 300071, 002351, 600724, 300828, 000530, 600855, 300589, 002655, 300364, 301382, 600230, 301131, 002235, 603116, 605488, 002311, 002987, 301227, 688567, 301023, 688280, 920779, 300143, 002777, 600500, 000677, 603895, 300583, 001270, 688314, 301498, 002226, 300168, 688207, 002975, 301536, 600201, 002585, 603066, 002055, 301487, 001318, 001223, 688006, 600261, 603929, 002369, 600808, 002593, 002053, 688275, 603706, 600537, 688687, 600728, 600538, 688686, 688362, 301299, 002565, 300173, 920227, 300741, 301399, 300442, 002812, 301168, 920066, 688469, 300829, 300303, 002057, 600185, 000792, 002647, 920706, 300825, 605500, 300627, 301248, 605366, 000758, 603916, 601360, 600375, 300459, 300590, 002648, 002714, 300956, 920371, 688653, 301468, 603339, 301591, 600749, 920021, 002178, 688213, 688078, 301188, 002877, 603713, 000035, 000909, 002240, 002691, 300821, 002671, 688226, 002866, 002214, 603607, 688004, 600200, 002361, 300354, 603991, 601121, 688425, 300516, 600835, 300107, 002302, 301030, 000701, 300067, 002523, 301378, 000795, 600828, 688059, 300295, 000882, 300918, 600626, 688297, 002137, 600586, 600571, 301366, 002730, 600251, 300535, 920599, 603236, 001268, 688318, 300019, 300011, 603053, 002564, 300771, 601789, 300730, 000615, 002533, 002283, 300209, 000722, 603719, 002849, 000560, 300626, 002394, 301171, 002160, 002855, 688393, 002909, 301135, 002803, 688076, 301618, 002660, 000885, 000859, 300970, 601908, 603348, 600838, 600641, 603017, 688101, 300243, 301050, 920017, 300778, 300021, 301220, 002120, 600053, 688225, 300796, 300559, 002108, 002535, 603466, 002106, 600331, 920679, 300220, 600456, 301289, 300836, 603197, 600879, 605319, 002888, 920368, 603887, 603950, 688426, 603696, 002037, 688183, 002243, 300126, 605389, 300454, 920179, 301133, 002980, 300352, 603100, 600562, 300353, 688551, 300284, 920508, 688292, 300916, 002681, 603048, 920002, 600793, 300027, 300320, 000020, 603809, 300624, 002207, 300709, 002892, 300112, 300497, 000558, 600399, 600426, 002810, 300823, 688455, 600783, 603982, 920837, 688499, 600318, 920663, 301077, 300580, 000823, 600585, 300131, 688721, 688456, 002354, 603273, 002237, 688579, 002657, 603179, 002869, 688789, 603028, 300315, 002526, 603557, 920926, 301290, 003011, 600361, 920871, 300690, 002111, 688371, 603879, 300345, 600802, 000816, 688701, 603057, 688692, 300967, 605090, 301057, 688035, 600604, 600706, 002355, 300603, 603516, 688629, 002548, 002149, 300293, 301630, 688516, 688396, 920870, 603383, 300542, 301319, 300182, 300657, 603995, 002517, 002402, 300310, 688165, 688060, 002380, 300101, 002547, 002427, 300499, 300385, 002153, 000532, 920570, 688299, 002522, 688103, 688778, 301381, 300712, 688400, 603176, 300547, 603067, 920802, 920873, 605298, 920455, 000936, 920425, 300007, 920454, 603610, 002510, 000960, 688584, 300213, 301558, 601231, 300726, 300252, 300484, 600579, 603458, 002476, 600444, 001896, 300249, 600751, 920942, 301279, 300549, 300166, 003038, 688069, 300637, 300348, 600470, 300925, 600497, 002279, 003021, 000561, 603983, 300467, 600693, 300415, 688503, 688106, 603091, 000519, 000551, 002403, 300536, 603595, 300662, 300565, 002012, 605580, 920943, 002445, 002792, 689009, 600435, 688466, 600246, 002031, 600170, 002098, 600666, 600966, 002423, 603969, 300102, 001337, 688111, 002628, 300109, 603980, 301016, 301596, 002961, 603583, 300492, 603177, 301193, 920088, 002663, 603499, 301187, 002048, 603023, 920271, 603004, 600778, 300196, 688515, 300233, 600888, 300607, 300780, 301078, 600370, 688027, 000850, 300493, 301048, 300370, 002206, 002625, 301302, 601778, 300643, 002140, 600206, 300847, 002101, 000070, 300072, 300437, 300300, 300281, 688023, 300618, 920000, 002426, 301255, 600868, 688698, 605086, 603042, 000678, 300053, 301632, 002088, 600819, 002536, 603194, 300756, 300819, 300161, 002575, 301606, 000925, 920427, 300978, 600203, 002376, 920964, 001336, 002471, 001226, 603035, 301528, 000655, 603123, 002685, 002146, 601177, 300099, 301273, 300887, 688707, 600235, 000761, 688080, 301046, 301622, 301011, 688118, 002900, 688602, 605376, 688088, 300009, 301609, 300106, 300679, 000751, 002465, 688107, 300319, 603937, 600111, 300446, 301179, 001311, 920957, 002083, 300346, 002290, 603960, 603019, 300997, 300386, 603102, 600797, 300644, 688283, 003023, 002632, 601399, 603227, 002191, 002775, 603002, 300342, 688509, 002183, 301283, 300553, 300773, 603958, 688556, 301306, 603219, 000685, 603231, 603906, 688448, 603078, 301571, 603527, 688090, 002552, 688203, 300971, 600848, 688693, 601615, 920039, 301459, 301511, 001400, 000060, 300984, 600630, 920019, 001288, 688160, 002767, 002102, 600961, 300443, 600060, 002307, 000901, 001211, 920090, 301538, 300083, 300578, 600507, 301575, 300840, 603015, 300195, 002759, 002472, 300586, 600794, 002072, 605305, 300455, 920735, 301252, 600169, 301095, 002141, 688010, 000759, 301266, 603306, 601005, 300191, 600610, 301312, 300063, 002406, 688071, 300466, 300317, 600100, 920208, 603010, 601208, 300513, 301015, 300145, 605333, 920992, 600884, 688025, 002164, 301413, 300798, 688357, 002155, 300069, 000301, 600151, 688665, 600262, 600460, 001300, 688095, 600366, 301045, 920016, 002094, 000825, 603616, 600650, 000856, 002026, 002096, 002624, 601388, 300556, 603218, 300700, 300509, 600322, 300503, 600231, 688047, 300689, 600984, 000555, 000709, 601702, 003009, 605388, 300528, 002937, 600156, 000612, 300183, 002291, 600761, 920033, 300815, 002154, 301217, 688049, 688717, 002443, 688510, 688017, 002233, 002881, 300830, 300882, 603275, 300518, 605365, 002539, 600106, 603285, 300720, 688613, 600379, 300100, 300801, 301189, 688777, 600117, 300227, 002527, 000881, 920374, 002858, 002762, 301002, 600309, 300041, 601992, 603171, 002368, 603767, 600522, 688370, 300606, 000702, 300002, 603977, 301040, 688383, 000676, 000010, 920792, 603125, 603829, 920123, 300219, 605286, 600570, 002298, 000889, 920060, 001255, 688327, 688138, 002570, 601718, 605196, 605289, 300285, 600063, 002530, 600475, 603898, 603297, 688323, 603165, 301268, 301526, 920974, 601089, 688159, 603280, 002209, 688562, 605077, 300736, 600866, 002249, 688102, 002674, 600183, 603890, 601698, 605006, 002180, 002520, 688648, 688309, 688238, 002696, 301008, 688257, 301197, 002617, 300054, 688552, 688523, 600588, 920662, 300986, 002045, 600653, 000803, 601279, 600550, 300488, 603278, 003006, 603956, 600590, 688558, 001395, 300540, 300718, 600592, 688230, 603885, 002050, 000036, 300976, 000630, 301502, 300905, 920526, 600962, 300296, 002478, 603045, 300078, 300645, 002258, 300224, 600118, 003016, 300940, 603115, 002357, 600307, 301392, 300433, 300843, 300774, 300854, 300035, 002255, 603363, 002848, 603859, 002195, 688376, 688129, 688169, 002515, 688333, 300539, 301053, 603058, 300135, 603031, 605167, 603359, 003005, 600678, 300177, 600316, 300263, 688600, 603093, 301066, 002827, 301550, 002269, 300440, 603569, 300571, 301122, 300665, 605128, 300651, 603138, 301079, 600389, 688716, 301216, 688295, 300809, 301070, 601113, 002347, 300474, 000628, 000887, 600939, 300816, 300382, 688113, 002979, 301383, 300579, 300522, 300403, 300004, 920167, 688311, 002212, 300777, 688411, 300369, 300136, 603601, 002263, 688036, 000920, 920469, 301325, 301185, 300113, 003043, 300024, 603021, 688168, 002899, 001696, 688259, 688073, 688339, 300410, 003036, 300471, 301163, 688512, 920693, 002921, 002645, 300716, 300428, 002953, 603602, 688720, 688099, 001209, 601618, 920394, 600162, 920726, 600654, 603629, 002378, 301128, 301233, 300989, 002743, 688065, 300600, 002683, 300258, 600715, 603650, 002158, 300670, 002965, 002480, 600489, 688685, 600362, 603131, 001207, 603666, 300898, 000880, 603208, 920476, 688347, 688018, 603238, 301196, 600881, 002273, 603113, 600549, 301108, 920082, 920533, 301208, 300483, 688305, 300512, 688610, 300316, 002085, 688639, 301038, 688081, 002453, 688146, 920491, 002295, 300661, 603867, 603630, 688543, 000811, 301229, 920982, 688244, 605179, 300576, 002638, 603052, 002027, 603985, 920751, 688630, 300584, 603001, 300569, 301356, 688757, 603187, 603191, 300873, 300468, 688680, 688279, 688132, 600207, 300723, 603269, 300328, 000959, 002951, 002126, 301446, 603360, 000066, 600376, 688251, 301326, 600850, 300687, 002066, 300867, 301518, 300043, 688563, 603680, 301292, 002492, 920592, 002021, 600103, 603997, 300965, 600745, 600496, 605155, 688359, 600782, 002667, 603978, 002182, 300323, 688322, 600282, 688041, 000586, 301007, 605088, 600527, 300786, 002992, 920429, 605123, 002222, 600105, 688201, 002455, 603380, 603159, 300490, 688328, 300922, 000970, 688052, 002184, 301093, 688500, 002895, 300255, 300400, 688608, 002365, 920274, 301510, 002420, 688485, 688122, 301421, 301611, 300953, 300487, 002042, 002487, 301600, 002735, 600580, 002531, 688577, 603507, 301517, 600737, 603815, 301026, 603606, 600425, 603716, 300596, 600038, 301141, 002733, 688312, 920781, 001212, 688196, 300017, 603838, 688147, 601727, 603081, 300229, 300839, 601929, 688379, 600233, 688166, 002097, 600595, 920245, 300260, 920719, 002434, 603993, 600676, 688709, 603036, 300541, 300779, 600536, 688097, 603211, 603662, 001379, 301087, 600545, 603915, 603992, 301228, 688326, 002915, 301000, 600770, 600337, 300660, 688037, 003002, 688605, 600635, 301261, 920422, 300139, 301662, 003001, 301369, 600363, 301307, 002104, 920895, 603687, 002952, 000058, 920576, 300885, 300959, 002664, 300045, 300057, 002910, 301522, 002643, 920100, 920879, 920639, 301150, 000831, 002490, 002896, 600521, 920273, 300129, 300655, 600509, 605588, 600686, 001332, 603518, 600990, 300901, 301125, 603150, 000777, 300904, 002679, 300950, 600208, 301063, 603663, 002501, 600691, 688550, 002534, 603009, 300968, 600096, 001283, 301191, 688186, 002338, 920174, 688126, 300056, 600416, 688568, 301019, 002562, 002639, 300609, 603505, 002747, 920247, 600114, 002430, 301091, 300820, 300818, 300868, 688077, 688320, 688518, 601956, 688199, 920690, 603173, 920357, 600113, 920807, 002052, 603618, 920046, 603889, 301110, 600777, 601958, 300740, 920790, 603135, 603169, 601798, 301327, 601226, 001331, 300478, 688083, 300768, 600644, 300250, 002170, 300128, 300368, 002611, 920175, 002202, 600547, 601799, 300772, 600581, 300930, 920786, 688308, 920925, 600255, 920061, 601515, 920906, 603215, 688372, 300340, 601100, 600259, 688089, 688369, 603271, 601086, 300892, 000592, 002115, 600711, 002919, 600353, 300409, 002734, 920826, 603456, 920299, 000617, 002488, 920015, 301269, 002044, 002215, 002418, 301388, 688210, 920580, 920834, 300881, 002629, 301056, 300969, 603151, 603181, 301121, 002251, 600744, 600268, 301162, 300686, 920961, 688270, 605169, 301232, 920720, 603388, 600326, 603398, 688657, 600980, 600869, 601116, 002796, 605189, 002549, 920593, 920184, 601020, 002778, 300631, 002067, 300432, 688117, 002177, 300638, 300692, 301107, 000819, 002366, 300748, 920438, 605589, 688038, 688007, 301021, 601137, 002765, 920300, 688596, 920651, 688157, 002046, 002846, 603588, 688156, 002017, 688786, 600382, 920892, 920145, 601218, 000932, 688401, 002225, 000753, 002084, 002998, 600468, 301082, 603979, 920978, 301043, 603667, 920396, 002256, 600226, 002616, 603216, 002941, 001282, 603090, 920225, 600889, 688719, 601069, 300658, 301181, 301013, 300018, 000045, 300066, 603391, 300617, 300270, 300388, 300276, 603690, 600301, 603080, 600141, 300703, 300713, 300464, 300491, 688776, 002069, 000559, 688258, 688585, 002131, 002468, 301234, 000657, 688591, 920509, 600392, 688480, 000006, 300952, 601611, 300935, 688306, 300920, 300290, 301590, 688519, 603011, 688128, 301120, 300963, 300602, 688066, 000422, 002392, 920839, 603758, 300877, 688799, 603110, 605255, 920748, 603878, 603863, 301209"
    
    failed_stocks = [code.strip() for code in raw.split(",") if code.strip()]
    print(len(failed_stocks))
    if failed_stocks:
        retry_failed_stocks(failed_stocks, mode='minute')

