# -*- encoding: UTF-8 -*-

import akshare as ak
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import requests
import yaml
import concurrent.futures
from utils import setup_logging
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


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
            # 设置为7天前的早上9点
            start_date = (datetime.now() - timedelta(days=7)).replace(hour=9, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        # 添加延迟，避免请求过于频繁
        time.sleep(0.5)
            
        # 获取分钟数据
        df = ak.stock_zh_a_hist_min_em(
            symbol=stock_code,
            start_date=start_date,
            end_date=end_date,
            period='1',  # 1分钟
            adjust='qfq'  # 前复权
        )
        
        if df is not None and not df.empty:
            # 确保时间列是datetime类型
            df['时间'] = pd.to_datetime(df['时间'])
            # 添加股票代码列
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
            # start_date = '19700101'
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
        else:
            # stock_zh_a_hist 需要 yyyymmdd 格式
            start_date = pd.to_datetime(start_date).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 添加延迟，避免请求过于频繁
        time.sleep(0.6)
        
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if df is not None and not df.empty:
            # 确保日期列是datetime类型
            df['日期'] = pd.to_datetime(df['日期'])
            # 添加股票代码列
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
        
        # 获取最新数据日期
        latest_date = get_latest_date(parquet_path)
        
        # 获取数据
        df = fetch_minute_data(stock_code, 
                            start_date=latest_date.strftime('%Y-%m-%d %H:%M:%S') if latest_date else None)
        if df is None:
            return None
            
        # 如果已有数据，合并新旧数据
        if latest_date is not None and os.path.exists(parquet_path):
            try:
                old_df = pd.read_parquet(parquet_path)
                old_count = len(old_df)
                df = pd.concat([old_df, df]).drop_duplicates(subset=['时间']).sort_values('时间')
                new_count = len(df) - old_count
            except Exception as e:
                logging.error(f"[分钟] 合并股票 {stock_code} 分钟数据失败: {str(e)}")
                return None
        else:
            new_count = len(df)
        
        if not df.empty:
            try:
                # 保存为Parquet文件
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


def update_minute_data():
    """更新所有股票的分钟数据"""
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    minute_dir = os.path.join(data_dir, 'minute')
    
    if not os.path.exists(minute_dir):
        os.makedirs(minute_dir)
    
    try:
        # 获取所有A股列表
        stock_list = ak.stock_zh_a_spot_em()
        if stock_list is None or stock_list.empty:
            logging.error("[分钟] 获取股票列表失败")
            return
            
        stock_codes = stock_list['代码'].tolist()
        
        # 使用线程池并行处理，减少并发数
        results = {}
        failed_stocks = []
        
        # 创建进度条
        pbar = tqdm(total=len(stock_codes), desc="更新分钟数据", unit="只")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # 减少并发数
            future_to_stock = {executor.submit(process_stock, code, minute_dir): code for code in stock_codes}
            for future in concurrent.futures.as_completed(future_to_stock):
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
        
        # 创建进度条
        pbar = tqdm(total=len(stock_codes), desc="更新日线数据", unit="只")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 日线数据可以多一些并发
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
                    # 更新进度条描述
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
                    '失败': len(still_failed),
                    '总记录': sum(r['total'] for r in results.values()) if results else 0,
                    '新增': sum(r['new'] for r in results.values()) if results else 0
                })
    pbar.close()
    total_stocks = len(results)
    total_records = sum(r['total'] for r in results.values())
    total_new = sum(r['new'] for r in results.values())
    logging.info(f"[{mode}] 重试完成，成功处理 {total_stocks} 只股票，总记录数: {total_records}，新增记录数: {total_new}")
    if still_failed:
        logging.warning(f"[{mode}] 以下股票仍然处理失败: {', '.join(still_failed)}")
    return still_failed


def write_one_stock_daily(row, daily_dir):
    # 跳过最新价为NaN的股票
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


def update_daily_data_snapshot():
    """
    高效批量更新当日日线数据：
    1. 用 ak.stock_zh_a_spot_em() 获取所有A股收盘快照
    2. 整理为 daily parquet 格式
    3. 只更新当天数据，合并去重
    """
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    daily_dir = os.path.join(data_dir, 'daily')
    if not os.path.exists(daily_dir):
        os.makedirs(daily_dir)

    # 1. 获取所有A股快照
    try:
        spot_df = ak.stock_zh_a_spot_em()
    except Exception as e:
        logging.error(f"[日线快照] 获取A股快照失败: {e}")
        return
    if spot_df is None or spot_df.empty:
        logging.error("[日线快照] 获取A股快照为空")
        return

    # 2. 字段映射，整理为daily parquet格式
    today = datetime.now().strftime('%Y-%m-%d')
    daily_cols = ['日期','股票代码','开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌幅','涨跌额','换手率','代码']
    rename_map = {
        '代码': '代码',
        '名称': '名称',
        '今开': '开盘',
        '最新价': '收盘',
        '最高': '最高',
        '最低': '最低',
        '成交量': '成交量',
        '成交额': '成交额',
        '振幅': '振幅',
        '涨跌幅': '涨跌幅',
        '涨跌额': '涨跌额',
        '换手率': '换手率',
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
    max_workers = min(8, os.cpu_count() or 4)
    delisted_codes = []
    failed_codes = []
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(write_one_stock_daily, row, daily_dir) for idx, row in daily_df.iterrows()]
        for f in as_completed(futures):
            status, code = f.result()
            if status == 'success':
                success_count += 1
            elif status == 'delisted':
                delisted_codes.append(code)
            elif status == 'failed':
                failed_codes.append(code)
    logging.info(f"[日线快照] 批量更新完成，成功写入 {success_count} 只股票，退市/无最新价 {len(delisted_codes)} 只，写入失败 {len(failed_codes)} 只")
    if delisted_codes:
        logging.warning(f"[日线快照] 跳过退市/无最新价股票: {delisted_codes}")
    if failed_codes:
        logging.error(f"[日线快照] 写入失败股票: {failed_codes}")


if __name__ == "__main__":
    # 设置日志
    setup_logging('data_fetcher')
    
    # 执行分钟数据更新
    update_minute_data()
    
    # 执行日线数据更新
    update_daily_data()
    
    # 重试失败的股票
    # raw = "603398, 301377, 001314, 688210, 002200, 603186, 600360, 001268, 688362, 000820, 002980, 301282, 688300, 688652, 833346, 300943, 300042, 002986, 300308, 688449, 688726, 688543, 300502, 002685, 688129, 603228, 688683, 300337, 300159, 688798, 603065, 603657, 688552, 001309, 600199, 603083, 605258, 300868, 603758, 300968, 300131, 605118, 603920, 688020, 600589, 301183, 002636, 301148, 834058, 300173, 603063, 300820, 000638, 688178, 300870, 688386, 605599, 301169, 837212, 832876, 605133, 600810, 688700, 302132, 002558, 002837, 920111, 002846, 002521, 002190, 300457, 002669, 688668, 002519, 831961, 300719, 300537, 688127, 001298, 600533, 600610, 300274, 000530, 688170, 003019, 001395, 871263, 833533, 002094, 600601, 600844, 300951, 600367, 300680, 688262, 601838, 688450, 300353, 600191, 300939, 300339, 301387, 603161, 600963, 001388, 000508"
    # failed_stocks = [code.strip() for code in raw.split(",") if code.strip()]
    # if failed_stocks:
    #     retry_failed_stocks(failed_stocks, mode='minute')