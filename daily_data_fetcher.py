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
        logging.error(f"读取文件 {parquet_path} 失败: {str(e)}")
        return None


def get_latest_date_daily(parquet_path):
    """获取某只股票的最新日线数据日期"""
    if not os.path.exists(parquet_path):
        return None
    try:
        df = pd.read_parquet(parquet_path)
        return pd.to_datetime(df['日期'].max())
    except Exception as e:
        logging.error(f"读取日线文件 {parquet_path} 失败: {str(e)}")
        return None


def fetch_minute_data(stock_code, start_date=None, end_date=None):
    """获取单只股票的分钟数据"""
    try:
        if start_date is None:
            # 设置为7天前的早上9点
            start_date = (datetime.now() - timedelta(days=7)).replace(hour=9, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        # 获取分钟数据
        df = ak.stock_zh_a_hist_min_em(
            symbol=stock_code,
            start_date=start_date,
            end_date=end_date,
            period="1",  # 1分钟级别
            adjust="qfq"
        )
        
        if df is None or df.empty:
            logging.warning(f"股票 {stock_code} 没有数据")
            return None
            
        return df
        
    except requests.exceptions.RequestException as e:
        logging.error(f"获取股票 {stock_code} 分钟数据网络请求失败: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"获取股票 {stock_code} 分钟数据失败: {str(e)}")
        return None


def fetch_daily_data(stock_code, start_date=None, end_date=None):
    """获取单只股票的日线数据"""
    try:
        if start_date is None:
            # 默认从1990年开始
            start_date = '19700101'
        else:
            # stock_zh_a_hist 需要 yyyymmdd 格式
            start_date = pd.to_datetime(start_date).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        else:
            end_date = pd.to_datetime(end_date).strftime('%Y%m%d')
        
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if df is None or df.empty:
            logging.warning(f"股票 {stock_code} 没有日线数据")
            return None
            
        return df
        
    except requests.exceptions.RequestException as e:
        logging.error(f"获取股票 {stock_code} 日线数据网络请求失败: {str(e)}")
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
                logging.error(f"合并股票 {stock_code} 分钟数据失败: {str(e)}")
                return None
        else:
            new_count = len(df)
        
        if not df.empty:
            try:
                # 保存为Parquet文件
                df.to_parquet(parquet_path, index=False)
                logging.info(f"更新股票 {stock_code} 分钟数据成功，总记录数: {len(df)}，新增记录数: {new_count}")
                return {'code': stock_code, 'total': len(df), 'new': new_count}
            except Exception as e:
                logging.error(f"保存股票 {stock_code} 分钟数据失败: {str(e)}")
                return None
            
    except Exception as e:
        logging.error(f"处理股票 {stock_code} 分钟数据失败: {str(e)}")
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
                df = pd.concat([old_df, df]).drop_duplicates(subset=['日期']).sort_values('日期')
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
            logging.error("获取股票列表失败")
            return
            
        stock_codes = stock_list['代码'].tolist()
        
        # 使用线程池并行处理
        results = {}
        failed_stocks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
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
                    logging.error(f'股票 {stock} 处理失败: {exc}')
                    failed_stocks.append(stock)
        
        # 打印汇总信息
        total_stocks = len(results)
        total_records = sum(r['total'] for r in results.values())
        total_new = sum(r['new'] for r in results.values())
        logging.info(f"更新完成，成功处理 {total_stocks} 只股票，总记录数: {total_records}，新增记录数: {total_new}")
        if failed_stocks:
            logging.warning(f"以下股票处理失败: {', '.join(failed_stocks)}")
            
    except Exception as e:
        logging.error(f"更新数据失败: {str(e)}")


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
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
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
        total_stocks = len(results)
        total_records = sum(r['total'] for r in results.values())
        total_new = sum(r['new'] for r in results.values())
        logging.info(f"[日线] 更新完成，成功处理 {total_stocks} 只股票，总记录数: {total_records}，新增记录数: {total_new}")
        if failed_stocks:
            logging.warning(f"[日线] 以下股票处理失败: {', '.join(failed_stocks)}")
    except Exception as e:
        logging.error(f"[日线] 更新数据失败: {str(e)}")


if __name__ == "__main__":
    # 设置日志
    setup_logging('data_fetcher')
    
    # 执行分钟数据更新
    update_minute_data()
    # 执行日线数据更新
    update_daily_data() 