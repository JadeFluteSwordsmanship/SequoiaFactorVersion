# -*- coding: UTF-8 -*-
import datetime
import akshare as ak
import logging
import os
import pandas as pd


def setup_logging(name='sequoia'):
    """设置日志配置
    
    Args:
        name: 日志文件名前缀，默认为'sequoia'
    """
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # 生成日志文件名，包含日期
    log_filename = f'logs/{name}_{datetime.datetime.now().strftime("%Y%m%d")}.log'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


# 是否是工作日
def is_weekday():
    return datetime.datetime.now().weekday() < 5


def is_trading_day():
    try:
        # 获取交易日历
        trade_cal = ak.tool_trade_date_hist_sina()
        today = datetime.datetime.now().date()  # 获取今天的日期对象
        # 检查今天是否在交易日历中
        return today in trade_cal['trade_date'].values
    except Exception as e:
        logging.error(f"获取交易日历失败: {str(e)}")
        # 如果获取失败，退回到工作日判断
        return is_weekday()


def get_today_str():
    return datetime.datetime.now().strftime('%Y-%m-%d')


def get_trading_dates(start_date: str = None, end_date: str = None, window: int = None):
    """
    获取指定日期范围或窗口的交易日列表
    支持三种用法：
    1. 指定start_date和window：返回start_date之后的window个交易日
    2. 指定end_date和window：返回end_date之前的window个交易日
    3. 指定start_date和end_date：返回区间内所有交易日
    Args:
        start_date: 开始日期，格式：'2023-01-01'
        end_date: 结束日期，格式：'2023-12-31'
        window: 交易日数量
    Returns:
        交易日列表，格式：['2023-01-03', '2023-01-04', ...]
    """
    try:
        # 使用akshare获取交易日历
        trading_calendar = ak.tool_trade_date_hist_sina()
        trading_calendar['trade_date'] = pd.to_datetime(trading_calendar['trade_date'])
        trading_dates = trading_calendar['trade_date'].dt.strftime('%Y-%m-%d').tolist()
        
        if start_date and window:
            # 从start_date起，取window个交易日
            start_dt = pd.to_datetime(start_date)
            filtered = [d for d in trading_dates if pd.to_datetime(d) >= start_dt]
            result = filtered[:window]
            logging.info(f"获取到 {len(result)} 个交易日 (from {start_date}, window={window})")
            return result
        elif end_date and window:
            # 取end_date及之前的window个交易日
            end_dt = pd.to_datetime(end_date)
            filtered = [d for d in trading_dates if pd.to_datetime(d) <= end_dt]
            result = filtered[-window:]
            logging.info(f"获取到 {len(result)} 个交易日 (to {end_date}, window={window})")
            return result
        elif start_date and end_date:
            # 区间所有交易日
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            result = [d for d in trading_dates if start_dt <= pd.to_datetime(d) <= end_dt]
            logging.info(f"获取到 {len(result)} 个交易日 (from {start_date} to {end_date})")
            return result
        else:
            logging.warning("get_trading_dates: 需要指定start_date+window，end_date+window，或start_date+end_date")
            return []
    except Exception as e:
        logging.error(f"获取交易日历失败: {e}")
        return []