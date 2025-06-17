# -*- coding: UTF-8 -*-
import datetime
import akshare as ak
import logging


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