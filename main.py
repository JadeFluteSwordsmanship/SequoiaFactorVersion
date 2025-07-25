# -*- encoding: UTF-8 -*-

import utils
import logging
import work_flow
import settings
import schedule
import time
import datetime
from pathlib import Path
from utils import setup_logging


def job():
    print(f"{datetime.datetime.now()} 执行job")
    if utils.is_trading_day():
        print(f"[{datetime.datetime.now()}] 检测到交易日，开始执行数据更新...")
        work_flow.prepare()
    else:
        print(f"[{datetime.datetime.now()}] 非交易日，跳过执行")


# 设置日志
setup_logging('main')

# 初始化设置
settings.init()

if settings.config['cron']:
    EXEC_TIME = "21:40"
    print(f"[{datetime.datetime.now()}] 启动定时任务模式，执行时间设置为: {EXEC_TIME}")
    logging.info(f"[{datetime.datetime.now()}] 启动定时任务模式，执行时间设置为: {EXEC_TIME}")
    print(f"[{datetime.datetime.now()}] 程序将持续运行，等待定时执行...")
    schedule.every().day.at(EXEC_TIME).do(job)

    while True:
        schedule.run_pending()
        time.sleep(2)
else:
    print(f"[{datetime.datetime.now()}] 启动单次执行模式")
    job()
