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
    if utils.is_trading_day():
        work_flow.prepare()


# 设置日志
setup_logging('main')

# 初始化设置
settings.init()

if settings.config['cron']:
    EXEC_TIME = "15:30"
    schedule.every().day.at(EXEC_TIME).do(job)

    while True:
        schedule.run_pending()
        time.sleep(2)
else:
    work_flow.prepare()
