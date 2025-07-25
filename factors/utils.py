from factors.factor_base import FactorBase
import time
import logging
import sys
sys.path.append('.')
from utils import setup_logging

def initialize_all_factors(codes=None, end_date=None, force=False):
    """
    初始化并批量计算所有注册因子，写入parquet。
    Args:
        codes: 股票代码列表，默认全市场
        end_date: 截止日期，默认今天
        force: 是否强制重算覆盖
    """
    setup_logging('factor_init')
    from tqdm import tqdm
    factors = FactorBase.get_all_factors()
    for name, factor_class in tqdm(list(factors.items()), desc="初始化因子", ncols=80):
        start_time = time.time()
        logging.info(f"[initialize_all_factors] 初始化因子: {name} 开始于 {time.strftime('%Y-%m-%d %H:%M:%S')}")
        factor_class.initialize_all(codes=codes, end_date=end_date, force=force)
        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(f"[initialize_all_factors] 因子: {name} 初始化完成，耗时 {elapsed:.2f} 秒，结束于 {time.strftime('%Y-%m-%d %H:%M:%S')}") 