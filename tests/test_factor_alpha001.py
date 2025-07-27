import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from factors.factor_volume_price_sync import *
from factors.factor_price_structure import *
from factors.factor_price_volatility_breakout import *
from factors import *
# mock data_reader.get_daily_data
import types
from time import time

def main():
    start_time = time()
    end_date = '2025-07-25'
    alpha = Alpha012()
    result = alpha.compute(alpha.list_current_stocks(), end_date)
    
    print(f'{alpha.name}因子输出:')
    print(result[result['date'] == '2025-07-25'])
    end_time = time()
    print(f'{alpha.name}因子计算时间: {end_time - start_time}秒')
    
    # print(result)

    # print(f"all factors: {alpha.list_all_factors().sort_values(by='name')}")

if __name__ == '__main__':
    main()