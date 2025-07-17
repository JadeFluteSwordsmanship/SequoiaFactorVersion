import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from factors.factor_volume_price_sync import *

# mock data_reader.get_daily_data
import types

def main():
    end_date = '2025-07-17'
    alpha = Alpha045()
    result = alpha.compute(alpha.list_current_stocks()[:100], end_date)
    print(f'{alpha.name}因子输出:')
    print(result)

if __name__ == '__main__':
    main() 