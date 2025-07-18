import pandas as pd
import numpy as np
from factors.factor_base import FactorBase

class Alpha002(FactorBase):
    name = "Alpha002"
    direction = -1  # 越小代表多空失衡加剧，可能反转
    description = (
        "Alpha002：涨跌幅增速/多空失衡度因子。\n"
        "公式：Alpha002 = -1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1)\n"
        "刻画多空失衡变动情况，用((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)表示多空力量不平衡度，\n"
        "再对该值做1阶差分并取相反数。\n"
        "信号解读：该因子越小，代表多空失衡度加速恶化，可能预示短期反转。"
    )
    data_requirements = {
        'daily': {'window': 3}  # 需要close, high, low，窗口3天保证能算出1阶差分
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 多空失衡度
        df['imbalance'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        # 防止分母为0
        df.loc[df['high'] == df['low'], 'imbalance'] = np.nan

        # 1阶差分
        df['delta_imbalance'] = df.groupby('stock_code')['imbalance'].diff()
        # 乘以-1
        df['alpha002'] = -1 * df['delta_imbalance']

        # 输出结果
        result = df[['stock_code', 'trade_date', 'alpha002']].dropna(subset=['alpha002']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'alpha002': 'value'
        })
        return result.reset_index(drop=True) 