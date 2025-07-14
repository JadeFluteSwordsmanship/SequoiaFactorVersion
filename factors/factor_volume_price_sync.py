import pandas as pd
import numpy as np
from .factor_base import FactorBase

class Alpha001(FactorBase):
    name = "Alpha001"
    direction = -1  # 因子值大代表量价背离，押注反转
    description = (
        "Alpha001：量价同步性短周期因子。\n"
        "该因子衡量股票在过去6个交易日内，成交量变化（Δlog(vol)）与日内收益率((close-open)/open)的横截面排名序列之间的相关性。\n"
        "具体做法：\n"
        "  - 每日对所有股票分别计算Δlog(成交量)和日内收益率，并在横截面上分别排名；\n"
        "  - 对每只股票，计算其最近6日这两个排名序列的相关系数，并取相反数。\n"
        "信号解读：\n"
        "  - 因子值为正，代表近期量变与价变同步性较弱，可能存在量价背离（如放量但价格未同步上涨），提示短期反转或调整风险；\n"
        "  - 因子值为负，代表量价同步性较强，量增价升或量减价跌，反映趋势延续性。\n"
        "该因子常用于捕捉短周期内量价关系的变化，辅助判断趋势持续或反转。"
    )
    data_requirements = {
        'daily': {'window': 10}  # 需要volume, open, close
    }

    def _compute_impl(self, data, end_date, batch=False):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # Δlog(vol)
        df['log_vol'] = np.log(df['vol'].replace(0, np.nan))
        df['delta_log_vol'] = df.groupby('stock_code')['log_vol'].diff()

        # (close - open) / open
        df['ret'] = (df['close'] - df['open']) / df['open']

        # 横截面排名（按日期，降序）
        df['rank_delta_log_vol'] = df.groupby('trade_date')['delta_log_vol'].rank(method='average', pct=False, ascending=False)
        df['rank_ret'] = df.groupby('trade_date')['ret'].rank(method='average', pct=False, ascending=False)

        # 计算 rolling correlation
        def corr_roll(x: pd.DataFrame):
            return x['rank_delta_log_vol'].rolling(window=6, min_periods=6).corr(x['rank_ret'])

        df['alpha001'] = df.groupby('stock_code')[['rank_delta_log_vol', 'rank_ret']].apply(corr_roll)
        df['alpha001'] = -1 * df['alpha001']  # 乘以 -1

        # 输出结果
        result = df[['stock_code', 'trade_date', 'alpha001']].dropna(subset=['alpha001']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'alpha001': 'value'
        })

        return result.reset_index(drop=True) 