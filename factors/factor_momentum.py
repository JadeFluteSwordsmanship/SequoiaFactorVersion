import pandas as pd
from factors.factor_base import factor_registry, FactorBase


class Momentum60D(FactorBase):
    name = "mom_60d"
    data_requirements = {'daily': {'window': 61}}  # 需要61天日线

    def compute(self, codes, end_date):
        data = self.fetch_data(codes, end_date)
        df = data['daily']
        df = df.sort_values(['code', 'trade_date'])
        df['mom_60d'] = df.groupby('code')['close'].transform(
            lambda x: x / x.rolling(60).mean() - 1
        )
        latest = df.groupby('code').tail(1)
        return latest[['code', 'trade_date', 'mom_60d']] 