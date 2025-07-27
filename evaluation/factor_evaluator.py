import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from typing import Optional, Union, Type
from factors.factor_base import FactorBase
from data_reader import get_daily_data, list_available_stocks
from utils import get_trading_dates

class FactorEvaluator:
    """
    因子评估器：支持直接传入FactorBase子类（或其实例），或直接传入DataFrame。
    自动计算因子值和未来收益率，支持period/buy_price/sell_price等参数。
    """
    def __init__(
        self,
        factor: Optional[Union[Type[FactorBase], FactorBase]] = None,
        codes: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: int = 1,
        buy_price: str = "close",
        sell_price: str = "close",
        factor_df: Optional[pd.DataFrame] = None,
        return_df: Optional[pd.DataFrame] = None,
        window: Optional[int] = None,
        **kwargs
    ):
        """
        factor: FactorBase子类或其实例（如Alpha001），优先使用
        codes: 股票代码列表，默认全市场
        start_date: 开始日期，如果为空则根据window自动推断
        end_date: 截止日期，默认今天
        period: 收益计算周期（几日收益）
        buy_price: 买入价字段（如"close"/"open"）
        sell_price: 卖出价字段（如"close"/"open"）
        factor_df: 直接传入的因子值DataFrame（['code','date','factor','value']）
        return_df: 直接传入的收益率DataFrame（['code','date','future_return']）
        window: 取数窗口，默认255 + period + 5
        """
        self.period = period
        self.start_date = start_date
        self.end_date = end_date
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.codes = codes
        self.factor = factor() if isinstance(factor, type) else factor
        self.factor_df = factor_df
        self.return_df = return_df
        self.window = window
        
        # 确定统一的时间范围
        if self.end_date is None:
            self.end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        if self.window is None:
                self.window = 255 + self.period + 5
        
        if self.start_date is None:

            trading_dates = get_trading_dates(end_date=self.end_date, window=self.window)
            if len(trading_dates) > 0:
                self.start_date = trading_dates[0]
            else:
                self.start_date = self.end_date
        
        # 自动加载数据
        if self.factor_df is None and self.factor is not None:
            self.factor_df = self._load_factor_df()
        if self.return_df is None:
            self.return_df = self._load_return_df()
        # 合并
        if self.factor_df is not None and self.return_df is not None:
            self.merged = pd.merge(self.factor_df, self.return_df, on=['code', 'date'], how='inner')
        else:
            self.merged = None

    def _load_factor_df(self):
        """优先读取因子文件，如果文件不存在则计算"""
        if self.codes is None:
            self.codes = list_available_stocks('daily')
        
        if self.factor is not None:
            # 优先尝试读取因子文件
            try:
                factor_df = self.factor.read_factor_file()
                if factor_df is not None and len(factor_df) > 0:
                    # 过滤时间范围
                    factor_df['date'] = pd.to_datetime(factor_df['date'])
                    mask = (factor_df['date'] >= self.start_date) & (factor_df['date'] <= self.end_date)
                    factor_df = factor_df[mask].copy()
                    
                    if len(factor_df) > 0:
                        # 确保列顺序一致
                        if set(['code','date','factor','value']).issubset(factor_df.columns):
                            return factor_df[['code','date','factor','value']]
                        elif set(['code','date','value']).issubset(factor_df.columns):
                            return factor_df[['code','date','value']]
                        else:
                            return factor_df
            except Exception as e:
                print(f"读取因子文件失败，将重新计算: {e}")
            
            # 如果文件不存在或读取失败，则计算
            df = self.factor.compute(self.codes, self.end_date, self.window)
            # 过滤时间范围
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                mask = (df['date'] >= self.start_date) & (df['date'] <= self.end_date)
                df = df[mask].copy()
            
            # 只保留必要列
            if set(['code','date','factor','value']).issubset(df.columns):
                return df[['code','date','factor','value']]
            elif set(['code','date','value']).issubset(df.columns):
                return df[['code','date','value']]
            else:
                return df
        return None

    def _load_return_df(self):
        """计算未来收益率，确保时间范围一致"""
        if self.codes is None:
            self.codes = list_available_stocks('daily')
        
        # 取足够窗口以确保覆盖时间范围
        daily = get_daily_data(self.codes, self.end_date, self.window)
        daily = daily.sort_values(['stock_code','trade_date'])
        
        # 过滤时间范围
        daily['trade_date'] = pd.to_datetime(daily['trade_date'])
        mask = (daily['trade_date'] >= self.start_date) & (daily['trade_date'] <= self.end_date)
        daily = daily[mask].copy()
        
        g = daily.groupby('stock_code')
        # 买入价shift
        if self.buy_price == 'close':
            buy = g[self.buy_price].shift(0)
            buy_shift = 0
        elif self.buy_price in ['open','high','low']:
            buy = g[self.buy_price].shift(-1)
            buy_shift = -1
        else:
            raise ValueError("buy_price仅支持'close'或'open'/'high'/'low'")

        sell = g[self.sell_price].shift(-self.period + buy_shift)
        ret = (sell - buy) / buy
        ret_df = daily[['stock_code','trade_date']].copy()
        ret_df['future_return'] = ret.values
        ret_df = ret_df.rename(columns={'stock_code':'code','trade_date':'date'})
        # 只保留有未来收益的行
        ret_df = ret_df.dropna(subset=['future_return'])
        return ret_df.reset_index(drop=True)

    def calc_ic(self, method='pearson'):
        """
        计算每期的IC（Information Coefficient）。
        method: 'pearson' 或 'spearman'
        返回: DataFrame，index为date，columns=['IC']
        """
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        ic_list = []
        for date, group in self.merged.groupby('date'):
            if method == 'pearson':
                ic = group['value'].corr(group['future_return'], method='pearson')
            elif method == 'spearman':
                ic = group['value'].corr(group['future_return'], method='spearman')
            else:
                raise ValueError('method must be "pearson" or "spearman"')
            ic_list.append({'date': date, 'IC': ic})
        return pd.DataFrame(ic_list).set_index('date')

    def calc_rank_ic(self):
        """
        计算每期的RankIC（秩相关系数，通常为Spearman相关系数）。
        返回: DataFrame，index为date，columns=['RankIC']
        """
        return self.calc_ic(method='spearman').rename(columns={'IC': 'RankIC'})

    def calc_ir(self, method='pearson'):
        """
        计算IR（IC均值/IC标准差）。
        method: 'pearson' 或 'spearman'
        返回: float
        """
        ic_series = self.calc_ic(method=method)['IC']
        return ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan

    def calc_sharpe(self, annualize=True, periods_per_year=252):
        """
        计算因子多空组合的Sharpe比率。
        annualize: 是否年化
        periods_per_year: 年化周期数（如252为日度）
        返回: float
        """
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        sharpe_list = []
        for date, group in self.merged.groupby('date'):
            n = len(group)
            if n < 10:
                continue
            long = group.nlargest(int(n * 0.3), 'value')['future_return'].mean()
            short = group.nsmallest(int(n * 0.3), 'value')['future_return'].mean()
            spread = long - short
            sharpe_list.append(spread)
        spread_series = pd.Series(sharpe_list)
        mean = spread_series.mean()
        std = spread_series.std()
        sharpe = mean / std if std != 0 else np.nan
        if annualize:
            sharpe = sharpe * np.sqrt(periods_per_year)
        return sharpe

    # 可扩展更多评估函数，如分组回测、回撤、净值曲线、回归等 