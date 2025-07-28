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
        
        # 初始化缓存属性
        self._ic_cache = {}
        self._rank_ic_cache = None
        self._ir_cache = {}
        self._sharpe_long_only_cache = {}

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
        
        # 检查缓存
        if method in self._ic_cache:
            return self._ic_cache[method]
        
        ic_list = []
        for date, group in self.merged.groupby('date'):
            if method == 'pearson':
                ic = group['value'].corr(group['future_return'], method='pearson')
            elif method == 'spearman':
                ic = group['value'].corr(group['future_return'], method='spearman')
            else:
                raise ValueError('method must be "pearson" or "spearman"')
            ic_list.append({'date': date, 'IC': ic})
        
        result = pd.DataFrame(ic_list).set_index('date')
        # 缓存结果
        self._ic_cache[method] = result
        return result

    def calc_rank_ic(self):
        """
        计算每期的RankIC（秩相关系数，通常为Spearman相关系数）。
        返回: DataFrame，index为date，columns=['RankIC']
        """
        # 检查缓存
        if self._rank_ic_cache is not None:
            return self._rank_ic_cache
        
        result = self.calc_ic(method='spearman').rename(columns={'IC': 'RankIC'})
        # 缓存结果
        self._rank_ic_cache = result
        return result

    def calc_ir(self, method='pearson'):
        """
        计算IR（IC均值/IC标准差）。
        method: 'pearson' 或 'spearman'
        返回: float
        """
        # 检查缓存
        if method in self._ir_cache:
            return self._ir_cache[method]
        
        ic_series = self.calc_ic(method=method)['IC']
        result = ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan
        # 缓存结果
        self._ir_cache[method] = result
        return result

    def calc_sharpe_long_only(self, annualize=True, periods_per_year=252, top_n=5):
        """
        计算因子纯多头组合的Sharpe比率。
        annualize: 是否年化
        periods_per_year: 年化周期数（如252为日度）
        top_n: 选择因子值最高的前N只股票
        返回: float
        """
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        
        # 创建缓存键
        cache_key = (annualize, periods_per_year, top_n)
        
        # 检查缓存
        if cache_key in self._sharpe_long_only_cache:
            return self._sharpe_long_only_cache[cache_key]
        
        portfolio_returns = []
        for date, group in self.merged.groupby('date'):
            n = len(group)
            if n < top_n:
                continue
            # 选择因子值最高的top_n只股票，等权重买入
            top_stocks = group.nlargest(top_n, 'value')
            portfolio_return = top_stocks['future_return'].mean()
            portfolio_returns.append(portfolio_return)
        
        if len(portfolio_returns) == 0:
            result = np.nan
        else:
            returns_series = pd.Series(portfolio_returns)
            mean_return = returns_series.mean()
            std_return = returns_series.std()
            sharpe = mean_return / std_return if std_return != 0 else np.nan
            if annualize:
                sharpe = sharpe * np.sqrt(periods_per_year)
            result = sharpe
        
        # 缓存结果
        self._sharpe_long_only_cache[cache_key] = result
        return result

    def clear_cache(self):
        """
        清除所有缓存的计算结果，强制重新计算。
        在数据更新后调用此方法。
        """
        self._ic_cache.clear()
        self._rank_ic_cache = None
        self._ir_cache.clear()
        self._sharpe_long_only_cache.clear()

    def get_cache_info(self):
        """
        获取缓存信息，用于调试和监控。
        返回: dict，包含各缓存的键数量
        """
        return {
            'ic_cache_keys': list(self._ic_cache.keys()),
            'rank_ic_cached': self._rank_ic_cache is not None,
            'ir_cache_keys': list(self._ir_cache.keys()),
            'sharpe_cache_keys': list(self._sharpe_long_only_cache.keys())
        }
    
    def ic_stats(self, method='pearson'):
        ic_df = self.calc_ic(method=method)
        ic = ic_df['IC'].dropna()
        n = len(ic)
        mean = ic.mean()
        std = ic.std(ddof=1)
        pos_ratio = (ic > 0).mean()
        t_value = mean / (std / np.sqrt(n)) if std > 0 else np.nan
        return pd.Series({
            'mean': mean,
            'std': std,
            'IR': mean / std if std > 0 else np.nan,
            't': t_value,
            'pos_ratio': pos_ratio,
            'max': ic.max(),
            'min': ic.min(),
            'p5': ic.quantile(0.05),
            'p95': ic.quantile(0.95),
            'count': n
        })

    # 可扩展更多评估函数，如分组回测、回撤、净值曲线、回归等 