import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr, pearsonr
from typing import Optional, Union, Type
from factors.factor_base import FactorBase
from data_reader import get_daily_data, list_available_stocks
from utils import get_trading_dates

# 添加新的工具函数
def _robust_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    """Clip tails to improve stability before correlation."""
    if x.empty or y.empty:
        return np.nan
    x = x.clip(x.quantile(0.005), x.quantile(0.995))
    y = y.clip(y.quantile(0.005), y.quantile(0.995))
    return x.corr(y, method=method)

def _config_key(*parts) -> tuple:
    """Immutable key for caching based on parts (dicts become tuples)."""
    out = []
    for p in parts:
        if isinstance(p, dict):
            out.append(tuple(sorted(p.items())))
        elif isinstance(p, (list, tuple, set)):
            out.append(tuple(p))
        else:
            out.append(p)
    return tuple(out)

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
        self._ir_cache = {}
        self._sharpe_long_only_cache = {}
        self._daily_ir_cache = {}
        self._group_returns_cache = {}
        # 新增缓存
        self._cache = {}

    def _load_factor_df(self):
        """优先读取因子文件，如果文件不存在则计算"""
        start_time = time.time()
        
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
                            elapsed_time = time.time() - start_time
                            print(f"_load_factor_df 读取文件完成，耗时: {elapsed_time:.4f} 秒")
                            return factor_df[['code','date','factor','value']]
                        elif set(['code','date','value']).issubset(factor_df.columns):
                            elapsed_time = time.time() - start_time
                            print(f"_load_factor_df 读取文件完成，耗时: {elapsed_time:.4f} 秒")
                            return factor_df[['code','date','value']]
                        else:
                            elapsed_time = time.time() - start_time
                            print(f"_load_factor_df 读取文件完成，耗时: {elapsed_time:.4f} 秒")
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
                elapsed_time = time.time() - start_time
                print(f"_load_factor_df 计算完成，耗时: {elapsed_time:.4f} 秒")
                return df[['code','date','factor','value']]
            elif set(['code','date','value']).issubset(df.columns):
                elapsed_time = time.time() - start_time
                print(f"_load_factor_df 计算完成，耗时: {elapsed_time:.4f} 秒")
                return df[['code','date','value']]
            else:
                elapsed_time = time.time() - start_time
                print(f"_load_factor_df 计算完成，耗时: {elapsed_time:.4f} 秒")
                return df
        return None

    def _load_return_df(self):
        """计算未来收益率，确保时间范围一致"""
        start_time = time.time()
        
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
        
        elapsed_time = time.time() - start_time
        print(f"_load_return_df 计算完成，耗时: {elapsed_time:.4f} 秒")
        
        return ret_df.reset_index(drop=True)

    def calc_ic(self):
        """
        计算每期的IC（Information Coefficient，皮尔逊相关系数）。
        返回: DataFrame，index为date，columns=['IC']
        """
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        
        # 检查缓存
        if 'ic' in self._ic_cache:
            return self._ic_cache['ic']
        
        start_time = time.time()
        ic_list = []
        for date, group in self.merged.groupby('date'):
            ic = group['value'].corr(group['future_return'], method='pearson')
            ic_list.append({'date': date, 'IC': ic})
        
        result = pd.DataFrame(ic_list).set_index('date')
        # 缓存结果
        self._ic_cache['ic'] = result
        
        elapsed_time = time.time() - start_time
        print(f"calc_ic 计算完成，耗时: {elapsed_time:.4f} 秒")
        
        return result

    def calc_rank_ic(self):
        """
        计算每期的RankIC（秩相关系数，Spearman相关系数）。
        返回: DataFrame，index为date，columns=['RankIC']
        """
        # 检查缓存
        if 'rank_ic' in self._ic_cache:
            return self._ic_cache['rank_ic']
        
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        
        start_time = time.time()
        rank_ic_list = []
        for date, group in self.merged.groupby('date'):
            rank_ic = group['value'].corr(group['future_return'], method='spearman')
            rank_ic_list.append({'date': date, 'RankIC': rank_ic})
        
        result = pd.DataFrame(rank_ic_list).set_index('date')
        # 缓存结果
        self._ic_cache['rank_ic'] = result
        
        elapsed_time = time.time() - start_time
        print(f"calc_rank_ic 计算完成，耗时: {elapsed_time:.4f} 秒")
        
        return result

    def calc_ir(self):
        """
        计算IR（IC均值/IC标准差）。
        返回: float
        """
        # 检查缓存
        if 'ir' in self._ir_cache:
            return self._ir_cache['ir']
        
        ic_series = self.calc_ic()['IC']
        result = ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan
        # 缓存结果
        self._ir_cache['ir'] = result
        return result

    def calc_rank_ir(self):
        """
        计算RankIR（RankIC均值/RankIC标准差）。
        返回: float
        """
        # 检查缓存
        if 'rank_ir' in self._ir_cache:
            return self._ir_cache['rank_ir']
        
        rank_ic_series = self.calc_rank_ic()['RankIC']
        result = rank_ic_series.mean() / rank_ic_series.std() if rank_ic_series.std() != 0 else np.nan
        # 缓存结果
        self._ir_cache['rank_ir'] = result
        return result

    def calc_sharpe_long_only(self, annualize=True, periods_per_year=252, top_n=5):
        """
        计算因子纯多头组合的Sharpe比率。
        
        Args:
            annualize: 是否年化
            periods_per_year: 年化周期数（如252为日度）
            top_n: 选择因子值最高的前N只股票
            
        Returns:
            float: Sharpe比率
            
        Note:
            当period > 1时，会自动调整年化计算以考虑period的影响。
            例如，period=5时，每5天才有一次收益，年化时会相应调整。
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
                # 调整年化计算，考虑period的影响
                # 如果period=5，那么每5天才有一次收益，所以年化时需要调整
                adjusted_periods = periods_per_year / self.period
                sharpe = sharpe * np.sqrt(adjusted_periods)
            result = sharpe
        
        # 缓存结果
        self._sharpe_long_only_cache[cache_key] = result
        return result
     
    def daily_tstat_topn(self, top_n: int = 5) -> pd.DataFrame:
        """
        口径C：当天选Top-N个股票的横截面t值
        t_t = sqrt(N) * mean / std
        
        返回: DataFrame，包含 ['date', 'tstat', 'tstat_cum']
        """
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        
        # 创建缓存键
        cache_key = _config_key("tstat_topn", top_n)
        
        # 检查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        start_time = time.time()
        rows = []
        for date, group in self.merged.groupby('date'):
            top_group = group.nlargest(top_n, 'value')
            if len(top_group) < 2:
                continue
            mu, sd = top_group['future_return'].mean(), top_group['future_return'].std()
            tval = np.nan if sd == 0 else np.sqrt(len(top_group)) * mu / sd
            rows.append((date, tval))
        
        t = pd.Series(dict(rows)).sort_index().rename('tstat')
        result = pd.DataFrame({'tstat': t, 'tstat_cum': t.fillna(0).cumsum()})
        
        # 缓存结果
        self._cache[cache_key] = result
        
        elapsed_time = time.time() - start_time
        print(f"daily_tstat_topn 计算完成，耗时: {elapsed_time:.4f} 秒")
        
        return result

    def daily_ir_ic(self, window: Optional[int] = 60, method: str = "spearman") -> pd.DataFrame:
        """
        口径A：IR_IC（当日IC的z-score）
        IR_t = (IC_t - μ) / σ
        
        Args:
            window: 滚动窗口，None表示全样本
            method: 'pearson' 或 'spearman'
            
        Returns:
            DataFrame: 包含 ['IC', 'IR', 'IR_cum']
        """
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        
        # 创建缓存键
        cache_key = _config_key("ir_ic", window, method)
        
        # 检查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        start_time = time.time()
        
        # 计算IC序列
        ic = (self.merged.groupby('date')
                .apply(lambda g: _robust_corr(g['value'], g['future_return'], method))
                .rename('IC').sort_index())
        
        # 计算IR
        if window is None:
            mu, sd = 0.0, ic.std()
        else:
            mu, sd = ic.rolling(window).mean(), ic.rolling(window).std()
        
        ir = (ic - mu) / sd
        result = pd.DataFrame({'IC': ic, 'IR': ir, 'IR_cum': ir.fillna(0).cumsum()})
        
        # 缓存结果
        self._cache[cache_key] = result
        
        elapsed_time = time.time() - start_time
        print(f"daily_ir_ic 计算完成，耗时: {elapsed_time:.4f} 秒")
        
        return result

    def daily_ir_longshort(self, n_groups: int = 5, window: int = 60) -> pd.DataFrame:
        """
        口径B：IR_LS（日度多空收益的z-score）
        先按因子分组，取Top-Bottom的等权差收益r_t^LS
        IR_t = (r_t^LS - μ) / σ
        
        Args:
            n_groups: 分组数量
            window: 滚动窗口
            
        Returns:
            DataFrame: 包含 ['LS', 'IR', 'IR_cum']
        """
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        
        # 创建缓存键
        cache_key = _config_key("ir_ls", n_groups, window)
        
        # 检查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        start_time = time.time()
        
        # 计算多空收益
        ls_rows = []
        for date, group in self.merged.groupby('date'):
            # 按因子值排序并分组
            q = pd.qcut(group['value'].rank(method='first'), n_groups, labels=False)
            group = group.assign(q=q)
            top = group[group.q == n_groups-1]['future_return'].mean()
            bot = group[group.q == 0]['future_return'].mean()
            ls_rows.append((date, top - bot))
        
        ls = pd.Series(dict(ls_rows)).sort_index().rename('LS')
        
        # 计算IR
        mu, sd = 0.0, ls.rolling(window).std()
        ir = (ls - mu) / sd
        result = pd.DataFrame({'LS': ls, 'IR': ir, 'IR_cum': ir.fillna(0).cumsum()})
        
        # 缓存结果
        self._cache[cache_key] = result
        
        elapsed_time = time.time() - start_time
        print(f"daily_ir_longshort 计算完成，耗时: {elapsed_time:.4f} 秒")
        
        return result

    # 保持向后兼容性
    def calc_daily_ir(self, top_n: int = 5) -> pd.DataFrame:
        """
        向后兼容：调用daily_tstat_topn
        """
        return self.daily_tstat_topn(top_n)
     
    def clear_cache(self):
        """
        清除所有缓存的计算结果，强制重新计算。
        在数据更新后调用此方法。
        """
        self._ic_cache.clear()
        self._ir_cache.clear()
        self._sharpe_long_only_cache.clear()
        self._daily_ir_cache.clear()
        self._group_returns_cache.clear()
        self._cache.clear()

    def get_cache_info(self):
        """
        获取缓存信息，用于调试和监控。
        返回: dict，包含各缓存的键数量
        """
        return {
            'ic_cache_keys': list(self._ic_cache.keys()),
            'ir_cache_keys': list(self._ir_cache.keys()),
            'sharpe_cache_keys': list(self._sharpe_long_only_cache.keys()),
            'daily_ir_cache_keys': list(self._daily_ir_cache.keys()),
            'group_returns_cache_keys': list(self._group_returns_cache.keys()),
            'new_cache_keys': list(self._cache.keys())
        }
    
    def ic_stats(self):
        """
        计算IC的统计信息。
        返回: pd.Series，包含mean, std, IR, t, pos_ratio等统计量
        """
        ic_df = self.calc_ic()
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

    def rank_ic_stats(self):
        """
        计算RankIC的统计信息。
        返回: pd.Series，包含mean, std, RankIR, t, pos_ratio等统计量
        """
        rank_ic_df = self.calc_rank_ic()
        rank_ic = rank_ic_df['RankIC'].dropna()
        n = len(rank_ic)
        mean = rank_ic.mean()
        std = rank_ic.std(ddof=1)
        pos_ratio = (rank_ic > 0).mean()
        t_value = mean / (std / np.sqrt(n)) if std > 0 else np.nan
        return pd.Series({
            'mean': mean,
            'std': std,
            'RankIR': mean / std if std > 0 else np.nan,
            't': t_value,
            'pos_ratio': pos_ratio,
            'max': rank_ic.max(),
            'min': rank_ic.min(),
            'p5': rank_ic.quantile(0.05),
            'p95': rank_ic.quantile(0.95),
            'count': n
        })

    def calc_group_returns(self, n_groups: int = 5, weight_type: str = 'equal'):
        """
        计算分组回测收益
        
        Args:
            n_groups: 分组数量，默认5组
            weight_type: 权重类型，'equal'为等权，'weighted'为市值加权
            
        Returns:
            DataFrame: 包含每组的累计净值曲线
            
        Note:
            当period > 1时，会自动将period天收益转换为等效日收益再计算累计净值。
            这确保了不同period设置下的累计收益计算正确性。
        """
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        
        # 缓存键
        cache_key = (n_groups, weight_type)
        
        # 检查缓存
        if cache_key in self._group_returns_cache:
            return self._group_returns_cache[cache_key]
        
        start_time = time.time()
        
        # 按日期分组计算
        group_returns = {}
        dates = []
        
        for date, group in self.merged.groupby('date'):
            if len(group) < n_groups:  # 股票数量少于分组数，跳过
                continue
                
            dates.append(date)
            
            # 按因子值排序并分组
            sorted_group = group.sort_values('value', ascending=False)
            group_size = len(sorted_group) // n_groups
            
            # 计算每组的收益
            for i in range(n_groups):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < n_groups - 1 else len(sorted_group)
                
                group_stocks = sorted_group.iloc[start_idx:end_idx]
                
                if weight_type == 'equal':
                    # 等权重
                    group_return = group_stocks['future_return'].mean()
                elif weight_type == 'weighted':
                    # 市值加权（这里用因子值作为权重，实际应用中可能需要真实市值）
                    weights = group_stocks['value'].abs()  # 使用因子值的绝对值作为权重
                    weights = weights / weights.sum()  # 归一化
                    group_return = (group_stocks['future_return'] * weights).sum()
                else:
                    raise ValueError("weight_type 必须是 'equal' 或 'weighted'")
                
                if i not in group_returns:
                    group_returns[i] = []
                group_returns[i].append(group_return)
        
        # 转换为DataFrame
        result_df = pd.DataFrame(group_returns, index=dates)
        result_df.index.name = 'date'
        
        # 计算累计净值（从1开始）
        # 修正：当period > 1时，future_return是period天的收益，不能直接用cumprod
        # 需要将period天的收益转换为等效的日收益，然后再cumprod
        # 统一处理：将period天收益转换为等效日收益
        daily_equivalent_returns = (1 + result_df) ** (1 / self.period) - 1
        cumulative_df = (1 + daily_equivalent_returns).cumprod()
        
        # 缓存结果
        self._group_returns_cache[cache_key] = cumulative_df
        
        elapsed_time = time.time() - start_time
        print(f"calc_group_returns 计算完成，耗时: {elapsed_time:.4f} 秒")
        
        return cumulative_df
    
    def plot_group_returns(self, n_groups: int = 5, weight_type: str = 'equal', 
                          figsize: tuple = (12, 8), show_plot: bool = True, save: bool = True):
        """
        绘制分组回测净值曲线图
        
        Args:
            n_groups: 分组数量，默认5组
            weight_type: 权重类型，'equal'为等权，'weighted'为市值加权
            figsize: 图片大小
            show_plot: 是否显示图片
            save: 是否保存图片
        """
        # 计算分组收益
        cumulative_df = self.calc_group_returns(n_groups, weight_type)
        
        if cumulative_df.empty:
            print(f"没有分组回测数据可绘制")
            return
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 定义颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # 绘制每组净值曲线
        for i in range(n_groups):
            if i in cumulative_df.columns:
                color = colors[i % len(colors)]
                label = f'第{i+1}组'
                if i == 0:
                    label += f' (因子值最大 {100//n_groups}%)'
                elif i == n_groups - 1:
                    label += f' (因子值最小 {100//n_groups}%)'
                else:
                    label += f' ({i*100//n_groups}%-{(i+1)*100//n_groups}%)'
                
                ax.plot(cumulative_df.index, cumulative_df[i], 
                       color=color, linewidth=2, marker='o', markersize=3, 
                       label=label, alpha=0.8)
        
        # 设置图表属性
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('净值', fontsize=12)
        
        # 设置x轴日期格式 - 优化显示效果
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # 根据数据量动态调整标签间隔
        if len(cumulative_df) > 50:
            interval = max(1, len(cumulative_df) // 20)  # 最多显示20个标签
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        else:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        
        # 优化标签显示
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # 自动调整布局以避免标签被截断
        plt.subplots_adjust(bottom=0.15)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 设置标题
        weight_text = "等权" if weight_type == 'equal' else "加权"
        title = f"分组净值曲线 - {weight_text}，{self.factor.name}，{n_groups}组"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 添加图例
        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 显示最终净值
        if not cumulative_df.empty:
            final_values = cumulative_df.iloc[-1]
            for i, value in final_values.items():
                ax.text(0.02, 0.95 - i*0.05, f'第{i+1}组: {value:.4f}', 
                       transform=ax.transAxes, fontsize=9, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='lightblue', alpha=0.7))
        
        # 保存图片
        if save:
            weight_suffix = "等权" if weight_type == 'equal' else "加权"
            save_path = f"E:/data/回测/{self.factor.name}_分组净值_{weight_suffix}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        # 显示图片
        if show_plot:
            plt.show()
        
        return fig, ax
    
    def plot_equal_weight_returns(self, n_groups: int = 5, figsize: tuple = (12, 8), 
                                 show_plot: bool = True, save: bool = True):
        """
        绘制等权分组回测净值曲线图
        
        Args:
            n_groups: 分组数量，默认5组
            figsize: 图片大小
            show_plot: 是否显示图片
            save: 是否保存图片
        """
        return self.plot_group_returns(n_groups, 'equal', figsize, show_plot, save)
    
    def plot_weighted_returns(self, n_groups: int = 5, figsize: tuple = (12, 8), 
                             show_plot: bool = True, save: bool = True):
        """
        绘制加权分组回测净值曲线图
        
        Args:
            n_groups: 分组数量，默认5组
            figsize: 图片大小
            show_plot: 是否显示图片
            save: 是否保存图片
        """
        return self.plot_group_returns(n_groups, 'weighted', figsize, show_plot, save)

    # 可扩展更多评估函数，如分组回测、回撤、净值曲线、回归等 
     
    def plot_ic_chart(self, ic_type: str = 'IC', figsize: tuple = (12, 8), 
                    show_plot: bool = True, save: bool = True):
        """
        绘制IC分布图，包含每期IC柱状图和累积IC折线图
        
        Args:
            ic_type: 'IC' 或 'RankIC'，选择绘制哪种IC
            figsize: 图片大小
            show_plot: 是否显示图片
            save_path: 保存路径，如果为None则不保存
        """
        if ic_type.upper() == 'IC':
            ic_df = self.calc_ic()
            title = f"每期 IC 图，{self.factor.name}"
        elif ic_type.upper() == 'RANKIC':
            ic_df = self.calc_rank_ic()
            title = f"每期 RankIC 图，{self.factor.name}"
        else:
            raise ValueError("ic_type 必须是 'IC' 或 'RankIC'")
        
        if ic_df.empty:
            print(f"没有{ic_type}数据可绘制")
            return
        
        # 准备数据
        df = ic_df.copy()
        ic_col = 'IC' if ic_type.upper() == 'IC' else 'RankIC'
        df['累积' + ic_col] = df[ic_col].cumsum()
        
        # 创建图形
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制柱状图（左轴）
        bars = ax1.bar(df.index, df[ic_col], alpha=0.7, color='royalblue', 
                    width=0.8, label=ic_col)
        ax1.set_xlabel('日期')
        ax1.set_ylabel(ic_col, color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        
        # 设置x轴日期格式 - 优化显示效果
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # 根据数据量动态调整标签间隔
        if len(df) > 50:
            interval = max(1, len(df) // 20)  # 最多显示20个标签
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        else:
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        
        # 优化标签显示
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # 创建右轴
        ax2 = ax1.twinx()
        
        # 绘制累积线图（右轴）
        line = ax2.plot(df.index, df['累积' + ic_col], color='red', 
                    linewidth=2, marker='o', markersize=4, 
                    label='累积' + ic_col)
        ax2.set_ylabel('累积' + ic_col, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 添加网格
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 设置标题
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                bbox_to_anchor=(0.01, 0.99))
        
        # 调整布局 - 为x轴标签留出空间
        plt.subplots_adjust(bottom=0.15)
        
        # 显示当前值
        if not df.empty:
            current_ic = df[ic_col].iloc[-1]
            current_cum = df['累积' + ic_col].iloc[-1]
            ax1.text(0.02, 0.98, f'{ic_col}: {current_ic:.4f}', 
                    transform=ax1.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
            ax2.text(0.02, 0.92, f'累积{ic_col}: {current_cum:.4f}', 
                    transform=ax2.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='lightcoral', alpha=0.8))
        
        # 保存图片
        if save:
            plt.savefig(f"E:/data/回测/{self.factor.name}_ic.png", dpi=300, bbox_inches='tight')
            print(f"图片已保存到: E:/data/回测/{self.factor.name}_ic.png")
        
        # 显示图片
        if show_plot:
            plt.show()
        
        return fig, (ax1, ax2)
    
    def plot_daily_ir_chart(self, top_n: int = 5, figsize: tuple = (12, 8), 
                        show_plot: bool = True, save: bool = True):
        """
        绘制每日IR分布图，包含每期IR柱状图和累积IR折线图
        
        Args:
            top_n: 选择因子值最高的前N只股票
            figsize: 图片大小
            show_plot: 是否显示图片
            save_path: 保存路径，如果为None则不保存
        """
        daily_ir_df = self.calc_daily_ir(top_n)
        
        if daily_ir_df.empty:
            print(f"没有每日IR数据可绘制")
            return
        
        # 准备数据
        df = daily_ir_df.copy()
        df['累积IR'] = df['IR'].cumsum()
        
        # 创建图形
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制柱状图（左轴）
        bars = ax1.bar(df.index, df['IR'], alpha=0.7, color='royalblue', 
                    width=0.8, label='IR')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('IR', color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        
        # 设置x轴日期格式 - 优化显示效果
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # 根据数据量动态调整标签间隔
        if len(df) > 50:
            interval = max(1, len(df) // 20)  # 最多显示20个标签
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        else:
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        
        # 优化标签显示
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # 创建右轴
        ax2 = ax1.twinx()
        
        # 绘制累积线图（右轴）
        line = ax2.plot(df.index, df['累积IR'], color='red', 
                    linewidth=2, marker='o', markersize=4, 
                    label='IR累计值')
        ax2.set_ylabel('IR累计值', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 添加网格
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 设置标题
        plt.title(f"每期 IR 图 (Top {top_n})，{self.factor.name}", fontsize=14, fontweight='bold', pad=20)
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                bbox_to_anchor=(0.01, 0.99))
        
        # 调整布局 - 为x轴标签留出空间
        plt.subplots_adjust(bottom=0.15)
        
        # 显示当前值
        if not df.empty:
            current_ir = df['IR'].iloc[-1]
            current_cum = df['累积IR'].iloc[-1]
            ax1.text(0.02, 0.98, f'IR: {current_ir:.4f}', 
                    transform=ax1.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
            ax2.text(0.02, 0.92, f'IR累计值: {current_cum:.4f}', 
                    transform=ax2.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='lightcoral', alpha=0.8))
        
        # 保存图片
        if save:
            plt.savefig(f"E:/data/回测/{self.factor.name}_daily_ir.png", dpi=300, bbox_inches='tight')
            print(f"图片已保存到: E:/data/回测/{self.factor.name}_daily_ir.png")
        
        # 显示图片
        if show_plot:
            plt.show()
        
        return fig, (ax1, ax2)

    # 可扩展更多评估函数，如分组回测、回撤、净值曲线、回归等 

    # ==================== 新增绘图函数 ====================
    
    def plot_ir_distribution_panel(self, ir_df: pd.DataFrame, value_col: str = 'IR',
                                  title_top: str = "IR 分布图",
                                  title_bottom: str = "每期 IR 图",
                                  figsize: tuple = (12, 10), 
                                  show_plot: bool = True, save: bool = True):
        """
        绘制IR分布面板图：上图显示IR分布直方图+累计概率，下图显示每期IR柱状图+累计线
        
        Args:
            ir_df: 包含IR数据的DataFrame，必须有value_col列和可选的'IR_cum'列
            value_col: IR值的列名
            title_top: 上图标题
            title_bottom: 下图标题
            figsize: 图片大小
            show_plot: 是否显示图片
            save: 是否保存图片
        """
        if ir_df.empty:
            print(f"没有IR数据可绘制")
            return
        
        # 准备数据
        df = ir_df.copy()
        if 'IR_cum' not in df.columns:
            df['IR_cum'] = df[value_col].fillna(0).cumsum()
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1])
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 上图：IR分布直方图 + 累计概率
        ir_values = df[value_col].dropna()
        if len(ir_values) > 0:
            # 直方图
            n, bins, patches = ax1.hist(ir_values, bins=20, alpha=0.7, color='skyblue', 
                                       density=True, label='概率密度')
            ax1.set_xlabel(value_col)
            ax1.set_ylabel('概率密度', color='skyblue')
            ax1.tick_params(axis='y', labelcolor='skyblue')
            
            # 累计概率线
            ax1_twin = ax1.twinx()
            hist, bin_edges = np.histogram(ir_values, bins=20, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            cum_prob = np.cumsum(hist) / hist.sum()
            ax1_twin.plot(bin_centers, cum_prob, 'r-', linewidth=2, marker='o', 
                         markersize=4, label='累计概率')
            ax1_twin.set_ylabel('累计概率', color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
            
            # 图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.set_title(title_top, fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 下图：每期IR柱状图 + 累计线
        x = df.index
        bars = ax2.bar(x, df[value_col], alpha=0.7, color='royalblue', 
                      width=0.8, label=value_col)
        ax2.set_xlabel('日期')
        ax2.set_ylabel(value_col, color='royalblue')
        ax2.tick_params(axis='y', labelcolor='royalblue')
        
        # 设置x轴日期格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        if len(df) > 50:
            interval = max(1, len(df) // 20)
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        else:
            ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # 累计线（右轴）
        ax2_twin = ax2.twinx()
        line = ax2_twin.plot(x, df['IR_cum'], color='red', 
                           linewidth=2, marker='o', markersize=4, 
                           label='IR累计值')
        ax2_twin.set_ylabel('IR累计值', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # 图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax2.set_title(title_bottom, fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save and self.factor is not None:
            save_path = f"E:/data/回测/{self.factor.name}_ir_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        # 显示图片
        if show_plot:
            plt.show()
        
        return fig, (ax1, ax2)
    
    def plot_ir_ic_panel(self, window: Optional[int] = 60, method: str = "spearman",
                        figsize: tuple = (12, 10), show_plot: bool = True, save: bool = True):
        """
        绘制IR_IC面板图（口径A）
        """
        ir_df = self.daily_ir_ic(window, method)
        title_top = f"IR_IC 分布图 (window={window}, method={method})"
        title_bottom = f"每期 IR_IC 图 (window={window}, method={method})"
        return self.plot_ir_distribution_panel(ir_df, 'IR', title_top, title_bottom, 
                                             figsize, show_plot, save)
    
    def plot_ir_ls_panel(self, n_groups: int = 5, window: int = 60,
                        figsize: tuple = (12, 10), show_plot: bool = True, save: bool = True):
        """
        绘制IR_LS面板图（口径B）
        """
        ir_df = self.daily_ir_longshort(n_groups, window)
        title_top = f"IR_LS 分布图 (groups={n_groups}, window={window})"
        title_bottom = f"每期 IR_LS 图 (groups={n_groups}, window={window})"
        return self.plot_ir_distribution_panel(ir_df, 'IR', title_top, title_bottom, 
                                             figsize, show_plot, save)
    
    def plot_tstat_topn_panel(self, top_n: int = 5,
                             figsize: tuple = (12, 10), show_plot: bool = True, save: bool = True):
        """
        绘制tstat_TopN面板图（口径C）
        """
        tstat_df = self.daily_tstat_topn(top_n)
        title_top = f"tstat_Top{top_n} 分布图"
        title_bottom = f"每期 tstat_Top{top_n} 图"
        return self.plot_ir_distribution_panel(tstat_df, 'tstat', title_top, title_bottom, 
                                             figsize, show_plot, save)
    
    def plot_group_navs(self, n_groups: int = 5, overlap: bool = True,
                       figsize: tuple = (12, 8), show_plot: bool = True, save: bool = True):
        """
        绘制分组净值曲线图（重叠组合日频）
        
        Args:
            n_groups: 分组数量
            overlap: 是否处理重叠组合
            figsize: 图片大小
            show_plot: 是否显示图片
            save: 是否保存图片
        """
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        
        # 计算分组收益
        rows = {}
        for date, group in self.merged.groupby('date'):
            q = pd.qcut(group['value'].rank(method='first'), n_groups, labels=False)
            group = group.assign(q=q)
            for i in range(n_groups):
                gi = group[group.q == i]
                if len(gi) == 0:
                    continue
                r = gi['future_return'].mean()
                rows.setdefault(i, []).append((date, r))
        
        # 转换为DataFrame
        df = pd.DataFrame({i: pd.Series({d: v for d, v in rows_i}).sort_index()
                          for i, rows_i in rows.items()})
        
        # 处理重叠组合
        if overlap and self.period > 1:
            daily = df.rolling(self.period).mean()
        else:
            daily = df
        
        daily.index.name = 'date'
        
        # 计算净值
        daily = daily.sort_index().fillna(0.0)
        nav = (1 + daily).cumprod()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 定义颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # 绘制每组净值曲线
        for i in range(n_groups):
            if i in nav.columns:
                color = colors[i % len(colors)]
                label = f'第{i+1}组'
                if i == 0:
                    label += f' (因子值最大 {100//n_groups}%)'
                elif i == n_groups - 1:
                    label += f' (因子值最小 {100//n_groups}%)'
                else:
                    label += f' ({i*100//n_groups}%-{(i+1)*100//n_groups}%)'
                
                ax.plot(nav.index, nav[i], color=color, linewidth=2, 
                       marker='o', markersize=3, label=label, alpha=0.8)
        
        # 设置图表属性
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('净值', fontsize=12)
        ax.set_title(f"分组净值曲线 - {self.factor.name if self.factor else 'Factor'}，{n_groups}组", 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 设置x轴日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        if len(nav) > 50:
            interval = max(1, len(nav) // 20)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        else:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save and self.factor is not None:
            save_path = f"E:/data/回测/{self.factor.name}_group_navs.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        # 显示图片
        if show_plot:
            plt.show()
        
        return fig, ax 