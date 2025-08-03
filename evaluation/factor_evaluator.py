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
     
    def calc_daily_ir(self, top_n: int = 5) -> pd.DataFrame:
        """
        每期计算 top_n 组合的 return_mean / return_std（可理解为 Long-Only IR）
        返回: DataFrame，包含 ['date', 'IR']
        """
        if self.merged is None:
            raise ValueError("未能合并因子值和收益率数据")
        
        # 创建缓存键
        cache_key = top_n
        
        # 检查缓存
        if cache_key in self._daily_ir_cache:
            return self._daily_ir_cache[cache_key]
        
        start_time = time.time()
        daily_irs = []
        for date, group in self.merged.groupby('date'):
            top_group = group.nlargest(top_n, 'value')
            if len(top_group) < 2:
                continue
            mean_r = top_group['future_return'].mean()
            std_r = top_group['future_return'].std()
            ir = mean_r / std_r if std_r != 0 else np.nan
            daily_irs.append({'date': date, 'IR': ir})
        
        result = pd.DataFrame(daily_irs).set_index('date').sort_index()
        
        # 缓存结果
        self._daily_ir_cache[cache_key] = result
        
        elapsed_time = time.time() - start_time
        print(f"calc_daily_ir 计算完成，耗时: {elapsed_time:.4f} 秒")
        
        return result
     
    def clear_cache(self):
        """
        清除所有缓存的计算结果，强制重新计算。
        在数据更新后调用此方法。
        """
        self._ic_cache.clear()
        self._ir_cache.clear()
        self._sharpe_long_only_cache.clear()
        self._daily_ir_cache.clear()

    def get_cache_info(self):
        """
        获取缓存信息，用于调试和监控。
        返回: dict，包含各缓存的键数量
        """
        return {
            'ic_cache_keys': list(self._ic_cache.keys()),
            'ir_cache_keys': list(self._ir_cache.keys()),
            'sharpe_cache_keys': list(self._sharpe_long_only_cache.keys()),
            'daily_ir_cache_keys': list(self._daily_ir_cache.keys())
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

    # 可扩展更多评估函数，如分组回测、回撤、净值曲线、回归等 
     
    def plot_ic_chart(self, ic_type: str = 'IC', figsize: tuple = (12, 8), 
                    show_plot: bool = True, save_path: Optional[str] = None):
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
            title = "每期 IC 图"
        elif ic_type.upper() == 'RANKIC':
            ic_df = self.calc_rank_ic()
            title = "每期 RankIC 图"
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
        
        # 设置x轴日期格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
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
        
        # 调整布局
        plt.tight_layout()
        
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
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        # 显示图片
        if show_plot:
            plt.show()
        
        return fig, (ax1, ax2)
    
    def plot_daily_ir_chart(self, top_n: int = 5, figsize: tuple = (12, 8), 
                        show_plot: bool = True, save_path: Optional[str] = None):
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
        
        # 设置x轴日期格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
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
        plt.title(f"每期 IR 图 (Top {top_n})", fontsize=14, fontweight='bold', pad=20)
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                bbox_to_anchor=(0.01, 0.99))
        
        # 调整布局
        plt.tight_layout()
        
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
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        # 显示图片
        if show_plot:
            plt.show()
        
        return fig, (ax1, ax2)

    # 可扩展更多评估函数，如分组回测、回撤、净值曲线、回归等 