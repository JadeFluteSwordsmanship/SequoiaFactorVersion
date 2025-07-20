import numpy as np
import pandas as pd
from .factor_base import FactorBase

class Custom003(FactorBase):
    name = "Custom003"
    direction = -1  # 高频收益方差大，表示日内波动剧烈，未来收益可能较低
    description = (
        "Custom003：高频收益方差因子（Realized Variance）。\n"
        "公式：RVar_i = Σ(r_ij^2)，其中r_ij为第i天第j分钟的收益率\n"
        "计算过程：\n"
        "1. 对每只股票的每分钟数据计算收益率；\n"
        "2. 按股票和日期分组，计算当日所有分钟收益率的方差；\n"
        "3. 将方差作为因子值。\n"
        "解读：\n"
        "  - 高频收益方差大：表示日内价格波动剧烈，市场不稳定；\n"
        "  - 高频收益方差小：表示日内价格相对稳定，市场平稳；\n"
        "  - 方向（direction=-1）：假设日内波动剧烈时未来收益较低。"
    )
    data_requirements = {
        'minute': {'window': 241}  # 9:30-15:00，共241个分钟数据，pct_change后有效数据240个
    }

    def _compute_impl(self, data):
        df = data['minute'].copy()
        df = df.sort_values(['stock_code', 'datetime'])

        # 提取日期部分，用于分组
        df['trade_date'] = pd.to_datetime(df['datetime']).dt.date

        # 按股票和日期分组，计算每分钟收益率
        df['return'] = df.groupby(['stock_code', 'trade_date'])['close'].pct_change()

        # 按股票和日期分组，计算当日收益率方差，并去重
        result = df.groupby(['stock_code', 'trade_date'])['return'].var().reset_index()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'return': 'value'
        })
        
        # 放大方差值，避免数值过小损失精度
        result['value'] = result['value'] * 100000
        
        # 过滤掉NaN值
        result = result.dropna(subset=['value'])
        
        return result.reset_index(drop=True)


class Custom004(FactorBase):
    name = "Custom004"
    direction = 1  # 偏度为正表示右偏（正收益更多），可能表示上涨趋势，未来收益较高
    description = (
        "Custom004：高频收益偏度因子（Realized Skewness）。\n"
        "公式：RSkew_i = (sqrt(N) * Σ(r_ij^3)) / RVar_i^(3/2)\n"
        "其中：\n"
        "  r_ij 为第i天第j分钟的收益率\n"
        "  N 为当日分钟数（通常为241）\n"
        "  RVar_i 为第i天的已实现方差\n"
        "计算过程：\n"
        "1. 计算每分钟收益率的三次方；\n"
        "2. 按股票和日期分组，计算当日所有分钟收益率三次方的和；\n"
        "3. 乘以sqrt(N)并除以方差^(3/2)，得到偏度。\n"
        "解读：\n"
        "  - 偏度 > 0：右偏，正收益更多，可能表示上涨趋势；\n"
        "  - 偏度 < 0：左偏，负收益更多，可能表示下跌趋势；\n"
        "  - 方向（direction=1）：假设右偏时（正偏度）未来收益较高。"
    )
    data_requirements = {
        'minute': {'window': 241}  # 9:30-15:00，共241个分钟数据，pct_change后有效数据240个
    }

    def _compute_impl(self, data):
        df = data['minute'].copy()
        df = df.sort_values(['stock_code', 'datetime'])

        # 提取日期部分，用于分组
        df['trade_date'] = pd.to_datetime(df['datetime']).dt.date

        # 按股票和日期分组，计算每分钟收益率
        df['return'] = df.groupby(['stock_code', 'trade_date'])['close'].pct_change()

        # 计算收益率的三次方
        df['return_cubed'] = df['return'] ** 3

        # 按股票和日期分组，计算偏度
        # 先计算方差
        variance = df.groupby(['stock_code', 'trade_date'])['return'].var()
        
        # 计算收益率三次方的和
        sum_cubed = df.groupby(['stock_code', 'trade_date'])['return_cubed'].sum()
        
        # 计算偏度：sqrt(N) * sum(r^3) / var^(3/2)
        N = 240  # 当日有效分钟数（pct_change后第一行是NaN）
        skewness = (np.sqrt(N) * sum_cubed) / (variance ** (3/2))
        
        # 转换为DataFrame
        result = skewness.reset_index()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            0: 'value'
        })
        
        # 过滤掉NaN值
        result = result.dropna(subset=['value'])
        
        return result.reset_index(drop=True) 
    

class Custom005(FactorBase):
    name = "Custom005"
    direction = -1  # 峰度高表示分布尾部厚重，极端事件概率大，未来收益可能较低
    description = (
        "Custom005：高频收益峰度因子（Realized Kurtosis）。\n"
        "公式：RKurt_i = (N * Σ(r_ij^4)) / RVar_i^2\n"
        "其中：\n"
        "  r_ij 为第i天第j分钟的收益率\n"
        "  N 为当日有效分钟数（240）\n"
        "  RVar_i 为第i天的已实现方差\n"
        "计算过程：\n"
        "1. 计算每分钟收益率的四次方；\n"
        "2. 按股票和日期分组，计算当日所有分钟收益率四次方的和；\n"
        "3. 乘以N并除以方差^2，得到峰度。\n"
        "解读：\n"
        "  - 峰度 > 3：超峰态，分布尾部厚重，极端事件概率大；\n"
        "  - 峰度 < 3：低峰态，分布尾部较薄，极端事件概率小；\n"
        "  - 方向（direction=-1）：假设峰度高时（尾部厚重）未来收益较低。"
    )
    data_requirements = {
        'minute': {'window': 241}  # 9:30-15:00，共241个分钟数据，pct_change后有效数据240个
    }

    def _compute_impl(self, data):
        df = data['minute'].copy()
        df = df.sort_values(['stock_code', 'datetime'])

        # 提取日期部分，用于分组
        df['trade_date'] = pd.to_datetime(df['datetime']).dt.date

        # 按股票和日期分组，计算每分钟收益率
        df['return'] = df.groupby(['stock_code', 'trade_date'])['close'].pct_change()

        # 计算收益率的四次方
        df['return_quartic'] = df['return'] ** 4

        # 按股票和日期分组，计算峰度
        # 先计算方差
        variance = df.groupby(['stock_code', 'trade_date'])['return'].var()
        
        # 计算收益率四次方的和
        sum_quartic = df.groupby(['stock_code', 'trade_date'])['return_quartic'].sum()
        
        # 计算峰度：N * sum(r^4) / var^2
        N = 240  # 当日有效分钟数（pct_change后第一行是NaN）
        kurtosis = (N * sum_quartic) / (variance ** 2)
        
        # 转换为DataFrame
        result = kurtosis.reset_index()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            0: 'value'
        })
        
        # 过滤掉NaN值
        result = result.dropna(subset=['value'])
        
        return result.reset_index(drop=True) 
    

class Custom006(FactorBase):
    name = "Custom006"
    direction = 1  # 上行波动大表示上涨动能强，未来收益可能较高
    description = (
        "Custom006：高频上行波动因子。\n"
        "公式：上行波动 = (Σ(r_t * I_{r_t > 0})^2)^(1/2)\n"
        "其中：\n"
        "  r_t 为第t分钟的收益率\n"
        "  I_{r_t > 0} 为指示函数，当r_t > 0时为1，否则为0\n"
        "计算过程：\n"
        "1. 筛选出所有正收益率；\n"
        "2. 对正收益率求平方；\n"
        "3. 求和后开根号，得到上行波动。\n"
        "解读：\n"
        "  - 上行波动大：表示上涨动能强，正收益贡献大；\n"
        "  - 上行波动小：表示上涨动能弱，正收益贡献小；\n"
        "  - 方向（direction=1）：假设上行波动大时未来收益较高。"
    )
    data_requirements = {
        'minute': {'window': 241}  # 9:30-15:00，共241个分钟数据，pct_change后有效数据240个
    }

    def _compute_impl(self, data):
        df = data['minute'].copy()
        df = df.sort_values(['stock_code', 'datetime'])

        # 提取日期部分，用于分组
        df['trade_date'] = pd.to_datetime(df['datetime']).dt.date

        # 按股票和日期分组，计算每分钟收益率
        df['return'] = df.groupby(['stock_code', 'trade_date'])['close'].pct_change()

        # 计算上行波动：sqrt(sum(positive_returns^2))
        def calc_upward_volatility(returns):
            positive_returns = returns[returns > 0]
            if len(positive_returns) == 0:
                return 0
            return np.sqrt(np.sum(positive_returns ** 2))
        
        # 按股票和日期分组，计算上行波动
        upward_vol = df.groupby(['stock_code', 'trade_date'])['return'].apply(calc_upward_volatility)
        
        # 转换为DataFrame
        result = upward_vol.reset_index()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'return': 'value'
        })
        
        # 放大波动值，避免数值过小损失精度
        result['value'] = result['value'] * 100000
        
        # 过滤掉NaN值
        result = result.dropna(subset=['value'])
        
        return result.reset_index(drop=True)


class Custom007(FactorBase):
    name = "Custom007"
    direction = -1  # 下行波动大表示下跌动能强，未来收益可能较低
    description = (
        "Custom007：高频下行波动因子。\n"
        "公式：下行波动 = (Σ(r_t * I_{r_t < 0})^2)^(1/2)\n"
        "其中：\n"
        "  r_t 为第t分钟的收益率\n"
        "  I_{r_t < 0} 为指示函数，当r_t < 0时为1，否则为0\n"
        "计算过程：\n"
        "1. 筛选出所有负收益率；\n"
        "2. 对负收益率求平方；\n"
        "3. 求和后开根号，得到下行波动。\n"
        "解读：\n"
        "  - 下行波动大：表示下跌动能强，负收益贡献大；\n"
        "  - 下行波动小：表示下跌动能弱，负收益贡献小；\n"
        "  - 方向（direction=-1）：假设下行波动大时未来收益较低。"
    )
    data_requirements = {
        'minute': {'window': 241}  # 9:30-15:00，共241个分钟数据，pct_change后有效数据240个
    }

    def _compute_impl(self, data):
        df = data['minute'].copy()
        df = df.sort_values(['stock_code', 'datetime'])

        # 提取日期部分，用于分组
        df['trade_date'] = pd.to_datetime(df['datetime']).dt.date

        # 按股票和日期分组，计算每分钟收益率
        df['return'] = df.groupby(['stock_code', 'trade_date'])['close'].pct_change()

        # 计算下行波动：sqrt(sum(negative_returns^2))
        def calc_downward_volatility(returns):
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return 0
            return np.sqrt(np.sum(negative_returns ** 2))
        
        # 按股票和日期分组，计算下行波动
        downward_vol = df.groupby(['stock_code', 'trade_date'])['return'].apply(calc_downward_volatility)
        
        # 转换为DataFrame
        result = downward_vol.reset_index()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'return': 'value'
        })
        
        # 放大波动值，避免数值过小损失精度
        result['value'] = result['value'] * 100000
        
        # 过滤掉NaN值
        result = result.dropna(subset=['value'])
        
        return result.reset_index(drop=True) 
    

class Custom008(FactorBase):
    name = "Custom008"
    direction = 1  # 上行波动占比大，通常代表上涨动能主导
    description = (
        "Custom008：高频上行波动占比因子。\n"
        "公式：上行波动占比 = 上行波动 / 总波动\n"
        "其中：\n"
        "  上行波动 = (Σ(r_t * I_{r_t > 0})^2)^(1/2)\n"
        "  总波动 = (Σ(r_t^2))^(1/2)\n"
        "解读：\n"
        "  - 上行波动占比大，表示正收益波动在总波动中占主导。\n"
        "  - 方向（direction=1）：假设上行波动占比大时未来收益较高。"
    )
    data_requirements = {
        'minute': {'window': 241}
    }

    def _compute_impl(self, data):
        df = data['minute'].copy()
        df = df.sort_values(['stock_code', 'datetime'])
        df['trade_date'] = pd.to_datetime(df['datetime']).dt.date
        df['return'] = df.groupby(['stock_code', 'trade_date'])['close'].pct_change()

        def calc_upward_ratio(returns):
            pos = returns[returns > 0]
            up = np.sqrt(np.sum(pos ** 2)) if len(pos) > 0 else 0
            total = np.sqrt(np.sum(returns ** 2)) if len(returns) > 0 else np.nan
            if total == 0 or np.isnan(total):
                return np.nan
            return up / total

        ratio = df.groupby(['stock_code', 'trade_date'])['return'].apply(calc_upward_ratio)
        result = ratio.reset_index()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'return': 'value'
        })
        result = result.dropna(subset=['value'])
        return result.reset_index(drop=True)


class Custom009(FactorBase):
    name = "Custom009"
    direction = -1  # 下行波动占比大，通常代表下跌动能主导
    description = (
        "Custom009：高频下行波动占比因子。\n"
        "公式：下行波动占比 = 下行波动 / 总波动\n"
        "其中：\n"
        "  下行波动 = (Σ(r_t * I_{r_t < 0})^2)^(1/2)\n"
        "  总波动 = (Σ(r_t^2))^(1/2)\n"
        "解读：\n"
        "  - 下行波动占比大，表示负收益波动在总波动中占主导。\n"
        "  - 方向（direction=-1）：假设下行波动占比大时未来收益较低。"
    )
    data_requirements = {
        'minute': {'window': 241}
    }

    def _compute_impl(self, data):
        df = data['minute'].copy()
        df = df.sort_values(['stock_code', 'datetime'])
        df['trade_date'] = pd.to_datetime(df['datetime']).dt.date
        df['return'] = df.groupby(['stock_code', 'trade_date'])['close'].pct_change()

        def calc_downward_ratio(returns):
            neg = returns[returns < 0]
            down = np.sqrt(np.sum(neg ** 2)) if len(neg) > 0 else 0
            total = np.sqrt(np.sum(returns ** 2)) if len(returns) > 0 else np.nan
            if total == 0 or np.isnan(total):
                return np.nan
            return down / total

        ratio = df.groupby(['stock_code', 'trade_date'])['return'].apply(calc_downward_ratio)
        result = ratio.reset_index()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'return': 'value'
        })
        result = result.dropna(subset=['value'])
        return result.reset_index(drop=True) 
    
