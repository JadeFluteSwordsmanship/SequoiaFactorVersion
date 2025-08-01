import pandas as pd
import numpy as np
from .factor_base import FactorBase
from .numba_utils import consecutive_moneyflow_strength_numba

class Custom010(FactorBase):
    """
    Custom010：资金流净额占比因子。
    公式：资金流净额占比 = net_mf_amount / (所有买卖金额之和)
    衡量主力/大单净流入在总成交中的占比，反映资金流向强度。
    方向（direction=1）：净流入占比越大，未来收益可能越高。
    """
    name = "Custom010"
    direction = 1  # 净流入占比大，未来收益可能高
    description = (
        "Custom010：资金流净额占比因子。\n"
        "公式：资金流净额占比 = net_mf_amount / (所有买卖金额之和)\n"
        "衡量主力/大单净流入在总成交中的占比，反映资金流向强度。\n"
        "方向（direction=1）：净流入占比越大，未来收益可能越高。"
    )
    data_requirements = {
        'moneyflow': {'window': 1}
    }

    def _compute_impl(self, data):
        df = data['moneyflow'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        buy_cols = ['buy_sm_amount', 'buy_md_amount', 'buy_lg_amount', 'buy_elg_amount']
        sell_cols = ['sell_sm_amount', 'sell_md_amount', 'sell_lg_amount', 'sell_elg_amount']
        df['total_amount'] = df[buy_cols + sell_cols].sum(axis=1)
        df['value'] = df['net_mf_amount'] / df['total_amount']
        df['factor'] = self.name
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)

class Custom011(FactorBase):
    """
    Custom011：大单净流入占比因子。
    公式：大单净流入占比 = (buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount) / total_amount
    衡量主力/游资净流入在总成交中的占比。
    方向（direction=1）：大单净流入占比越大，未来收益可能越高。
    """
    name = "Custom011"
    direction = 1  # 大单净流入占比大，未来收益可能高
    description = (
        "Custom011：大单净流入占比因子。\n"
        "公式：大单净流入占比 = (buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount) / total_amount\n"
        "衡量主力/游资净流入在总成交中的占比。\n"
        "方向（direction=1）：大单净流入占比越大，未来收益可能越高。"
    )
    data_requirements = {
        'moneyflow': {'window': 1}
    }

    def _compute_impl(self, data):
        df = data['moneyflow'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        buy_lg = df['buy_lg_amount'].fillna(0) + df['buy_elg_amount'].fillna(0)
        sell_lg = df['sell_lg_amount'].fillna(0) + df['sell_elg_amount'].fillna(0)
        buy_cols = ['buy_sm_amount', 'buy_md_amount', 'buy_lg_amount', 'buy_elg_amount']
        sell_cols = ['sell_sm_amount', 'sell_md_amount', 'sell_lg_amount', 'sell_elg_amount']
        df['total_amount'] = df[buy_cols + sell_cols].sum(axis=1)
        df['value'] = (buy_lg - sell_lg) / df['total_amount']
        df['factor'] = self.name
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)

class Custom012(FactorBase):
    """
    Custom012：小单净流入占比因子。
    公式：小单净流入占比 = (buy_sm_amount - sell_sm_amount) / total_amount
    衡量散户净流入在总成交中的占比。
    方向（direction=-1）：小单净流入占比越大，未来收益可能越低。
    """
    name = "Custom012"
    direction = -1  # 小单净流入占比大，未来收益可能低
    description = (
        "Custom012：小单净流入占比因子。\n"
        "公式：小单净流入占比 = (buy_sm_amount - sell_sm_amount) / total_amount\n"
        "衡量散户净流入在总成交中的占比。\n"
        "方向（direction=-1）：小单净流入占比越大，未来收益可能越低。"
    )
    data_requirements = {
        'moneyflow': {'window': 1}
    }

    def _compute_impl(self, data):
        df = data['moneyflow'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        buy_cols = ['buy_sm_amount', 'buy_md_amount', 'buy_lg_amount', 'buy_elg_amount']
        sell_cols = ['sell_sm_amount', 'sell_md_amount', 'sell_lg_amount', 'sell_elg_amount']
        df['total_amount'] = df[buy_cols + sell_cols].sum(axis=1)
        df['value'] = (df['buy_sm_amount'] - df['sell_sm_amount']) / df['total_amount']
        df['factor'] = self.name
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)

class Custom013(FactorBase):
    """
    Custom013：大单流出占比因子。
    公式：大单流出占比 = (sell_lg_amount + sell_elg_amount) / total_amount
    衡量主力/游资大幅撤离的程度。
    方向（direction=-1）：大单流出占比越大，未来收益可能越低。
    """
    name = "Custom013"
    direction = -1  # 大单流出占比大，未来收益可能低
    description = (
        "Custom013：大单流出占比因子。\n"
        "公式：大单流出占比 = (sell_lg_amount + sell_elg_amount) / total_amount\n"
        "衡量主力/游资大幅撤离的程度。\n"
        "方向（direction=-1）：大单流出占比越大，未来收益可能越低。"
    )
    data_requirements = {
        'moneyflow': {'window': 1}
    }

    def _compute_impl(self, data):
        df = data['moneyflow'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        buy_cols = ['buy_sm_amount', 'buy_md_amount', 'buy_lg_amount', 'buy_elg_amount']
        sell_cols = ['sell_sm_amount', 'sell_md_amount', 'sell_lg_amount', 'sell_elg_amount']
        df['total_amount'] = df[buy_cols + sell_cols].sum(axis=1)
        df['value'] = (df['sell_lg_amount'] + df['sell_elg_amount']) / df['total_amount']
        df['factor'] = self.name
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)

class Custom100(FactorBase):
    """
    Custom100：连续资金流入强度因子。
    公式：连续资金流入强度 = Σ(连续N天的大单净流入占比 * 权重衰减)
    权重衰减：最近一天权重1.0，前一天0.8，前两天0.6，以此类推
    
    设计思路：
    1. 连续多天的大单净流入表明主力在建仓，而非短期投机
    2. 权重衰减确保最近的资金流更重要
    3. 只考虑净流入（正值），忽略流出情况
    4. 方向（direction=1）：连续流入强度越大，未来收益可能越高
    
    参数：
    - window: 回看天数，默认5天
    - min_consecutive_days: 最少连续流入天数，默认3天
    """
    name = "Custom100"
    direction = 1  # 连续流入强度大，未来收益可能高
    description = (
        "Custom100：连续资金流入强度因子。\n"
        "公式：连续资金流入强度 = Σ(连续N天的大单净流入占比 * 权重衰减)\n"
        "权重衰减：从1.0线性衰减到0.2，最近的数据权重最大\n"
        "设计思路：连续多天的大单净流入表明主力在建仓，而非短期投机。\n"
        "方向（direction=1）：连续流入强度越大，未来收益可能越高。"
    )
    data_requirements = {
        'moneyflow': {'window': 8}  # 需要5天的数据来计算连续流入
    }

    def _compute_impl(self, data):
        df = data['moneyflow'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        
        # 预分配结果列表，避免频繁的DataFrame创建
        all_codes = []
        all_dates = []
        all_values = []
        
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            
            # 计算大单净流入占比
            buy_lg = g['buy_lg_amount'].fillna(0) + g['buy_elg_amount'].fillna(0)
            sell_lg = g['sell_lg_amount'].fillna(0) + g['sell_elg_amount'].fillna(0)
            buy_cols = ['buy_sm_amount', 'buy_md_amount', 'buy_lg_amount', 'buy_elg_amount']
            sell_cols = ['sell_sm_amount', 'sell_md_amount', 'sell_lg_amount', 'sell_elg_amount']
            total_amount = g[buy_cols + sell_cols].sum(axis=1)
            lg_net_ratios = (buy_lg - sell_lg) / total_amount
            
            # 转换为numpy数组并处理NaN
            ratios_array = lg_net_ratios.to_numpy(dtype=np.float64)
            dates = g['trade_date'].values
            
            # 使用numba优化的连续流入强度计算
            consecutive_strength = consecutive_moneyflow_strength_numba(ratios_array, window=7, min_consecutive=3)
            
            # 只保留非NaN的值
            valid_mask = ~np.isnan(consecutive_strength)
            if valid_mask.any():
                all_codes.extend([code] * valid_mask.sum())
                all_dates.extend(dates[valid_mask])
                all_values.extend(consecutive_strength[valid_mask])
        
        # 一次性创建DataFrame
        res = pd.DataFrame({
            'code': all_codes,
            'date': all_dates,
            'factor': self.name,
            'value': all_values
        })
        
        return res.reset_index(drop=True) 
    
