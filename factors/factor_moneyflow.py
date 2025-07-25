import pandas as pd
from .factor_base import FactorBase

class Custom010(FactorBase):
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
        df['custom010'] = df['net_mf_amount'] / df['total_amount']
        result = df[['stock_code', 'trade_date', 'custom010']].dropna(subset=['custom010']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'custom010': 'value'
        })
        return result.reset_index(drop=True)

class Custom011(FactorBase):
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
        df['custom011'] = (buy_lg - sell_lg) / df['total_amount']
        result = df[['stock_code', 'trade_date', 'custom011']].dropna(subset=['custom011']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'custom011': 'value'
        })
        return result.reset_index(drop=True)

class Custom012(FactorBase):
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
        df['custom012'] = (df['buy_sm_amount'] - df['sell_sm_amount']) / df['total_amount']
        result = df[['stock_code', 'trade_date', 'custom012']].dropna(subset=['custom012']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'custom012': 'value'
        })
        return result.reset_index(drop=True)

class Custom013(FactorBase):
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
        df['custom013'] = (df['sell_lg_amount'] + df['sell_elg_amount']) / df['total_amount']
        result = df[['stock_code', 'trade_date', 'custom013']].dropna(subset=['custom013']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'custom013': 'value'
        })
        return result.reset_index(drop=True) 