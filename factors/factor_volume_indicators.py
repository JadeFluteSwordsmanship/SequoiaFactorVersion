import numpy as np
import pandas as pd
from .factor_base import FactorBase

class Alpha097(FactorBase):
    name = "Alpha097"
    direction = 1  # 成交量波动率大，可能表示活跃度提高，未来收益可能较高
    description = (
        "Alpha097：成交量波动率因子。\n"
        "公式：Alpha097 = STD(VOLUME, 10)\n"
        "计算过程：计算过去10日成交量的标准差。\n"
        "解读：\n"
        "  - 成交量波动率大：表示成交量变化剧烈，市场活跃度高；\n"
        "  - 成交量波动率小：表示成交量相对稳定，市场相对平静；\n"
        "  - 方向（direction=1）：假设成交量波动率大时未来收益较高。"
    )
    data_requirements = {
        'daily': {'window': 12}  # 10日标准差 + 少量冗余
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权成交量
        vol_col = 'adj_vol' if 'adj_vol' in df.columns else 'vol'

        # 计算10日成交量标准差
        df['alpha097'] = df.groupby('stock_code')[vol_col].transform(
            lambda x: x.rolling(10, min_periods=10).std()
        )

        # 输出结果
        result = df[['stock_code', 'trade_date', 'alpha097']].dropna(subset=['alpha097']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'alpha097': 'value'
        })
        return result.reset_index(drop=True)


class Alpha095(FactorBase):
    name = "Alpha095"
    direction = 1  # 成交金额波动率大，可能表示资金活跃度提高，未来收益可能较高
    description = (
        "Alpha095：成交金额波动率因子。\n"
        "公式：Alpha095 = STD(AMOUNT, 20)\n"
        "计算过程：计算过去20日成交金额的标准差。\n"
        "解读：\n"
        "  - 成交金额波动率大：表示资金流入流出变化剧烈，市场资金活跃度高；\n"
        "  - 成交金额波动率小：表示资金流动相对稳定；\n"
        "  - 方向（direction=1）：假设成交金额波动率大时未来收益较高。"
    )
    data_requirements = {
        'daily': {'window': 22}  # 20日标准差 + 少量冗余
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 直接使用daily数据中的amount字段
        df['alpha095'] = df.groupby('stock_code')['amount'].transform(
            lambda x: x.rolling(20, min_periods=20).std()
        )

        # 输出结果
        result = df[['stock_code', 'trade_date', 'alpha095']].dropna(subset=['alpha095']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'alpha095': 'value'
        })
        return result.reset_index(drop=True)


class Alpha100(FactorBase):
    name = "Alpha100"
    direction = 1  # 成交量波动率大，可能表示活跃度提高，未来收益可能较高
    description = (
        "Alpha100：成交量波动率因子（长期）。\n"
        "公式：Alpha100 = STD(VOLUME, 20)\n"
        "计算过程：计算过去20日成交量的标准差。\n"
        "解读：\n"
        "  - 与Alpha097类似，但使用20日窗口，更关注中长期成交量波动；\n"
        "  - 成交量波动率大：表示成交量变化剧烈，市场活跃度高；\n"
        "  - 成交量波动率小：表示成交量相对稳定，市场相对平静；\n"
        "  - 方向（direction=1）：假设成交量波动率大时未来收益较高。"
    )
    data_requirements = {
        'daily': {'window': 22}  # 20日标准差 + 少量冗余
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权成交量
        vol_col = 'adj_vol' if 'adj_vol' in df.columns else 'vol'

        # 计算20日成交量标准差
        df['alpha100'] = df.groupby('stock_code')[vol_col].transform(
            lambda x: x.rolling(20, min_periods=20).std()
        )

        # 输出结果
        result = df[['stock_code', 'trade_date', 'alpha100']].dropna(subset=['alpha100']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'alpha100': 'value'
        })
        return result.reset_index(drop=True)


class Alpha081(FactorBase):
    name = "Alpha081"
    direction = 1  # 成交量移动平均大，可能表示活跃度提高，未来收益可能较高
    description = (
        "Alpha081：成交量移动平均因子。\n"
        "公式：Alpha081 = SMA(VOLUME, 21, 2)\n"
        "计算过程：计算过去21日成交量的简单移动平均，权重为2。\n"
        "解读：\n"
        "  - 成交量移动平均大：表示近期成交量水平较高，市场活跃度好；\n"
        "  - 成交量移动平均小：表示近期成交量水平较低，市场相对冷清；\n"
        "  - 方向（direction=1）：假设成交量移动平均大时未来收益较高。"
    )
    data_requirements = {
        'daily': {'window': 23}  # 21日移动平均 + 少量冗余
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权成交量
        vol_col = 'adj_vol' if 'adj_vol' in df.columns else 'vol'

        # 计算21日成交量简单移动平均
        df['alpha081'] = df.groupby('stock_code')[vol_col].transform(
            lambda x: x.rolling(21, min_periods=21).mean()
        )

        # 输出结果
        result = df[['stock_code', 'trade_date', 'alpha081']].dropna(subset=['alpha081']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'alpha081': 'value'
        })
        return result.reset_index(drop=True)


class Alpha132(FactorBase):
    name = "Alpha132"
    direction = 1  # 成交金额移动平均大，可能表示资金活跃度提高，未来收益可能较高
    description = (
        "Alpha132：成交金额移动平均因子。\n"
        "公式：Alpha132 = MEAN(AMOUNT, 20)\n"
        "计算过程：计算过去20日成交金额的简单移动平均。\n"
        "解读：\n"
        "  - 成交金额移动平均大：表示近期资金流入水平较高，市场活跃度好；\n"
        "  - 成交金额移动平均小：表示近期资金流入水平较低，市场相对冷清；\n"
        "  - 方向（direction=1）：假设成交金额移动平均大时未来收益较高。"
    )
    data_requirements = {
        'daily': {'window': 22}  # 20日移动平均 + 少量冗余
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 直接使用daily数据中的amount字段
        df['alpha132'] = df.groupby('stock_code')['amount'].transform(
            lambda x: x.rolling(20, min_periods=20).mean()
        )

        # 输出结果
        result = df[['stock_code', 'trade_date', 'alpha132']].dropna(subset=['alpha132']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'alpha132': 'value'
        })
        return result.reset_index(drop=True) 