import numpy as np
import pandas as pd
from .factor_base import FactorBase
import talib
from .numba_utils import ts_rank_numba

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
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            vol = g[vol_col].to_numpy(dtype=np.float64)
            std_vol = talib.STDDEV(vol, timeperiod=10, nbdev=1)
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': std_vol
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res

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
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            amount = g['amount'].to_numpy(dtype=np.float64)
            std_amt = talib.STDDEV(amount, timeperiod=20, nbdev=1)
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': std_amt
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res

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
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            vol = g[vol_col].to_numpy(dtype=np.float64)
            std_vol = talib.STDDEV(vol, timeperiod=20, nbdev=1)
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': std_vol
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res

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
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            vol = g[vol_col].to_numpy(dtype=np.float64)
            # talib.SMA默认权重为1，和pandas mean一致
            sma_vol = talib.SMA(vol, timeperiod=21)
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': sma_vol
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res

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
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            amount = g['amount'].to_numpy(dtype=np.float64)
            mean_amt = talib.SMA(amount, timeperiod=20)
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': mean_amt
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res 