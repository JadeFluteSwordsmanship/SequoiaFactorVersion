import numpy as np
import pandas as pd
from .factor_base import FactorBase
import talib

class Alpha004(FactorBase):
    name = "Alpha004"
    direction = 1  # 因子值与未来收益正相关：因子值大→未来收益大，因子值小→未来收益小
    description = (
        "Alpha004：区间突破 + 区间内量能过滤 因子。\n"
        "公式：(((SUM(CLOSE,8)/8)+STD(CLOSE,8)) < (SUM(CLOSE,2)/2)) ? -1 "
        ": ( ((SUM(CLOSE,2)/2) < ((SUM(CLOSE,8)/8)-STD(CLOSE,8))) ? 1 "
        ": ( (VOLUME/MEAN(VOLUME,20) >= 1) ? 1 : -1 ) )\n"
        "逻辑：\n"
        "  1) 2日均价上穿 (8日均价 + 8日标准差) 视为向上突破 → 信号 -1（高位反转预期）。\n"
        "  2) 2日均价下破 (8日均价 - 8日标准差) 视为向下突破 → 信号 +1（低位反弹预期）。\n"
        "  3) 否则仍在通道内：若当前成交量 ≥ 20日均量 → +1；否则 -1。\n"
        "解读：向上突破时因子值为-1，表示高位反转预期（未来收益小）；向下突破时因子值为+1，表示低位反弹预期（未来收益大）。\n"
        "方向（direction=1 说明）：因子值与未来收益正相关，因子值大→未来收益大，因子值小→未来收益小。"
    )
    data_requirements = {
        'daily': {'window': 21}  # 满足 20 日均量、8 日带、2 日均价
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权价格（如果已在上游生成），否则使用原始 close
        close_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        vol_col = 'adj_vol' if 'adj_vol' in df.columns else 'vol'
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            close = g[close_col].to_numpy(dtype=np.float64)
            vol = g[vol_col].to_numpy(dtype=np.float64)
            # TA-Lib实现
            ma8 = talib.SMA(close, timeperiod=8)
            std8 = talib.STDDEV(close, timeperiod=8, nbdev=0)  # ddof=0
            ma2 = talib.SMA(close, timeperiod=2)
            vol_ma20 = talib.SMA(vol, timeperiod=20)
            # 上下轨
            upper = ma8 + std8
            lower = ma8 - std8
            # 量能比
            vol_ratio = vol / vol_ma20
            # 条件逻辑
            cond_up_break = ma2 > upper
            cond_down_break = ma2 < lower
            cond_vol_ok = vol_ratio >= 1
            # 计算因子值
            value = np.full(len(close), np.nan)
            valid_mask = ~(np.isnan(ma8) | np.isnan(std8) | np.isnan(ma2) | np.isnan(vol_ma20))
            # 上行突破
            value[valid_mask & cond_up_break] = -1
            # 下行突破
            value[valid_mask & ~cond_up_break & cond_down_break] = 1
            # 区间内：看量
            mask_inside = ~(cond_up_break | cond_down_break)
            value[valid_mask & mask_inside & cond_vol_ok] = 1
            value[valid_mask & mask_inside & ~cond_vol_ok] = -1
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': value
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res

class Alpha004_Enhanced(FactorBase):
    name = "Alpha004_Enhanced"
    direction = 1  # 因子值与未来收益正相关：因子值大→未来收益大，因子值小→未来收益小
    description = (
        "Alpha004_Enhanced：Alpha004的连续化改进版本，将离散的-1/+1信号改为连续因子值。\n"
        "设计思路：\n"
        "  1) 突破信号：根据突破程度计算连续值，突破越强信号越强\n"
        "  2) 区间内信号：结合量能比和在区间内的相对位置（距离中轨的远近）\n"
        "  3) 目标：因子值连续，因子值大→未来收益高，因子值小→未来收益低\n"
        "计算逻辑：\n"
        "  - 向上突破：根据突破幅度计算负值（-1到-0.1），突破越强负值越大\n"
        "  - 向下突破：根据突破幅度计算正值（0.1到1），突破越强正值越大\n"
        "  - 区间内：根据量能比和相对位置计算连续值（-0.5到0.5）\n"
    )
    data_requirements = {
        'daily': {'window': 21}  # 满足 20 日均量、8 日带、2 日均价
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权价格（如果已在上游生成），否则使用原始 close
        close_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        vol_col = 'adj_vol' if 'adj_vol' in df.columns else 'vol'
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            close = g[close_col].to_numpy(dtype=np.float64)
            vol = g[vol_col].to_numpy(dtype=np.float64)
            # TA-Lib实现
            ma8 = talib.SMA(close, timeperiod=8)
            std8 = talib.STDDEV(close, timeperiod=8, nbdev=0)  # ddof=0
            ma2 = talib.SMA(close, timeperiod=2)
            vol_ma20 = talib.SMA(vol, timeperiod=20)
            # 上下轨和中轨
            upper = ma8 + std8
            lower = ma8 - std8
            middle = ma8
            # 量能比
            vol_ratio = vol / vol_ma20
            # 计算因子值
            value = np.full(len(close), np.nan)
            valid_mask = ~(np.isnan(ma8) | np.isnan(std8) | np.isnan(ma2) | np.isnan(vol_ma20))
            # 向上突破：计算突破幅度，转换为负值
            up_break_mask = valid_mask & (ma2 > upper)
            if up_break_mask.any():
                break_ratio = (ma2[up_break_mask] - upper[up_break_mask]) / std8[up_break_mask]
                factor_value = -0.1 - 0.9 * np.tanh(break_ratio)
                value[up_break_mask] = factor_value
            # 向下突破：计算突破幅度，转换为正值
            down_break_mask = valid_mask & (ma2 < lower)
            if down_break_mask.any():
                break_ratio = (lower[down_break_mask] - ma2[down_break_mask]) / std8[down_break_mask]
                factor_value = 0.1 + 0.9 * np.tanh(break_ratio)
                value[down_break_mask] = factor_value
            # 区间内：结合量能比和相对位置
            inside_mask = valid_mask & ~(up_break_mask | down_break_mask)
            if inside_mask.any():
                relative_pos = (ma2[inside_mask] - middle[inside_mask]) / std8[inside_mask]
                vol_ratio_clipped = np.clip(vol_ratio[inside_mask], 0.1, 3.0)
                position_weight = -relative_pos * 0.3
                volume_weight = (vol_ratio_clipped - 1.0) * 0.2
                factor_value = position_weight + volume_weight
                factor_value = np.clip(factor_value, -0.5, 0.5)
                value[inside_mask] = factor_value
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': value
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res



