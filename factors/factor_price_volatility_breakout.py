import numpy as np
import pandas as pd
from .factor_base import FactorBase

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
        vol_col   = 'adj_vol'   if 'adj_vol'   in df.columns else 'vol'

        # --- 价格滚动统计 ---
        # 8日均价 & 8日标准差
        df['ma8']  = df.groupby('stock_code')[close_col].transform(lambda x: x.rolling(8,  min_periods=8).mean())
        df['std8'] = df.groupby('stock_code')[close_col].transform(lambda x: x.rolling(8,  min_periods=8).std(ddof=0))
        # 2日均价
        df['ma2']  = df.groupby('stock_code')[close_col].transform(lambda x: x.rolling(2,  min_periods=2).mean())

        # --- 成交量均值 ---
        df['vol_ma20'] = df.groupby('stock_code')[vol_col].transform(lambda x: x.rolling(20, min_periods=20).mean())

        # --- 上下轨 ---
        df['upper'] = df['ma8'] + df['std8']
        df['lower'] = df['ma8'] - df['std8']

        # --- 量能比 ---
        df['vol_ratio'] = df[vol_col] / df['vol_ma20']

        # --- 条件逻辑（完全等价于原公式的嵌套三元） ---
        cond_up_break   = df['ma2'] > df['upper']
        cond_down_break = df['ma2'] < df['lower']
        cond_vol_ok     = df['vol_ratio'] >= 1  # 包含 ==1
        
        # 先置 NaN
        df['alpha004'] = np.nan
        
        # 明确处理：只有所有滚动窗口数据都完整时才计算因子值
        # 检查哪些行有完整的滚动窗口数据
        valid_data = (
            df['ma8'].notna() & 
            df['std8'].notna() & 
            df['ma2'].notna() & 
            df['vol_ma20'].notna() & 
            df['vol_ratio'].notna()
        )
        
        # 上行突破
        df.loc[valid_data & cond_up_break, 'alpha004'] = -1
        
        # 下行突破
        df.loc[valid_data & ~cond_up_break & cond_down_break, 'alpha004'] = 1
        
        # 区间内：看量
        mask_inside = ~(cond_up_break | cond_down_break)
        df.loc[valid_data & mask_inside & cond_vol_ok,  'alpha004'] = 1
        df.loc[valid_data & mask_inside & ~cond_vol_ok, 'alpha004'] = -1
        
        # 输出结果，风格与前面一致
        result = df[['stock_code', 'trade_date', 'alpha004']].dropna(subset=['alpha004']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'alpha004': 'value'
        })
        return result.reset_index(drop=True)


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
        vol_col   = 'adj_vol'   if 'adj_vol'   in df.columns else 'vol'

        # --- 价格滚动统计 ---
        # 8日均价 & 8日标准差
        df['ma8']  = df.groupby('stock_code')[close_col].transform(lambda x: x.rolling(8,  min_periods=8).mean())
        df['std8'] = df.groupby('stock_code')[close_col].transform(lambda x: x.rolling(8,  min_periods=8).std(ddof=0))
        # 2日均价
        df['ma2']  = df.groupby('stock_code')[close_col].transform(lambda x: x.rolling(2,  min_periods=2).mean())

        # --- 成交量均值 ---
        df['vol_ma20'] = df.groupby('stock_code')[vol_col].transform(lambda x: x.rolling(20, min_periods=20).mean())

        # --- 上下轨和中轨 ---
        df['upper'] = df['ma8'] + df['std8']
        df['lower'] = df['ma8'] - df['std8']
        df['middle'] = df['ma8']  # 中轨就是8日均价

        # --- 量能比 ---
        df['vol_ratio'] = df[vol_col] / df['vol_ma20']

        # 先置 NaN
        df['alpha004_enhanced'] = np.nan
        
        # 明确处理：只有所有滚动窗口数据都完整时才计算因子值
        valid_data = (
            df['ma8'].notna() & 
            df['std8'].notna() & 
            df['ma2'].notna() & 
            df['vol_ma20'].notna() & 
            df['vol_ratio'].notna()
        )
        
        # --- 向上突破：计算突破幅度，转换为负值 ---
        up_break_mask = valid_data & (df['ma2'] > df['upper'])
        if up_break_mask.any():
            # 突破幅度：距离上轨的距离占标准差的百分比
            break_ratio = (df.loc[up_break_mask, 'ma2'] - df.loc[up_break_mask, 'upper']) / df.loc[up_break_mask, 'std8']
            # 将突破幅度映射到 -1 到 -0.1 的区间，突破越强负值越大
            factor_value = -0.1 - 0.9 * np.tanh(break_ratio)  # tanh确保在合理范围内
            df.loc[up_break_mask, 'alpha004_enhanced'] = factor_value
        
        # --- 向下突破：计算突破幅度，转换为正值 ---
        down_break_mask = valid_data & (df['ma2'] < df['lower'])
        if down_break_mask.any():
            # 突破幅度：距离下轨的距离占标准差的百分比
            break_ratio = (df.loc[down_break_mask, 'lower'] - df.loc[down_break_mask, 'ma2']) / df.loc[down_break_mask, 'std8']
            # 将突破幅度映射到 0.1 到 1 的区间，突破越强正值越大
            factor_value = 0.1 + 0.9 * np.tanh(break_ratio)
            df.loc[down_break_mask, 'alpha004_enhanced'] = factor_value
        
        # --- 区间内：结合量能比和相对位置 ---
        inside_mask = valid_data & ~(up_break_mask | down_break_mask)
        if inside_mask.any():
            # 相对位置：距离中轨的距离占标准差的百分比（-1到1）
            relative_pos = (df.loc[inside_mask, 'ma2'] - df.loc[inside_mask, 'middle']) / df.loc[inside_mask, 'std8']
            # 量能比：限制在合理范围内
            vol_ratio_clipped = np.clip(df.loc[inside_mask, 'vol_ratio'], 0.1, 3.0)
            
            # 计算区间内因子值：位置权重 + 量能权重
            # 位置权重：越靠近下轨（相对位置为负）因子值越大
            position_weight = -relative_pos * 0.3  # 范围约 -0.3 到 0.3
            
            # 量能权重：放量时因子值增大
            volume_weight = (vol_ratio_clipped - 1.0) * 0.2  # 范围约 -0.18 到 0.4
            
            # 综合因子值
            factor_value = position_weight + volume_weight
            # 限制在 -0.5 到 0.5 范围内
            factor_value = np.clip(factor_value, -0.5, 0.5)
            df.loc[inside_mask, 'alpha004_enhanced'] = factor_value
        
        # 输出结果，风格与前面一致
        result = df[['stock_code', 'trade_date', 'alpha004_enhanced']].dropna(subset=['alpha004_enhanced']).copy()
        result = result.rename(columns={
            'stock_code': 'code',
            'trade_date': 'date',
            'alpha004_enhanced': 'value'
        })
        return result.reset_index(drop=True)



