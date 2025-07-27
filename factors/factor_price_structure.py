import pandas as pd
import numpy as np
from factors.factor_base import FactorBase
import talib

class Alpha002(FactorBase):
    name = "Alpha002"
    direction = -1  # 越小代表多空失衡加剧，可能反转
    description = (
        "Alpha002：涨跌幅增速/多空失衡度因子。\n"
        "公式：Alpha002 = -1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1)\n"
        "刻画多空失衡变动情况，用((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)表示多空力量不平衡度，\n"
        "再对该值做1阶差分并取相反数。\n"
        "信号解读：该因子越小，代表多空失衡度加速恶化，可能预示短期反转。"
    )
    data_requirements = {
        'daily': {'window': 3}  # 需要close, high, low，窗口3天保证能算出1阶差分
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        
        # 多空失衡度
        df['imbalance'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        # 防止分母为0
        df.loc[df['high'] == df['low'], 'imbalance'] = np.nan
        
        # 1阶差分
        df['delta_imbalance'] = df.groupby('stock_code')['imbalance'].diff()
        # 乘以-1
        df['value'] = -1 * df['delta_imbalance']
        df['factor'] = self.name
        
        # 输出结果
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)
    
class Alpha015(FactorBase):
    name = "Alpha015"
    direction = -1  # 越大表示高开越多，可能超买，后续回补；越小表示低开或贴近前收，可能修复

    description = (
        "Alpha015：跳空偏离因子。\n"
        "公式：Alpha015 = OPEN / DELAY(CLOSE, 1) - 1\n"
        "逻辑解释：\n"
        "    - 衡量当日开盘价相对于昨日收盘价的跳空幅度；\n"
        "    - 若开盘大幅高于昨收（高开），因子为正；\n"
        "    - 若低开或贴近，因子值趋近于0或为负。\n"
        "用途：可用于捕捉跳空缺口、短期价格预期变化的事件型信号。\n"
        "    - 方向为 -1，代表因子值越大可能越偏离基本面、存在回补可能。"
    )

    data_requirements = {
        'daily': {'window': 2}
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        
        # 昨日收盘
        df['close_delay'] = df.groupby('stock_code')['close'].shift(1)
        
        # 计算跳空幅度
        df['value'] = df['open'] / df['close_delay'] - 1
        df['factor'] = self.name
        
        # 整理输出
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)

class Alpha012(FactorBase):
    name = "Alpha012"
    direction = -1  # 因子值越大，代表高开且收盘偏离均价，通常为短线超买，后续回补概率大
    description = (
        "Alpha012：高开偏离+收盘偏离均价因子。\n"
        "公式：Alpha012 = RANK((OPEN - MEAN(VWAP, 10))) * (-1 * RANK(ABS(CLOSE - VWAP)))\n"
        "逻辑解释：\n"
        "    - 第一部分衡量开盘价相对10日均加权均价的偏离程度，越大代表高开越多；\n"
        "    - 第二部分衡量收盘价与当日均价的绝对偏离，越大代表收盘远离均价，乘以-1后越小信号越强；\n"
        "    - 两者相乘，捕捉高开+收盘偏离均价的事件型信号。\n"
        "    - direction = -1，因子值大通常为短线超买，后续回补概率大。"
    )
    data_requirements = {
        'daily': {'window': 11}  # 需要open, close, vwap，窗口11天保证能算10日均值
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        
        # 使用TA-Lib计算10日均VWAP
        def sma_10(x):
            return talib.SMA(x.values, timeperiod=10)
        
        # df['mean_vwap_10'] = df.groupby('stock_code')['vwap'].transform(sma_10)
        df['mean_vwap_10'] = df.groupby('stock_code')['vwap'].transform(sma_10)
        
        # 第一部分：开盘-10日均VWAP
        df['open_minus_meanvwap'] = df['open'] - df['mean_vwap_10']
        df['rank_open_minus_meanvwap'] = df.groupby('trade_date')['open_minus_meanvwap'].rank(method='average')
        
        # 第二部分：收盘与VWAP的绝对偏离
        df['abs_close_vwap'] = (df['close'] - df['vwap']).abs()
        df['rank_abs_close_vwap'] = df.groupby('trade_date')['abs_close_vwap'].rank(method='average')
        
        # 组合
        df['value'] = df['rank_open_minus_meanvwap'] * (-1 * df['rank_abs_close_vwap'])
        df['factor'] = self.name
        
        # 输出
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)

class Alpha018(FactorBase):
    name = "Alpha018"
    direction = 1  # 越大表示本周收盘较上周同期涨幅大，趋势延续
    description = (
        "Alpha018：相对上周同日收盘价变化因子。\n"
        "公式：Alpha018 = CLOSE / DELAY(CLOSE, 5)\n"
        "逻辑解释：\n"
        "    - 衡量今日收盘价相对于上周同一交易日的涨跌幅；\n"
        "    - 因子值大于1表示本周上涨，趋势延续；小于1表示下跌。\n"
        "    - direction=1，因子值越大，未来收益越高。"
    )
    data_requirements = {
        'daily': {'window': 6}  # 需要close，窗口6天保证能算5日延迟
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        
        # 5日延迟收盘价
        df['close_delay5'] = df.groupby('stock_code')['close'].shift(5)
        df['value'] = df['close'] / df['close_delay5']
        df['factor'] = self.name
        
        # 输出
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)

class Alpha017(FactorBase):
    name = "Alpha017"
    direction = -1  # 偏离顶部 + 价格跌幅越大，值越大，可能抓住反转

    description = (
        "Alpha017：瞬时偏离 + 收盘变化加权。\n"
        "公式：Alpha017 = RANK((VWAP - MAX(VWAP, 15))) ** DELTA(CLOSE, 5)\n"
        "解释：\n"
        "  - 测量当前vwap与过去15日最高vwap之间的偏离程度，进行横截面排序；\n"
        "  - 乘以（更准确是幂运算）收盘价5日变动幅度；\n"
        "  - 若VWAP大幅偏离高点、且价格急跌，则该值放大；\n"
        "  - 用于捕捉趋势拐点或情绪衰减时刻。"
    )

    data_requirements = {
        'daily': {'window': 20}
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        
        # 使用TA-Lib计算15日内VWAP的最大值
        def max_15(x):
            return talib.MAX(x.values, timeperiod=15)
        
        df['vwap_max15'] = df.groupby('stock_code')['vwap'].transform(max_15)
        
        # VWAP 偏离（负数表示远离顶点）
        df['vwap_diff'] = df['vwap'] - df['vwap_max15']

        # 横截面rank
        df['vwap_rank'] = df.groupby('trade_date')['vwap_diff'].rank(method='average')

        # 收盘价5日变化幅度
        df['delta_close'] = df.groupby('stock_code')['close'].transform(lambda x: x.diff(5))
        
        # 幂运算（确保无负值）
        df['value'] = df['vwap_rank'] ** df['delta_close']
        df['factor'] = self.name
        
        # 整理结果
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)

class Alpha038(FactorBase):
    name = "Alpha038"
    direction = -1  # 高点过高，趋势持续衰竭可能

    description = (
        "Alpha038：高点偏离均值因子。\n"
        "公式：Alpha038 = (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)\n"
        "含义解释：\n"
        "1. 判断当前 HIGH 是否高于近20日 HIGH 的均值；\n"
        "2. 若是，则取 HIGH 的2日变化并乘以 -1，代表价格可能阶段过高；\n"
        "3. 否则输出 0，认为价格正常。\n"
        "解读：该因子反映了价格短期冲高但回落（顶部乏力）的迹象，适合用于识别反转信号。"
    )

    data_requirements = {
        'daily': {'window': 20}  # HIGH 的20日均值 + DELTA lag 2 需要 22 日
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        
        # 使用TA-Lib计算 rolling 均值（20日）
        def sma_20(x):
            return talib.SMA(x.values, timeperiod=20)
        
        df['mean_high_20'] = df.groupby('stock_code')['high'].transform(sma_20)
        
        # 计算 DELTA(HIGH, 2)
        df['delta_high_2'] = df.groupby('stock_code')['high'].transform(lambda x: x.diff(2))
        
        # 构造因子值
        df['value'] = np.where(df['high'] > df['mean_high_20'], -1 * df['delta_high_2'], 0)
        df['factor'] = self.name
        
        # 整理输出
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)

class Alpha013(FactorBase):
    name = "Alpha013"
    direction = 1  # 因子值大 → 后续正收益（反转/修复假设）。回测若相反改为 -1 即可。
    description = (
        "Alpha013：当日成交重心相对区间位置因子。\n"
        "公式：Alpha013 = sqrt(High * Low) - VWAP\n"
        "解释：sqrt(High*Low) 为当日高低价的几何均值，近似刻画价格区间中枢的保守估计；VWAP 为成交量加权均价，代表成交重心。\n"
        "含义：\n"
        "  - 因子 > 0：VWAP 低于几何中枢，成交集中在较低位置，价格向上出现“拉高未能形成高位换手”或“低位吸筹”两种可能；\n"
        "  - 因子 < 0：VWAP 高于几何中枢，成交集中在偏高区域，可能代表动能强（资金愿意在高位成交）或拉高派发，需要与其它量价/换手类因子结合判断。\n"
        "方向（direction=1 说明）：默认假设 VWAP 低位（因子大）后续存在修复/反弹机会 → 正向因子；若回测显示相反，可将 direction 改为 -1。\n"
        "注意：若使用复权体系，请先在上游生成 adj_high / adj_low / adj_vwap，再传入；本实现会自动优先使用 'adj_high','adj_low','adj_vwap' 字段。"
    )
    data_requirements = {
        'daily': {'window': 2}
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        
        # 优先使用复权字段
        high_col = 'adj_high' if 'adj_high' in df.columns else 'high'
        low_col  = 'adj_low'  if 'adj_low'  in df.columns else 'low'
        vwap_col = 'adj_vwap' if 'adj_vwap' in df.columns else 'vwap'
        
        # 计算几何均值，防御：去除非正值
        valid = (df[high_col] > 0) & (df[low_col] > 0)
        df['value'] = np.nan
        df.loc[valid, 'value'] = np.sqrt(df.loc[valid, high_col] * df.loc[valid, low_col]) - df.loc[valid, vwap_col]
        df['factor'] = self.name
        
        # 输出结果，风格与前面一致
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)


class Alpha185(FactorBase):
    name = "Alpha185"
    direction = 1  # 排名值大表示开盘价与收盘价差异小，可能表示趋势稳定，未来收益较高
    description = (
        "Alpha185：开盘价与收盘价差异排名因子。\n"
        "公式：Alpha185 = RANK((-1 * ((1 - (OPEN / CLOSE))^2)))\n"
        "计算过程：\n"
        "1. 计算开盘价与收盘价的比率：OPEN / CLOSE；\n"
        "2. 计算与1的差异：(1 - (OPEN / CLOSE))；\n"
        "3. 平方后取负值：-1 * ((1 - (OPEN / CLOSE))^2)；\n"
        "4. 在横截面上进行排名：RANK(...)。\n"
        "解读：\n"
        "  - 因子值大：开盘价与收盘价接近，日内波动小，趋势稳定；\n"
        "  - 因子值小：开盘价与收盘价差异大，日内波动大，可能不稳定；\n"
        "  - 方向（direction=1）：假设开盘价与收盘价接近时（趋势稳定）未来收益较高。"
    )
    data_requirements = {
        'daily': {'window': 2}  # 只需要当日数据
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权价格
        open_col = 'adj_open' if 'adj_open' in df.columns else 'open'
        close_col = 'adj_close' if 'adj_close' in df.columns else 'close'

        # 计算开盘价与收盘价的比率
        df['open_close_ratio'] = df[open_col] / df[close_col]

        # 计算与1的差异的平方，并取负值
        df['alpha185_raw'] = -1 * ((1 - df['open_close_ratio']) ** 2)

        # 在横截面上进行排名
        df['value'] = df.groupby('trade_date')['alpha185_raw'].rank(method='average')
        df['factor'] = self.name
        
        # 输出结果
        result = df[['stock_code', 'trade_date', 'factor', 'value']].dropna(subset=['value']).copy()
        result = result.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return result.reset_index(drop=True)


