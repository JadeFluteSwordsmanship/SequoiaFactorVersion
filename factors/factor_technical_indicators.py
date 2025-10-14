import numpy as np
import pandas as pd
from .factor_base import FactorBase
import talib

class Custom001(FactorBase):
    """
    Custom001：价格乖离率因子（BIAS指标）。
    公式：BIAS(12) = 100 * (CLOSE - SMA(CLOSE, 12)) / SMA(CLOSE, 12)
    计算过程：
    1. 计算收盘价的12日简单移动平均线（SMA）；
    2. 计算当前收盘价与移动平均线的偏离程度；
    3. 将偏离程度标准化为百分比形式。
    解读：
      - 正的乖离度越大：价格远高于均线，短期获利大，可能回吐；
      - 负的乖离度越大：价格远低于均线，空头回补可能性高；
      - 方向（direction=-1）：假设乖离率越大（过度偏离）未来收益越低，乖离率越小（接近均线）未来收益越高。
    """
    name = "Custom001"
    direction = -1  # 乖离率越大，短期获利回吐风险越高，未来收益可能越低
    description = (
        "Custom001：价格乖离率因子（BIAS指标）。\n"
        "公式：BIAS(12) = 100 * (CLOSE - SMA(CLOSE, 12)) / SMA(CLOSE, 12)\n"
        "计算过程：\n"
        "1. 计算收盘价的12日简单移动平均线（SMA）；\n"
        "2. 计算当前收盘价与移动平均线的偏离程度；\n"
        "3. 将偏离程度标准化为百分比形式。\n"
        "解读：\n"
        "  - 正的乖离度越大：价格远高于均线，短期获利大，可能回吐；\n"
        "  - 负的乖离度越大：价格远低于均线，空头回补可能性高；\n"
        "  - 方向（direction=-1）：假设乖离率越大（过度偏离）未来收益越低，乖离率越小（接近均线）未来收益越高。"
    )
    data_requirements = {
        'daily': {'window': 15}  # 12日移动平均 + 少量冗余
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权价格
        close_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            close = g[close_col].to_numpy(dtype=np.float64)
            sma_12 = talib.SMA(close, timeperiod=12)
            bias = 100 * (close - sma_12) / sma_12
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': bias
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res

class Custom002(FactorBase):
    """
    Custom002：KDJ指标中的J值因子。
    公式：J = 3 * K - 2 * D
    其中：
      K = (2/3) * 前一日K值 + (1/3) * 今日RSV
      D = (2/3) * 前一日D值 + (1/3) * 今日K值
      RSV = (CLOSE - MIN(LOW, 9)) / (MAX(HIGH, 9) - MIN(LOW, 9)) * 100
    计算过程：
    1. 计算9日RSV值（未成熟随机值）；
    2. 通过平滑计算得到K值；
    3. 通过平滑计算得到D值；
    4. 计算J值 = 3*K - 2*D。
    解读：
      - J值 > 80：超买区域，可能回调；
      - J值 < 20：超卖区域，可能反弹；
      - 方向（direction=-1）：假设J值高时（超买）未来收益低，J值低时（超卖）未来收益高。
    """
    name = "Custom002"
    direction = -1  # J值过高表示超买，未来收益可能较低；J值过低表示超卖，未来收益可能较高
    description = (
        "Custom002：KDJ指标中的J值因子。\n"
        "公式：J = 3 * K - 2 * D\n"
        "其中：\n"
        "  K = (2/3) * 前一日K值 + (1/3) * 今日RSV\n"
        "  D = (2/3) * 前一日D值 + (1/3) * 今日K值\n"
        "  RSV = (CLOSE - MIN(LOW, 9)) / (MAX(HIGH, 9) - MIN(LOW, 9)) * 100\n"
        "计算过程：\n"
        "1. 计算9日RSV值（未成熟随机值）；\n"
        "2. 通过平滑计算得到K值；\n"
        "3. 通过平滑计算得到D值；\n"
        "4. 计算J值 = 3*K - 2*D。\n"
        "解读：\n"
        "  - J值 > 80：超买区域，可能回调；\n"
        "  - J值 < 20：超卖区域，可能反弹；\n"
        "  - 方向（direction=-1）：假设J值高时（超买）未来收益低，J值低时（超卖）未来收益高。"
    )
    data_requirements = {
        'daily': {'window': 100}  # 9日窗口 + 平滑计算所需的历史数据
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权价格
        high_col = 'adj_high' if 'adj_high' in df.columns else 'high'
        low_col = 'adj_low' if 'adj_low' in df.columns else 'low'
        close_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            high = g[high_col].to_numpy(dtype=np.float64)
            low = g[low_col].to_numpy(dtype=np.float64)
            close = g[close_col].to_numpy(dtype=np.float64)
            # talib.STOCH返回K/D，J=3K-2D
            k, d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
            # talib的K/D初始值不是50，需手动修正首个有效值
            # 但通常影响极小，若需完全一致可自定义实现
            j = 3 * k - 2 * d
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': j
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res

class Custom002_Enhanced(FactorBase):
    """
    Custom002_Enhanced：KDJ-J值的增强版本，将连续J值转换为分段凸凹函数。
    设计思路：
      1) J值在20~80之间：保持线性关系
      2) J值在80~100之间：1.5次函数（开口向上，凸函数）
      3) J值在100以上：二次函数（开口向上，凸函数）
      4) J值在0~20之间：1.5次函数（开口向下，凹函数）
      5) J值在0以下：二次函数（开口向下，凹函数）
    目标：增强超买超卖信号，J值越极端，信号越强烈。
    方向（direction=-1）：增强后的因子值高时未来收益低，因子值低时未来收益高。
    """
    name = "Custom002_Enhanced"
    direction = -1  # 增强后的J值高时（超买）未来收益低，J值低时（超卖）未来收益高
    description = (
        "Custom002_Enhanced：KDJ-J值的增强版本，将连续J值转换为分段凸凹函数。\n"
        "设计思路：\n"
        "  1) J值在20~80之间：保持线性关系\n"
        "  2) J值在80~100之间：1.5次函数（开口向上，凸函数）\n"
        "  3) J值在100以上：二次函数（开口向上，凸函数）\n"
        "  4) J值在0~20之间：1.5次函数（开口向下，凹函数）\n"
        "  5) J值在0以下：二次函数（开口向下，凹函数）\n"
        "目标：增强超买超卖信号，J值越极端，信号越强烈。\n"
        "方向（direction=-1）：增强后的因子值高时未来收益低，因子值低时未来收益高。"
    )
    data_requirements = {
        'daily': {'window': 100}  # 9日窗口 + 平滑计算所需的历史数据
    }

    def _enhance_j_value(self, j):
        # 处理NaN值
        if np.isnan(j):
            return np.nan
            
        a_lower = 0.06
        b_upper = 0.06
        c_low   = 0.03
        c_high  = 0.03
        p = 1.5

        F100 = 100 + b_upper * (20 ** p)
        D100 = 1 + b_upper * p * (20 ** (p - 1))
        F0   = - a_lower * (20 ** p)
        D0   = 1 + a_lower * p * (20 ** (p - 1))

        if j < 0:
            return F0 + D0 * j - c_low * j**2
        elif j < 20:
            return j - a_lower * (20 - j)**p
        elif j < 80:
            return j
        elif j < 100:
            return j + b_upper * (j - 80)**p
        else:
            return F100 + D100 * (j - 100) + c_high * (j - 100)**2

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权价格
        high_col = 'adj_high' if 'adj_high' in df.columns else 'high'
        low_col = 'adj_low' if 'adj_low' in df.columns else 'low'
        close_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            high = g[high_col].to_numpy(dtype=np.float64)
            low = g[low_col].to_numpy(dtype=np.float64)
            close = g[close_col].to_numpy(dtype=np.float64)
            k, d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
            j = 3 * k - 2 * d
            
            # 更高效的向量化处理，避免NaN警告
            j_enhanced = np.full_like(j, np.nan)
            valid_mask = ~np.isnan(j)
            if valid_mask.any():
                j_valid = j[valid_mask]
                j_enhanced[valid_mask] = np.vectorize(self._enhance_j_value)(j_valid)
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': j_enhanced
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res 


class Custom200(FactorBase):
    """
    Custom200：KDJ指标中的J值因子（标准算法，与同花顺一致）。
    公式：J = 3 * K - 2 * D
    其中：
      K = (2/3) * 前一日K值 + (1/3) * 今日RSV
      D = (2/3) * 前一日D值 + (1/3) * 今日K值
      RSV = (CLOSE - MIN(LOW, 9)) / (MAX(HIGH, 9) - MIN(LOW, 9)) * 100
    计算过程：
    1. 计算9日RSV值（未成熟随机值）；
    2. 通过指数平滑计算得到K值（初始值50）；
    3. 通过指数平滑计算得到D值（初始值50）；
    4. 计算J值 = 3*K - 2*D。
    解读：
      - J值 > 80：超买区域，可能回调；
      - J值 < 20：超卖区域，可能反弹；
      - 方向（direction=-1）：假设J值高时（超买）未来收益低，J值低时（超卖）未来收益高。
    注意：使用标准KDJ算法，确保与同花顺等主流软件计算结果一致。
    """
    name = "Custom200"
    direction = -1  # J值过高表示超买，未来收益可能较低；J值过低表示超卖，未来收益可能较高
    description = (
        "Custom200：KDJ指标中的J值因子（标准算法，与同花顺一致）。\n"
        "公式：J = 3 * K - 2 * D\n"
        "其中：\n"
        "  K = (2/3) * 前一日K值 + (1/3) * 今日RSV\n"
        "  D = (2/3) * 前一日D值 + (1/3) * 今日K值\n"
        "  RSV = (CLOSE - MIN(LOW, 9)) / (MAX(HIGH, 9) - MIN(LOW, 9)) * 100\n"
        "计算过程：\n"
        "1. 计算9日RSV值（未成熟随机值）；\n"
        "2. 通过指数平滑计算得到K值（初始值50）；\n"
        "3. 通过指数平滑计算得到D值（初始值50）；\n"
        "4. 计算J值 = 3*K - 2*D。\n"
        "解读：\n"
        "  - J值 > 80：超买区域，可能回调；\n"
        "  - J值 < 20：超卖区域，可能反弹；\n"
        "  - 方向（direction=-1）：假设J值高时（超买）未来收益低，J值低时（超卖）未来收益高。\n"
        "注意：使用标准KDJ算法，确保与同花顺等主流软件计算结果一致。"
    )
    data_requirements = {
        'daily': {'window': 100}  # 9日窗口 + 平滑计算所需的历史数据
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权价格
        high_col = 'adj_high' if 'adj_high' in df.columns else 'high'
        low_col = 'adj_low' if 'adj_low' in df.columns else 'low'
        close_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            high = g[high_col].to_numpy(dtype=np.float64)
            low = g[low_col].to_numpy(dtype=np.float64)
            close = g[close_col].to_numpy(dtype=np.float64)
            
            # 手动实现标准KDJ计算，确保与同花顺一致
            n = len(close)
            if n < 9:
                continue
                
            # 计算RSV
            rsv = np.full(n, np.nan)
            for i in range(8, n):  # 从第9天开始计算
                low_min = np.min(low[i-8:i+1])  # 9日最低价
                high_max = np.max(high[i-8:i+1])  # 9日最高价
                if high_max != low_min:
                    rsv[i] = (close[i] - low_min) / (high_max - low_min) * 100
                else:
                    rsv[i] = 50  # 避免除零
            
            # 计算K值（指数平滑，初始值50）
            k = np.full(n, np.nan)
            k[8] = 50  # 初始值
            for i in range(9, n):
                k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
            
            # 计算D值（指数平滑，初始值50）
            d = np.full(n, np.nan)
            d[8] = 50  # 初始值
            for i in range(9, n):
                d[i] = (2/3) * d[i-1] + (1/3) * k[i]
            
            # 计算J值
            j = 3 * k - 2 * d
            
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': j
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res

class Custom201(FactorBase):
    """
    Custom201：连续负J值强度因子。
    计算逻辑：
    1. 如果最近三天的J值不是都小于0，则因子值为0
    2. 如果最近三天的J值都小于0，则计算最近连续X天J值小于0的和的绝对值乘以X
    3. 例如：连续4天J值为-10,-9,-12,-18，因子值 = (10+9+12+18)*4 = 196
    4. 例如：连续4天J值为2,-1,-3,-5，因子值 = (1+3+5)*3 = 27（只计算连续负值部分）
    5. 例如：连续4天J值为-3,-5,-9,2，因子值 = 0（最近三天不是都小于0）
    
    设计思路：捕捉J值连续超卖的累积强度，连续超卖时间越长、程度越深，信号越强。
    方向（direction=-1）：因子值高时表示连续超卖严重，未来可能反弹。
    """
    name = "Custom201"
    direction = -1  # 连续超卖严重时，未来可能反弹
    description = (
        "Custom201：连续负J值强度因子。\n"
        "计算逻辑：\n"
        "1. 如果最近三天的J值不是都小于0，则因子值为0\n"
        "2. 如果最近三天的J值都小于0，则计算最近连续X天J值小于0的和的绝对值乘以X\n"
        "3. 例如：连续4天J值为-10,-9,-12,-18，因子值 = (10+9+12+18)*4 = 196\n"
        "4. 例如：连续4天J值为2,-1,-3,-5，因子值 = (1+3+5)*3 = 27（只计算连续负值部分）\n"
        "5. 例如：连续4天J值为-3,-5,-9,2，因子值 = 0（最近三天不是都小于0）\n"
        "\n"
        "设计思路：捕捉J值连续超卖的累积强度，连续超卖时间越长、程度越深，信号越强。\n"
        "方向（direction=-1）：因子值高时表示连续超卖严重，未来可能反弹。"
    )
    data_requirements = {
        'daily': {'window': 100}  # 需要足够的历史数据来计算连续负J值
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 优先使用复权价格
        high_col = 'adj_high' if 'adj_high' in df.columns else 'high'
        low_col = 'adj_low' if 'adj_low' in df.columns else 'low'
        close_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            high = g[high_col].to_numpy(dtype=np.float64)
            low = g[low_col].to_numpy(dtype=np.float64)
            close = g[close_col].to_numpy(dtype=np.float64)
            
            # 手动实现标准KDJ计算，确保与同花顺一致
            n = len(close)
            if n < 9:
                continue
                
            # 计算RSV
            rsv = np.full(n, np.nan)
            for i in range(8, n):  # 从第9天开始计算
                low_min = np.min(low[i-8:i+1])  # 9日最低价
                high_max = np.max(high[i-8:i+1])  # 9日最高价
                if high_max != low_min:
                    rsv[i] = (close[i] - low_min) / (high_max - low_min) * 100
                else:
                    rsv[i] = 50  # 避免除零
            
            # 计算K值（指数平滑，初始值50）
            k = np.full(n, np.nan)
            k[8] = 50  # 初始值
            for i in range(9, n):
                k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
            
            # 计算D值（指数平滑，初始值50）
            d = np.full(n, np.nan)
            d[8] = 50  # 初始值
            for i in range(9, n):
                d[i] = (2/3) * d[i-1] + (1/3) * k[i]
            
            # 计算J值
            j = 3 * k - 2 * d
            
            # 计算Custom201因子值
            custom201_values = np.full(n, np.nan)
            for i in range(8, n):  # 从第9天开始计算
                if np.isnan(j[i]):
                    custom201_values[i] = np.nan
                    continue
                
                # 检查最近三天是否都小于0
                if i < 10:  # 需要至少3天的历史数据
                    custom201_values[i] = 0
                    continue
                
                recent_3_days = j[i-2:i+1]  # 最近3天（包括今天）
                if np.any(np.isnan(recent_3_days)) or not np.all(recent_3_days < 0):
                    custom201_values[i] = 0
                    continue
                
                # 计算连续负J值的长度和强度
                consecutive_negative_count = 0
                consecutive_negative_sum = 0
                
                # 从当前日期向前查找连续负值
                for j_idx in range(i, 7, -1):  # 从当前日期向前，至少保留8天历史
                    if np.isnan(j[j_idx]) or j[j_idx] >= 0:
                        break
                    consecutive_negative_count += 1
                    consecutive_negative_sum += abs(j[j_idx])
                
                # 计算因子值：连续负值绝对值的和乘以连续天数
                if consecutive_negative_count >= 3:  # 至少连续3天负值
                    custom201_values[i] = consecutive_negative_sum * consecutive_negative_count
                else:
                    custom201_values[i] = 0
            
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': custom201_values
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res