import pandas as pd
import numpy as np
from .factor_base import FactorBase
from scipy.stats import rankdata
from numba import njit
from .numba_utils import ts_rank_numba, rolling_corr_numba, rolling_max_numba, decay_linear_numba
import talib

# Custom4系列

class Alpha001(FactorBase):
    """
    Alpha001：量价同步性短周期因子。
    公式：Alpha001 = -1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6)
    该因子衡量股票在过去6个交易日内，成交量变化（Δlog(vol)）与日内收益率((close-open)/open)的横截面排名序列之间的相关性。
    具体做法：
      - 每日对所有股票分别计算Δlog(成交量)和日内收益率，并在横截面上分别排名；
      - 对每只股票，计算其最近6日这两个排名序列的相关系数，并取相反数。
    信号解读：
      - 因子值为正，代表近期量变与价变同步性较弱，可能存在量价背离（如放量但价格未同步上涨），提示短期反转或调整风险；
      - 因子值为负，代表量价同步性较强，量增价升或量减价跌，反映趋势延续性。
    该因子常用于捕捉短周期内量价关系的变化，辅助判断趋势持续或反转。
    """
    name = "Alpha001"
    direction = -1  # 因子值大代表量价背离，押注反转
    description = (
        "Alpha001：量价同步性短周期因子。\n"
        "公式：Alpha001 = -1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6)\n"
        "该因子衡量股票在过去6个交易日内，成交量变化（Δlog(vol)）与日内收益率((close-open)/open)的横截面排名序列之间的相关性。\n"
        "具体做法：\n"
        "  - 每日对所有股票分别计算Δlog(成交量)和日内收益率，并在横截面上分别排名；\n"
        "  - 对每只股票，计算其最近6日这两个排名序列的相关系数，并取相反数。\n"
        "信号解读：\n"
        "  - 因子值为正，代表近期量变与价变同步性较弱，可能存在量价背离（如放量但价格未同步上涨），提示短期反转或调整风险；\n"
        "  - 因子值为负，代表量价同步性较强，量增价升或量减价跌，反映趋势延续性。\n"
        "该因子常用于捕捉短周期内量价关系的变化，辅助判断趋势持续或反转。"
    )
    data_requirements = {
        'daily': {'window': 7}  # 需要volume, open, close
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        # Δlog(vol)
        df['log_vol'] = np.log(df['vol'].replace(0, np.nan))
        df['delta_log_vol'] = df.groupby('stock_code')['log_vol'].diff()
        # (close - open) / open
        df['ret'] = (df['close'] - df['open']) / df['open']
        # 横截面排名
        df['rank_delta_log_vol'] = df.groupby('trade_date')['delta_log_vol'].rank(method='average', pct=False, ascending=False)
        df['rank_ret'] = df.groupby('trade_date')['ret'].rank(method='average', pct=False, ascending=False)
        # 计算 rolling correlation（用numba）
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            x = g['rank_delta_log_vol'].to_numpy(dtype=np.float64)
            y = g['rank_ret'].to_numpy(dtype=np.float64)
            value = -rolling_corr_numba(x, y, 6)
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
    
class Alpha005(FactorBase):
    """
    Alpha005：短周期成交量-高点同步因子。
    公式：Alpha005 = -1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3)
    该因子衡量股票在过去3个交易日内，5日窗口内成交量与最高价横截面排名的相关性，并取最大值再乘以 -1。
    计算过程如下：
    1. 对每只股票，在过去5日内分别计算成交量和最高价的TS横截面排名（TSRANK）序列；
    2. 以滚动窗口方式，在过去5日内计算两者的相关系数；
    3. 在过去3个交易日中取最大值；
    4. 最终取其相反数，作为因子值。
    解释：当成交量与高点排名相关性增强（即同步上升），可能为短期超买信号，因子值变小；
    当同步性下降，可能预示走势趋弱或反转，因子值变大。
    """
    name = "Alpha005"
    direction = -1  # 因子值越大，量价脱钩越显著，表示可能反转
    description = (
        "Alpha005：短周期成交量-高点同步因子。\n"
        "公式：Alpha005 = -1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3)\n"
        "该因子衡量股票在过去3个交易日内，5日窗口内成交量与最高价横截面排名的相关性，并取最大值再乘以 -1。\n"
        "计算过程如下：\n"
        "1. 对每只股票，在过去5日内分别计算成交量和最高价的TS横截面排名（TSRANK）序列；\n"
        "2. 以滚动窗口方式，在过去5日内计算两者的相关系数；\n"
        "3. 在过去3个交易日中取最大值；\n"
        "4. 最终取其相反数，作为因子值。\n"
        "解释：当成交量与高点排名相关性增强（即同步上升），可能为短期超买信号，因子值变小；\n"
        "当同步性下降，可能预示走势趋弱或反转，因子值变大。"
    )
    data_requirements = {
        'daily': {'window': 11}  # 需要volume和high，滚动窗口内保证足够数据
    }

    def _compute_impl(self, data):
        df = data['daily'][['stock_code', 'trade_date', 'high', 'vol']].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            high = g['high'].to_numpy(dtype=np.float64)
            vol  = g['vol' ].to_numpy(dtype=np.float64)

            vol_tr  = ts_rank_numba(vol,  5)   # TSRANK(VOLUME, 5)
            high_tr = ts_rank_numba(high, 5)   # TSRANK(HIGH,   5)

            corr = rolling_corr_numba(vol_tr, high_tr, 5)      # CORR(..., 5)
            mx   = rolling_max_numba(corr, 3)                  # TSMAX(..., 3)
            value = -mx                                        # 乘 -1

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
    
class Alpha016(FactorBase):
    """
    Alpha016：成交量与VWAP同步性的趋势反转因子。
    公式：Alpha016 = -1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)
    计算步骤：
    1. 每日对所有股票进行横截面排序（RANK），分别作用于成交量(VOLUME)和加权均价(VWAP)；
    2. 对每只股票计算过去5日内的两个rank序列的相关性（CORR）;
    3. 然后再对这个相关性值进行横截面排名；
    4. 最后对该rank序列进行5日窗口内的最大值提取（TSMAX）；
    5. 因子值为其相反数，越小代表同步性越强，可能为趋势延续信号。
    解读：同步性增强（即成交量与VWAP同升或同降）时，趋势可能延续；反之为背离。
    """
    name = "Alpha016"
    direction = -1  # 趋势反转因子
    description = (
        "Alpha016：成交量与VWAP同步性的趋势反转因子。\n"
        "公式：Alpha016 = -1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)\n"
        "计算步骤：\n"
        "1. 每日对所有股票进行横截面排序（RANK），分别作用于成交量(VOLUME)和加权均价(VWAP)；\n"
        "2. 对每只股票计算过去5日内的两个rank序列的相关性（CORR）;\n"
        "3. 然后再对这个相关性值进行横截面排名；\n"
        "4. 最后对该rank序列进行5日窗口内的最大值提取（TSMAX）；\n"
        "5. 因子值为其相反数，越小代表同步性越强，可能为趋势延续信号。\n"
        "解读：同步性增强（即成交量与VWAP同升或同降）时，趋势可能延续；反之为背离。"
    )
    data_requirements = {
        'daily': {'window': 9} 
    }

    def _compute_impl(self, data):
        df = data['daily'][['stock_code', 'trade_date', 'vol', 'vwap']].copy()
        df = df.sort_values(['trade_date', 'stock_code'])

        # 1) 横截面 rank
        df['rank_vol']  = df.groupby('trade_date')['vol'] .rank(method='average', ascending=False)
        df['rank_vwap'] = df.groupby('trade_date')['vwap'].rank(method='average', ascending=False)

        # 2) 对每只股票算 5 日滚动相关 （numba）
        df = df.sort_values(['stock_code', 'trade_date'])
        corr_holder = np.empty(len(df))
        corr_holder[:] = np.nan

        for code, g in df.groupby('stock_code', sort=False):
            idx = g.index.values
            x = g['rank_vol'].to_numpy(dtype=np.float64)
            y = g['rank_vwap'].to_numpy(dtype=np.float64)
            corr = rolling_corr_numba(x, y, 5)
            corr_holder[idx] = corr

        df['corr'] = corr_holder

        # 3) corr 的横截面 rank
        df['corr_rank'] = df.groupby('trade_date')['corr'].rank(method='average')

        # 4) 每支股票上对 corr_rank 做 5 日 TSMAX（numba）
        alpha_holder = np.empty(len(df))
        alpha_holder[:] = np.nan

        for code, g in df.groupby('stock_code', sort=False):
            idx = g.index.values
            cr = g['corr_rank'].to_numpy(dtype=np.float64)
            mx = rolling_max_numba(cr, 5)
            alpha_holder[idx] = -mx   # 最后一步乘 -1

        df['value'] = alpha_holder

        res = df[['stock_code', 'trade_date', 'value']].dropna(subset=['value']).copy()
        res.columns = ['code', 'date', 'value']
        res['factor'] = self.name
        return res[['code', 'date', 'factor', 'value']].reset_index(drop=True)
    
class Alpha032(FactorBase):
    """
    Alpha032：高点与成交量的短期同步性因子。
    公式：Alpha032 = -1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)
    计算步骤：
    1. 每日对所有股票的 HIGH 与 VOLUME 进行横截面排名（RANK）；
    2. 对每只股票计算其近3日的两个 rank 序列之间的相关性（CORR）；
    3. 对该相关性值在横截面上再做 RANK；
    4. 对该 RANK 序列再做滚动3日求和（TS_SUM）；
    5. 最后乘以 -1，因子值越小表示高量同步性越强，可能趋势延续。
    用途：适合捕捉短期量价关系变化，识别趋势与反转信号。
    """
    name = "Alpha032"
    direction = -1  # 越小代表同步性越强，可能趋势延续；越大代表背离增强，可能反转
    description = (
        "Alpha032：高点与成交量的短期同步性因子。\n"
        "公式：Alpha032 = -1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)\n"
        "计算步骤：\n"
        "1. 每日对所有股票的 HIGH 与 VOLUME 进行横截面排名（RANK）；\n"
        "2. 对每只股票计算其近3日的两个 rank 序列之间的相关性（CORR）；\n"
        "3. 对该相关性值在横截面上再做 RANK；\n"
        "4. 对该 RANK 序列再做滚动3日求和（TS_SUM）；\n"
        "5. 最后乘以 -1，因子值越小表示高量同步性越强，可能趋势延续。\n"
        "用途：适合捕捉短期量价关系变化，识别趋势与反转信号。"
    )
    data_requirements = {
        'daily': {'window': 7}
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        # 每日横截面rank
        df['rank_high'] = df.groupby('trade_date')['high'].rank(method='average', ascending=False)
        df['rank_vol'] = df.groupby('trade_date')['vol'].rank(method='average', ascending=False)
        # 每只股票做rolling相关性（numba）
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            x = g['rank_high'].to_numpy(dtype=np.float64)
            y = g['rank_vol'].to_numpy(dtype=np.float64)
            corr = rolling_corr_numba(x, y, 3)
            # 横截面rank
            g['corr'] = corr
            g['corr_rank'] = g['corr'].rank(method='average')
            # TS_SUM: rolling sum 3日
            g['ts_sum'] = pd.Series(np.convolve(g['corr_rank'], np.ones(3), 'full')[:len(g['corr_rank'])], index=g.index)
            g['value'] = -1 * g['ts_sum']
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'factor': self.name,
                'value': g['value']
            })
            out.append(tmp)
        res = pd.concat(out, ignore_index=True)
        res = res.dropna(subset=['value']).reset_index(drop=True)
        return res


class Alpha042(FactorBase):
    """
    Alpha042：高点波动率 + 量价同步性因子。
    公式：Alpha042 = (-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10)
    计算过程：
    1. 对每只股票计算其过去10日HIGH的标准差，作为高点波动率；
    2. 在每日横截面上，对该标准差进行排名（RANK）；
    3. 对每只股票，在过去10日窗口内计算HIGH与VOLUME的相关系数；
    4. 将步骤2结果乘以 -1，再与步骤3结果相乘，得到因子值。
    解读：高点波动越大、量价越同步，则因子值越小，表示可能趋势延续；反之可能反转。
    """
    name = "Alpha042"
    direction = -1  # 越小代表高点波动率越高，且量价同步性更强

    description = (
        "Alpha042：高点波动率 + 量价同步性因子。\n"
        "公式：Alpha042 = (-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10)\n"
        "计算过程：\n"
        "1. 对每只股票计算其过去10日HIGH的标准差，作为高点波动率；\n"
        "2. 在每日横截面上，对该标准差进行排名（RANK）；\n"
        "3. 对每只股票，在过去10日窗口内计算HIGH与VOLUME的相关系数；\n"
        "4. 将步骤2结果乘以 -1，再与步骤3结果相乘，得到因子值。\n"
        "解读：高点波动越大、量价越同步，则因子值越小，表示可能趋势延续；"
        "反之可能反转。"
    )

    data_requirements = {
        'daily': {'window': 12}  # 预留更多天以防前置缺失
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            high = g['high'].to_numpy(dtype=np.float64)
            vol = g['vol'].to_numpy(dtype=np.float64)
            std_high = talib.STDDEV(high, timeperiod=10, nbdev=1)
            corr = rolling_corr_numba(high, vol, 10)
            g['std_high'] = std_high
            g['corr'] = corr
            out.append(g)
        df2 = pd.concat(out, ignore_index=True)
        # 横截面rank
        df2['rank_std'] = df2.groupby('trade_date')['std_high'].rank(method='average')
        df2['value'] = -1 * df2['rank_std'] * df2['corr']
        res = df2[['stock_code', 'trade_date', 'value']].dropna(subset=['value']).copy()
        res = res.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        res['factor'] = self.name
        return res[['code', 'date', 'factor', 'value']].reset_index(drop=True)


class Alpha044(FactorBase):
    """
    Alpha044：低点成交量同步 + VWAP变化衰减因子。
    公式：
    Alpha044 = TSRANK(DECAYLINEAR(CORR(LOW, MEAN(VOLUME,10), 7), 6), 4)
              + TSRANK(DECAYLINEAR(DELTA(VWAP, 3), 10), 15)
    步骤说明：
    1. 计算10日均量，并对LOW与其做7日相关性；
    2. 对该相关性做线性衰减加权（6日），再做4日TSRANK；
    3. 计算VWAP的3日变动，做10日线性衰减，再进行15日TSRANK；
    4. 最终两部分相加作为因子值。
    解读：用于同时捕捉低点成交关系和价格变动的趋势信号。
    """
    name = "Alpha044"
    direction = 1  # 越大表示低点与成交量更同步，VWAP变化大

    description = (
        "Alpha044：低点成交量同步 + VWAP变化衰减因子。\n"
        "公式：\n"
        "Alpha044 = TSRANK(DECAYLINEAR(CORR(LOW, MEAN(VOLUME,10), 7), 6), 4)\n"
        "          + TSRANK(DECAYLINEAR(DELTA(VWAP, 3), 10), 15)\n"
        "步骤说明：\n"
        "1. 计算10日均量，并对LOW与其做7日相关性；\n"
        "2. 对该相关性做线性衰减加权（6日），再做4日TSRANK；\n"
        "3. 计算VWAP的3日变动，做10日线性衰减，再进行15日TSRANK；\n"
        "4. 最终两部分相加作为因子值。\n"
        "解读：用于同时捕捉低点成交关系和价格变动的趋势信号。"
    )

    data_requirements = {
        'daily': {'window': 29}
    }

    def _decay_linear(self, arr, period):
        # 使用numba优化的decay_linear函数，速度提升10-50倍
        return decay_linear_numba(arr, period)

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        
        # 预分配结果列表，避免频繁的DataFrame创建
        all_codes = []
        all_dates = []
        all_values = []
        
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            vol = g['vol'].to_numpy(dtype=np.float64)
            low = g['low'].to_numpy(dtype=np.float64)
            vwap = g['vwap'].to_numpy(dtype=np.float64)
            dates = g['trade_date'].values
            
            # mean volume
            mean_vol_10 = talib.SMA(vol, timeperiod=10)
            # corr(LOW, mean_vol_10, 7)
            corr = rolling_corr_numba(low, mean_vol_10, 7)
            # decaylinear(corr, 6)
            decay_corr = self._decay_linear(corr, 6)
            # tsrank(..., 4)
            tsrank_corr = ts_rank_numba(decay_corr, 4)
            # delta(vwap, 3)
            delta_vwap = np.concatenate([np.full(3, np.nan), vwap[3:] - vwap[:-3]])
            # decaylinear(delta, 10)
            decay_delta = self._decay_linear(delta_vwap, 10)
            # tsrank(..., 15)
            tsrank_delta = ts_rank_numba(decay_delta, 15)
            value = tsrank_corr + tsrank_delta
            
            # 只保留非NaN的值
            valid_mask = ~np.isnan(value)
            if valid_mask.any():
                all_codes.extend([code] * valid_mask.sum())
                all_dates.extend(dates[valid_mask])
                all_values.extend(value[valid_mask])
        
        # 一次性创建DataFrame
        res = pd.DataFrame({
            'code': all_codes,
            'date': all_dates,
            'factor': self.name,
            'value': all_values
        })
        
        return res


class Alpha045(FactorBase):
    """
    Alpha045：加权价变化速率 + 长期量价同步性因子。
    公式：Alpha045 = RANK(DELTA(0.6*CLOSE + 0.4*OPEN, 1)) * RANK(CORR(VWAP, MEAN(VOLUME, 150), 15))
    计算步骤：
    1. 构造加权价：0.6 * CLOSE + 0.4 * OPEN，并计算1日差值，再对每日横截面排名；
    2. 对每只股票，计算150日均量；并在15日窗口内对VWAP与其进行rolling相关性；
    3. 将该相关性值按日期在横截面上进行排名；
    4. 两者相乘得到因子值。
    解读：因子综合刻画了短期价格动量与长期量价联动性，值越小可能提示反转信号。
    """
    name = "Alpha045"
    direction = -1  # 越小代表价格下跌+量价背离，可能反转

    description = (
        "Alpha045：加权价变化速率 + 长期量价同步性因子。\n"
        "公式：Alpha045 = RANK(DELTA(0.6*CLOSE + 0.4*OPEN, 1)) * RANK(CORR(VWAP, MEAN(VOLUME, 150), 15))\n"
        "计算步骤：\n"
        "1. 构造加权价：0.6 * CLOSE + 0.4 * OPEN，并计算1日差值，再对每日横截面排名；\n"
        "2. 对每只股票，计算150日均量；并在15日窗口内对VWAP与其进行rolling相关性；\n"
        "3. 将该相关性值按日期在横截面上进行排名；\n"
        "4. 两者相乘得到因子值。\n"
        "解读：因子综合刻画了短期价格动量与长期量价联动性，值越小可能提示反转信号。"
    )

    data_requirements = {
        'daily': {'window': 166}  # 为了能算出150日均值 + 15日滚动 + 1阶差
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            close = g['close'].to_numpy(dtype=np.float64)
            open_ = g['open'].to_numpy(dtype=np.float64)
            vol = g['vol'].to_numpy(dtype=np.float64)
            vwap = g['vwap'].to_numpy(dtype=np.float64)
            weighted_price = 0.6 * close + 0.4 * open_
            delta_price = np.concatenate([[np.nan], weighted_price[1:] - weighted_price[:-1]])
            mean_vol_150 = talib.SMA(vol, timeperiod=150)
            corr = rolling_corr_numba(vwap, mean_vol_150, 15)
            tmp = pd.DataFrame({
                'code': code,
                'date': g['trade_date'].values,
                'delta_price': delta_price,
                'corr': corr
            })
            out.append(tmp)
        df_all = pd.concat(out, ignore_index=True)
        # 横截面rank（在所有股票的同一天上做）
        df_all['rank_delta'] = df_all.groupby('date')['delta_price'].rank(method='average')
        df_all['rank_corr'] = df_all.groupby('date')['corr'].rank(method='average')
        df_all['value'] = df_all['rank_delta'] * df_all['rank_corr']
        df_all['factor'] = self.name
        res = df_all[['code', 'date', 'factor', 'value']].dropna(subset=['value']).reset_index(drop=True)
        return res


class Alpha139(FactorBase):
    """
    Alpha139：开盘价与成交量相关性因子。
    公式：Alpha139 = -1 * CORR(OPEN, VOLUME, 10)
    计算过程：
    1. 对每只股票，计算过去10日内开盘价与成交量的滚动相关系数；
    2. 取相关系数的负值作为因子值。
    解读：
      - 因子值为正：开盘价与成交量负相关，可能表示价格下跌时成交量放大，或价格上涨时成交量萎缩；
      - 因子值为负：开盘价与成交量正相关，可能表示价量同向变化；
      - 方向（direction=1）：假设开盘价与成交量负相关时（因子值大）未来收益高，正相关时（因子值小）未来收益低。
    """
    name = "Alpha139"
    direction = 1  # 因子值与未来收益正相关：开盘价与成交量负相关时因子值为正，预期未来收益高
    description = (
        "Alpha139：开盘价与成交量相关性因子。\n"
        "公式：Alpha139 = -1 * CORR(OPEN, VOLUME, 10)\n"
        "计算过程：\n"
        "1. 对每只股票，计算过去10日内开盘价与成交量的滚动相关系数；\n"
        "2. 取相关系数的负值作为因子值。\n"
        "解读：\n"
        "  - 因子值为正：开盘价与成交量负相关，可能表示价格下跌时成交量放大，或价格上涨时成交量萎缩；\n"
        "  - 因子值为负：开盘价与成交量正相关，可能表示价量同向变化；\n"
        "  - 方向（direction=1）：假设开盘价与成交量负相关时（因子值大）未来收益高，正相关时（因子值小）未来收益低。"
    )
    data_requirements = {
        'daily': {'window': 12}  # 10日滚动窗口 + 少量冗余
    }

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            open_ = g['open'].to_numpy(dtype=np.float64)
            vol = g['vol'].to_numpy(dtype=np.float64)
            corr = rolling_corr_numba(open_, vol, 10)
            value = -corr
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

class Custom400(FactorBase):
    """
    Custom400：识别“前期有过一轮上涨 → 近1.5~2个月缓跌缩量缩波动 → 低位刚反弹”的形态。
    分数越高，越贴近该形态，期望反弹强度越大（direction=1）。

    可调参数：
      N_peak=60           # 观察前序高点与回撤深度的窗口
      center_depth=0.60   # 回撤深度中心（close / HHV_60）
      width_depth=0.08    # 回撤深度容差
      center_len=35     # 距60日高点的期望天数（约1.5~2个月）
      width_len=10.0
      center_volr=0.50    # 量能衰减（VOL_MA10 / VOL_MA60）的中心
      width_volr=0.20
      center_atrr=0.50    # 波动收敛（ATR10 / ATR60）的中心
      width_atrr=0.20
      rally_low=1.30      # 过去60日振幅的下限门槛（HHV60/LLV60）
      rally_high=2.00     # 上限做线性压缩
      weights = dict(depth=0.25, length=0.20, dry=0.15, volcmp=0.10, rally=0.15, trigger=0.15)
    """
    name = "Custom400"
    direction = 1
    description = (
        "前涨—缓跌缩量缩波动—低位反弹 的复合打分；值越大越像目标形态。"
    )
    data_requirements = {
        "daily": {"window": 120}  # 需要open/high/low/close/vol；ATR与均线/60日窗口
    }

    # --------- 小工具：高斯打分 & 区间0-1缩放 ---------
    @staticmethod
    def _gauss_score(x, center, width):
        # exp(-((x-c)/w)^2)，并剪裁到[0,1]
        s = np.exp(-np.square((x - center) / (width + 1e-12)))
        return np.clip(s, 0.0, 1.0)

    @staticmethod
    def _scale01(x, low, high):
        return np.clip((x - low) / (high - low + 1e-12), 0.0, 1.0)

    def _compute_impl(self, data):
        df = data["daily"].copy()
        df = df.sort_values(["stock_code", "trade_date"])
        g = df.groupby("stock_code", group_keys=False)

        N_peak = 60

        # ---- 价格、成交量、ATR、均线等基础特征 ----
        df["ma5"]  = g["close"].transform(lambda s: s.rolling(5).mean())
        df["ma10"] = g["close"].transform(lambda s: s.rolling(10).mean())
        df["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
        df["v_ma10"] = g["vol"].transform(lambda s: s.rolling(10).mean())
        df["v_ma20"] = g["vol"].transform(lambda s: s.rolling(20).mean())
        df["v_ma60"] = g["vol"].transform(lambda s: s.rolling(60).mean())

        df["hhv60"] = g["close"].transform(lambda s: s.rolling(N_peak).max())
        df["llv60"] = g["close"].transform(lambda s: s.rolling(N_peak).min())

        # True Range & ATR
        prev_close = g["close"].shift(1)

        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs()
        ], axis=1).max(axis=1)

        df["tr"] = tr

        # 用 groupby.transform 做滚动均值（不会打乱索引）
        df["atr10"] = g["tr"].transform(lambda s: s.rolling(10, min_periods=1).mean())
        df["atr60"] = g["tr"].transform(lambda s: s.rolling(60, min_periods=1).mean())
        # ---- 距离最近60日高点的天数（since last 60d high）----
        # 标注：当日close等于滚动60日高时为 True；用累计和+分组的技巧得到“自上次高点以来的天数”
        is_peak = (df["close"] == df["hhv60"]) & df["hhv60"].notna()
        grp_id = g.apply(lambda s: is_peak.loc[s.index].cumsum())
        df["days_since_peak"] = g.apply(lambda s: pd.Series(
            np.arange(len(s)), index=s.index
        )).groupby(grp_id).cumcount()
        # 在出现第一个60日高点之前的天数置为NaN（不计分）
        seen = g.apply(lambda s: is_peak.loc[s.index].cumsum()) > 0
        df.loc[~seen, "days_since_peak"] = np.nan

        # ---- 子评分计算（全部0~1）----
        # 1) 回撤深度：close / hhv60 ~ 0.5~0.7
        depth_ratio = df["close"] / df["hhv60"]
        s_depth = self._gauss_score(depth_ratio, center=0.60, width=0.08)

        # 2) 回撤时长：30~45日最优
        s_len = self._gauss_score(df["days_since_peak"], center=35, width=10.0)

        # 3) 量能衰减：v_ma10 / v_ma60 越低越好（地量）
        vol_ratio = df["v_ma10"] / df["v_ma60"]
        s_dry = self._gauss_score(vol_ratio, center=0.50, width=0.20)

        # 4) 波动收敛：atr10 / atr60 越低越好（地波动）
        atr_ratio = df["atr10"] / df["atr60"]
        s_volcmp = self._gauss_score(atr_ratio, center=0.50, width=0.20)

        # 5) 前序行情强度：HHV60 / LLV60 足够大（≥1.3）
        amp = df["hhv60"] / df["llv60"]
        s_rally = self._scale01(amp, low=1.30, high=2.00)

        # 6) 反弹触发：均线转多 + 量能回暖 + 近3日转正
        ret3 = df["close"] / g["close"].shift(3) - 1.0
        t1 = self._scale01(df["close"] / df["ma10"] - 1.0, 0.00, 0.05)   # close相对MA10抬升
        t2 = self._scale01(df["ma5"] / df["ma10"] - 1.0, 0.00, 0.03)    # MA5金叉MA10
        t3 = self._scale01(df["vol"] / df["v_ma20"] - 1.0, 0.00, 0.50)  # 放量
        t4 = self._scale01(ret3, 0.00, 0.06)                             # 3日由弱转强
        s_trigger = (t1 + t2 + t3 + t4) / 4.0

        # ---- 加权几何平均组合（短板效应更强）----
        W = {"depth":0.25, "length":0.20, "dry":0.15, "volcmp":0.10, "rally":0.15, "trigger":0.15}
        eps = 1e-6
        combo_log = (
            W["depth"]   * np.log(s_depth   + eps) +
            W["length"]  * np.log(s_len     + eps) +
            W["dry"]     * np.log(s_dry     + eps) +
            W["volcmp"]  * np.log(s_volcmp  + eps) +
            W["rally"]   * np.log(s_rally   + eps) +
            W["trigger"] * np.log(s_trigger + eps)
        )
        df["value"] = np.exp(combo_log)  # 自然落在(0,1]，越大越像粉框
        df["factor"] = self.name

        out = df[["stock_code", "trade_date", "factor", "value"]].dropna(subset=["value"]).copy()
        out = out.rename(columns={"stock_code":"code", "trade_date":"date"})
        return out.reset_index(drop=True)