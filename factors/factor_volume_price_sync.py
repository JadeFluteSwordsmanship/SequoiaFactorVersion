import pandas as pd
import numpy as np
from .factor_base import FactorBase
from scipy.stats import rankdata
from numba import njit
from .numba_utils import ts_rank_numba, rolling_corr_numba, rolling_max_numba
import talib

class Alpha001(FactorBase):
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
        weights = np.arange(1, period + 1)
        def weighted_sum(x):
            if np.isnan(x).any():
                return np.nan
            return np.dot(x, weights) / weights.sum()
        return pd.Series(arr).rolling(period).apply(weighted_sum, raw=True).values

    def _compute_impl(self, data):
        df = data['daily'].copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        out = []
        for code, g in df.groupby('stock_code', sort=False):
            g = g.reset_index(drop=True)
            vol = g['vol'].to_numpy(dtype=np.float64)
            low = g['low'].to_numpy(dtype=np.float64)
            vwap = g['vwap'].to_numpy(dtype=np.float64)
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


class Alpha045(FactorBase):
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

