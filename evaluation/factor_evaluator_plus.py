
from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Optional, Union, Type, Dict, Any, Tuple, Iterable, List

import numpy as np
import pandas as pd

# ---- External deps expected in user's project ----
# from factors.factor_base import FactorBase
# from data_reader import get_daily_data, list_available_stocks
# from utils import get_trading_dates

# To keep this module self-contained for tests, we guard optional imports:
try:
    from factors.factor_base import FactorBase  # type: ignore
except Exception:  # pragma: no cover
    class FactorBase:  # minimal stub for type hints
        name: str = "UnknownFactor"
        direction: int = 1
        description: str = ""

try:
    from data_reader import get_daily_data, list_available_stocks  # type: ignore
except Exception:  # pragma: no cover
    def list_available_stocks(_type: str) -> list[str]:
        return []
    def get_daily_data(codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        raise RuntimeError("get_daily_data not available in this environment")

try:
    from utils import get_trading_dates  # type: ignore
except Exception:  # pragma: no cover
    def get_trading_dates(end_date: str, window: int) -> list[str]:
        # fallback: generate business days
        end = pd.Timestamp(end_date)
        dates = pd.bdate_range(end - pd.Timedelta(days=window*2), end)
        return [d.strftime("%Y-%m-%d") for d in dates[-window:]]

# ---------------- Configs ----------------

@dataclass(frozen=True)
class LabelSpec:
    period: int = 1
    buy_price: str = "close"     # "open" / "close" / "high" / "low"
    sell_price: str = "close"    # same as above

@dataclass(frozen=True)
class EvalWindow:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    window: Optional[int] = None  # default will be 255 + period + 5

# ---------------- Utilities ----------------

def _now_str() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")

def _robust_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    """Clip tails to improve stability before correlation."""
    if x.empty or y.empty:
        return np.nan
    x = x.clip(x.quantile(0.005), x.quantile(0.995))
    y = y.clip(y.quantile(0.005), y.quantile(0.995))
    return x.corr(y, method=method)

def _config_key(*parts: Any) -> Tuple:
    """Immutable key for caching based on parts (dicts become tuples)."""
    out = []
    for p in parts:
        if isinstance(p, dict):
            out.append(tuple(sorted(p.items())))
        elif isinstance(p, (list, tuple, set)):
            out.append(tuple(p))
        else:
            out.append(p)
    return tuple(out)

# ---------------- Core Evaluator ----------------

class FactorEvaluator:
    """
    评估器：
      - 输入：FactorBase 子类/实例 或 已有的 factor_df / return_df
      - 输出：IC / RankIC / IR（多口径）/ 分组收益 / 图表数据
    设计目标：
      - 纯计算与绘图解耦（绘图函数只吃 DataFrame）
      - 缓存粒度：基于 (LabelSpec, 时间窗) + 源数据哈希键
    """
    def __init__(
        self,
        factor: Optional[Union[Type[FactorBase], FactorBase]] = None,
        codes: Optional[list] = None,
        eval_window: EvalWindow = EvalWindow(),
        label: LabelSpec = LabelSpec(),
        factor_df: Optional[pd.DataFrame] = None,
        return_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> None:
        self.factor = factor() if isinstance(factor, type) else factor
        self.codes = codes
        self.eval_window = eval_window
        self.label = label
        self.factor_df = factor_df
        self.return_df = return_df

        # resolved dates/window
        end_date = self.eval_window.end_date or _now_str()
        window = self.eval_window.window or (255 + self.label.period + 5)
        if self.eval_window.start_date is None:
            # infer start by trading calendar
            trading_dates = get_trading_dates(end_date=end_date, window=window)
            start_date = trading_dates[0] if trading_dates else end_date
        else:
            start_date = self.eval_window.start_date
        self._resolved = EvalWindow(start_date, end_date, window)

        # caches
        self._cache: Dict[Tuple, Any] = {}

        # load data lazily
        self._merged: Optional[pd.DataFrame] = None

    # ---------- Data loading ----------

    def _load_factor_df(self) -> pd.DataFrame:
        if self.factor_df is not None:
            df = self.factor_df.copy()
        else:
            if self.codes is None:
                self.codes = list_available_stocks('daily')
            # Prefer reading from disk if Factor supports it
            df = None
            if self.factor is not None and hasattr(self.factor, "read_factor_file"):
                try:
                    df = self.factor.read_factor_file()
                except Exception:
                    df = None
            if df is None and self.factor is not None:
                df = self.factor.compute(self.codes, self._resolved.end_date, self._resolved.window)  # type: ignore
            if df is None:
                raise RuntimeError("factor_df cannot be loaded")

        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            m = (df['date'] >= self._resolved.start_date) & (df['date'] <= self._resolved.end_date)
            df = df.loc[m]

        # unify columns to ['code','date','value']
        cols = df.columns
        if set(['code','date','value']).issubset(cols):
            out = df[['code','date','value']].copy()
        elif set(['code','date','factor','value']).issubset(cols):
            out = df[['code','date','value']].copy()
        else:
            # try to guess: last column as value
            value_col = [c for c in cols if c not in ('code','date')][-1]
            out = df[['code','date', value_col]].rename(columns={value_col:'value'})
        out['date'] = pd.to_datetime(out['date'])
        # apply factor direction if provided
        if self.factor is not None and hasattr(self.factor, 'direction'):
            try:
                d = int(getattr(self.factor, 'direction'))
                if d in (-1, 1):
                    out['value'] = out['value'] * d
            except Exception:
                pass
        return out

    def _load_return_df(self) -> pd.DataFrame:
        if self.return_df is not None:
            df = self.return_df.copy()
        else:
            if self.codes is None:
                self.codes = list_available_stocks('daily')
            daily = get_daily_data(self.codes, self._resolved.end_date, self._resolved.window)
            daily = daily.sort_values(['stock_code','trade_date']).copy()
            daily['trade_date'] = pd.to_datetime(daily['trade_date'])
            mask = (daily['trade_date'] >= self._resolved.start_date) & (daily['trade_date'] <= self._resolved.end_date)
            daily = daily.loc[mask].copy()

            g = daily.groupby('stock_code', sort=False)
            # buy
            if self.label.buy_price == 'close':
                buy = g[self.label.buy_price].shift(0)
                buy_shift = 0
            elif self.label.buy_price in ['open','high','low']:
                buy = g[self.label.buy_price].shift(-1)
                buy_shift = -1
            else:
                raise ValueError("buy_price must be close/open/high/low")
            # sell
            sell = g[self.label.sell_price].shift(-self.label.period + buy_shift)
            ret = (sell - buy) / buy
            df = daily[['stock_code','trade_date']].copy()
            df['future_return'] = ret.values
            df = df.rename(columns={'stock_code':'code','trade_date':'date'})
            df = df.dropna(subset=['future_return']).reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _merged_df(self) -> pd.DataFrame:
        if self._merged is not None:
            return self._merged
        f = self._load_factor_df()
        r = self._load_return_df()
        merged = pd.merge(f, r, on=['code','date'], how='inner')
        self._merged = merged
        return merged

    # ---------- Public API (cached) ----------

    def ic_series(self, method: str = "pearson") -> pd.Series:
        """Return daily IC series (index=date)."""
        key = _config_key("ic", method, asdict(self._resolved), asdict(self.label))
        if key in self._cache:
            return self._cache[key]
        m = self._merged_df()
        ics = (m.groupby('date')
                 .apply(lambda g: g['value'].corr(g['future_return'], method=method))
                 .rename('IC').sort_index())
        self._cache[key] = ics
        return ics

    def rank_ic_series(self) -> pd.Series:
        return self.ic_series(method="spearman")

    def ic_stats(self, method: str = "pearson") -> pd.Series:
        ic = self.ic_series(method).dropna()
        n = len(ic)
        mean = ic.mean()
        std = ic.std(ddof=1)
        t_value = np.nan if std == 0 else mean / (std / np.sqrt(n))
        return pd.Series({
            'mean': mean,
            'std': std,
            'IR': np.nan if std==0 else mean/std,
            't': t_value,
            'pos_ratio': (ic>0).mean() if n>0 else np.nan,
            'max': ic.max() if n>0 else np.nan,
            'min': ic.min() if n>0 else np.nan,
            'p5': ic.quantile(0.05) if n>0 else np.nan,
            'p95': ic.quantile(0.95) if n>0 else np.nan,
            'count': n
        })

    # ---- Daily IR (three definitions) ----

    def daily_ir_ic(self, window: Optional[int] = 60, method: str = "spearman") -> pd.DataFrame:
        """IR_t = zscore(IC_t) with rolling window or full-sample if window=None"""
        ic = self.ic_series("pearson" if method=="pearson" else "spearman")
        if window is None:
            mu, sd = 0.0, ic.std()
        else:
            mu, sd = ic.rolling(window).mean(), ic.rolling(window).std()
        ir = (ic - mu) / sd
        return pd.DataFrame({'IC': ic, 'IR': ir, 'IR_cum': ir.fillna(0).cumsum()})

    def daily_ir_longshort(self, n_groups: int = 5, window: int = 60) -> pd.DataFrame:
        """
        Build daily LS return (Top - Bottom) then IR_t = (LS - 0) / rolling_std(LS).
        """
        m = self._merged_df()
        ls_rows = []
        for d,g in m.groupby('date'):
            # quantile by rank to avoid ties
            q = pd.qcut(g['value'].rank(method='first'), n_groups, labels=False)
            g = g.assign(q=q)
            top = g[g.q==n_groups-1]['future_return'].mean()
            bot = g[g.q==0]['future_return'].mean()
            ls_rows.append((d, top - bot))
        ls = pd.Series(dict(ls_rows)).sort_index().rename('LS')
        sd = ls.rolling(window).std()
        ir = (ls - 0.0) / sd
        return pd.DataFrame({'LS': ls, 'IR': ir, 'IR_cum': ir.fillna(0).cumsum()})

    def daily_tstat_topn(self, top_n: int = 5) -> pd.DataFrame:
        """
        Cross-section t-stat for Top-N basket each day: t = sqrt(N) * mean / std.
        """
        m = self._merged_df()
        rows = []
        for d,g in m.groupby('date'):
            gg = g.nlargest(top_n, 'value')
            if len(gg) < 2:
                continue
            mu, sd = gg['future_return'].mean(), gg['future_return'].std()
            tval = np.nan if sd == 0 else np.sqrt(len(gg)) * mu / sd
            rows.append((d, tval))
        t = pd.Series(dict(rows)).sort_index().rename('tstat')
        return pd.DataFrame({'tstat': t, 'tstat_cum': t.fillna(0).cumsum()})

    # ---- Grouped returns ----

    def grouped_returns(self, n_groups: int = 5, overlap: bool = True) -> pd.DataFrame:
        """
        Return daily series for each quantile group.
        If overlap=True, approximate overlapping portfolio by rolling mean over holding period.
        """
        m = self._merged_df()
        rows: Dict[int, List[Tuple[pd.Timestamp, float]]] = {}
        for d,g in m.groupby('date'):
            q = pd.qcut(g['value'].rank(method='first'), n_groups, labels=False)
            g = g.assign(q=q)
            for i in range(n_groups):
                gi = g[g.q == i]
                if len(gi)==0: 
                    continue
                r = gi['future_return'].mean()
                rows.setdefault(i, []).append((d, r))

        df = pd.DataFrame({i: pd.Series({d:v for d,v in rows_i}).sort_index()
                           for i, rows_i in rows.items()})
        # to daily overlapping series
        if overlap and self.label.period > 1:
            daily = df.rolling(self.label.period).mean()
        else:
            daily = df
        daily.index.name = 'date'
        return daily

    # ---- helpers ----

    def clear_cache(self) -> None:
        self._cache.clear()
        self._merged = None


# ---------------- Plot helpers (pure functions) ----------------

def prepare_hist_and_cum(series: pd.Series, bins: int = 20) -> pd.DataFrame:
    """Return DataFrame with mid, prob, cumprob for histogram."""
    s = series.dropna().astype(float)
    if s.empty:
        return pd.DataFrame(columns=['mid','prob','cumprob'])
    counts, edges = np.histogram(s.values, bins=bins, density=True)
    mids = 0.5*(edges[1:] + edges[:-1])
    prob = counts / counts.sum()
    cumprob = prob.cumsum()
    return pd.DataFrame({'mid': mids, 'prob': prob, 'cumprob': cumprob})
