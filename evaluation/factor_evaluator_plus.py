# factor_evaluator.py  —— with required start/end logic & labeled groups
from __future__ import annotations
import time
from typing import Optional, Union, Type, Dict, Any, Iterable, List

import numpy as np
import pandas as pd

from factors.factor_base import FactorBase
from data_reader import get_daily_data, list_available_stocks
from utils import get_trading_dates


# ----------------- helpers -----------------
def _winsor_clip(s: pd.Series, p1: float = 0.005, p2: float = 0.995) -> pd.Series:
    if s.empty:
        return s
    lo, hi = s.quantile(p1), s.quantile(p2)
    return s.clip(lo, hi)

def _robust_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    if x.empty or y.empty:
        return np.nan
    xx = _winsor_clip(x.astype(float))
    yy = _winsor_clip(y.astype(float))
    try:
        return xx.corr(yy, method=method)
    except Exception:
        return np.nan

def _config_key(*parts) -> tuple:
    out = []
    for p in parts:
        if isinstance(p, dict):
            out.append(tuple(sorted(p.items())))
        elif isinstance(p, (list, tuple, set)):
            out.append(tuple(p))
        else:
            out.append(p)
    return tuple(out)

def _expanding_zscore(s: pd.Series) -> pd.Series:
    m = s.expanding(min_periods=1).mean()
    v = s.expanding(min_periods=1).var(ddof=1)
    std = np.sqrt(v).replace(0, np.nan)
    return ((s - m) / std).fillna(0.0)

def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=1).mean()
    std = s.rolling(window, min_periods=1).std(ddof=1).replace(0, np.nan)
    return ((s - m) / std).fillna(0.0)

def _calc_window_from_dates(start_date: str, end_date: str) -> int:
    return len(get_trading_dates(start_date=start_date,end_date=end_date))+5



# ----------------- main evaluator -----------------
class FactorEvaluator:
    """
    因子评估器（带取数）：
    - 支持 FactorBase 子类/实例 或直接传 factor_df / return_df
    - 统一输出 DataFrame，绘图层只消费 DF
    """

    _FACTOR_COLS = ("code", "date", "value")
    _RETURN_COLS = ("code", "date", "future_return")

    def __init__(
        self,
        factor: Optional[Union[Type[FactorBase], FactorBase]] = None,
        codes: Optional[Iterable[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,     # 必填
        period: int = 1,
        buy_price: str = "close",
        sell_price: str = "close",
        factor_df: Optional[pd.DataFrame] = None,
        return_df: Optional[pd.DataFrame] = None,
        window: Optional[int] = None,       # 可选
        **kwargs: Any,
    ):
        # --------- 时间口径逻辑（你要求的） ---------
        if end_date is None:
            raise ValueError("end_date 是必填参数")
        self.end_date = end_date

        if start_date is not None:
            # 有 start_date：据此推导 window
            self.start_date = start_date
            self.window = _calc_window_from_dates(start_date, end_date) if window is None else int(window)
        else:
            # 无 start_date：必须给 window
            if window is None:
                raise ValueError("start_date 和 window 不能同时为空；至少给一个")
            self.window = int(window)
            # 用 window 倒推 start_date
            tds = get_trading_dates(end_date=self.end_date, window=self.window)
            if len(tds) == 0:
                raise ValueError("get_trading_dates 返回为空，请检查数据源或参数")
            self.start_date = pd.to_datetime(tds[0]).strftime("%Y-%m-%d")

        # --------- 其他配置 ---------
        self.period = int(period)
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.codes = list(codes) if codes is not None else None
        self.factor = factor() if isinstance(factor, type) else factor

        self.factor_df = factor_df
        self.return_df = return_df
        self.merged: Optional[pd.DataFrame] = None

        self._cache: Dict[tuple, Any] = {}

        # 自动加载数据
        self._ensure_data_ready()

    # ----------------- data loading -----------------
    def _ensure_codes(self):
        if self.codes is None:
            self.codes = list_available_stocks("daily")

    def _load_factor_df(self) -> Optional[pd.DataFrame]:
        if self.factor is None:
            return None

        t0 = time.time()
        self._ensure_codes()

        # 先尝试读文件
        try:
            f = self.factor.read_factor_file()
            if f is not None and len(f) > 0:
                f = f.copy()
        except Exception:
            f = None

        if f is None or f.empty:
            f = self.factor.compute(self.codes, self.end_date, self.window)

        if "date" in f.columns:
            f["date"] = pd.to_datetime(f["date"])
            mask = (f["date"] >= self.start_date) & (f["date"] <= self.end_date)
            f = f.loc[mask].copy()

        if set(self._FACTOR_COLS).issubset(f.columns):
            out = f[list(self._FACTOR_COLS)].copy()
        elif set(["code", "date", "factor", "value"]).issubset(f.columns):
            out = f[["code", "date", "value"]].copy()
        else:
            vcol = "value"
            candidates = [c for c in f.columns if c not in ("code", "date", "factor")]
            if "value" not in f and candidates:
                vcol = candidates[-1]
            out = f[["code", "date", vcol]].rename(columns={vcol: "value"})

        print(f"[load factor] done in {time.time()-t0:.3f}s, rows={len(out)}")
        return out.reset_index(drop=True)

    def _load_return_df(self) -> pd.DataFrame:
        t0 = time.time()
        self._ensure_codes()

        daily = get_daily_data(self.codes, self.end_date, self.window).sort_values(
            ["stock_code", "trade_date"]
        )
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        mask = (daily["trade_date"] >= self.start_date) & (daily["trade_date"] <= self.end_date)
        daily = daily.loc[mask].copy()

        g = daily.groupby("stock_code")

        if self.buy_price == "close":
            buy = g[self.buy_price].shift(0)
            buy_shift = 0
        elif self.buy_price in ["open", "high", "low"]:
            buy = g[self.buy_price].shift(-1)
            buy_shift = -1
        else:
            raise ValueError("buy_price 仅支持 'close' 或 'open'/'high'/'low'")

        sell = g[self.sell_price].shift(-self.period + buy_shift)
        ret = (sell - buy) / buy

        ret_df = daily[["stock_code", "trade_date"]].copy()
        ret_df["future_return"] = ret.values
        ret_df = ret_df.rename(columns={"stock_code": "code", "trade_date": "date"})
        ret_df = ret_df.dropna(subset=["future_return"]).reset_index(drop=True)

        print(f"[load return] done in {time.time()-t0:.3f}s, rows={len(ret_df)}")
        return ret_df

    def _ensure_data_ready(self):
        if self.factor_df is None and self.factor is not None:
            self.factor_df = self._load_factor_df()
        if self.return_df is None:
            self.return_df = self._load_return_df()

        if self.factor_df is not None and self.return_df is not None:
            merged = pd.merge(self.factor_df, self.return_df, on=["code", "date"], how="inner")
            merged["date"] = pd.to_datetime(merged["date"])
            self.merged = merged.sort_values("date").reset_index(drop=True)
        else:
            self.merged = None

    # ----------------- basic series -----------------
    def ic_series(self, method: str = "pearson", robust: bool = True) -> pd.Series:
        if self.merged is None:
            raise ValueError("data not ready")
        key = _config_key("ic_series", method, robust)
        if key in self._cache:
            return self._cache[key]
        if robust:
            ic = (self.merged.groupby("date")
                  .apply(lambda g: _robust_corr(g["value"], g["future_return"], method))
                  .rename("IC").sort_index())
        else:
            ic = (self.merged.groupby("date")
                  .apply(lambda g: g["value"].corr(g["future_return"], method=method))
                  .rename("IC").sort_index())
        self._cache[key] = ic
        return ic

    def rank_ic_series(self) -> pd.Series:
        return self.ic_series(method="spearman", robust=True).rename("RankIC")

    # ----- 分组工具：返回从高到低的标签顺序 -----
    @staticmethod
    def _group_labels(n_groups: int) -> List[str]:
        labels = [f"Q{i+1}" for i in range(n_groups)]
        labels[0] = "Q1(最高)"
        labels[-1] = f"Q{n_groups}(最低)"
        return labels

    # 依据因子大小分组，返回“从高到低”的分位编号（0..n-1）；
    # 注意：qcut 的 label False 是按值从小到大 -> 0 最小，n-1 最大；我们将来输出顺序会改成“最大到最小”。
    @staticmethod
    def _qcode_desc_by_value(values: pd.Series, n_groups: int) -> pd.Series:
        rk = values.rank(method="first")             # 值越大，rank 越大
        q = pd.qcut(rk, n_groups, labels=False)      # 0=最小 ... n-1=最大
        return q

    def longshort_series(self, n_groups: int = 5) -> pd.Series:
        if self.merged is None:
            raise ValueError("data not ready")
        key = _config_key("ls", n_groups)
        if key in self._cache:
            return self._cache[key]

        rows = []
        for d, g in self.merged.groupby("date"):
            q = self._qcode_desc_by_value(g["value"], n_groups)
            g = g.assign(q=q)
            top = g.loc[g.q == n_groups - 1, "future_return"].mean()
            bot = g.loc[g.q == 0, "future_return"].mean()
            rows.append((d, float(top - bot)))
        ls = pd.Series(dict(rows)).sort_index().rename("LS")
        self._cache[key] = ls
        return ls

    def group_daily_returns(self, n_groups: int = 5) -> pd.DataFrame:
        """
        各组当日等权收益（列为清晰标签，顺序=从高到低）：
        ['Q1(最高)', 'Q2', ..., f'Q{n}(最低)']
        """
        if self.merged is None:
            raise ValueError("data not ready")
        key = _config_key("group_daily_labeled", n_groups)
        if key in self._cache:
            return self._cache[key]

        rows: Dict[int, list] = {}
        for d, g in self.merged.groupby("date"):
            q = self._qcode_desc_by_value(g["value"], n_groups)
            g = g.assign(q=q)
            for i in range(n_groups):
                gi = g[g.q == i]
                if gi.empty:
                    continue
                r = gi["future_return"].mean()
                rows.setdefault(i, []).append((d, float(r)))

        # 列顺序从“高”到“低”：n-1, ..., 0
        labels = self._group_labels(n_groups)
        order_idx = list(range(n_groups - 1, -1, -1))  # n-1 ... 0

        df_parts = {}
        for j, i in enumerate(order_idx):
            series = pd.Series({d: v for d, v in rows.get(i, [])}).sort_index()
            df_parts[labels[j]] = series

        df = pd.DataFrame(df_parts)
        df.index.name = "date"
        self._cache[key] = df
        return df

    def tstat_topn_series(self, top_n: int = 5) -> pd.Series:
        if self.merged is None:
            raise ValueError("data not ready")
        key = _config_key("tstat_topn", top_n)
        if key in self._cache:
            return self._cache[key]
        rows = []
        for d, g in self.merged.groupby("date"):
            top = g.nlargest(top_n, "value")
            if len(top) < 2:
                continue
            mu = top["future_return"].mean()
            sd = top["future_return"].std(ddof=1)
            tval = 0.0 if (sd is None or sd == 0 or np.isnan(sd)) else np.sqrt(len(top)) * mu / sd
            rows.append((d, float(tval)))
        t = pd.Series(dict(rows)).sort_index().rename(f"tstat_top{top_n}")
        self._cache[key] = t
        return t

    # ----------------- one-stop daily metrics -----------------
    def daily_metrics(
        self,
        n_groups: int = 5,
        top_n: int = 5,
        ir_mode: str = "expanding",  # "expanding" | "rolling"
        ir_window: int = 60,
    ) -> pd.DataFrame:
        key = _config_key("daily_metrics", n_groups, top_n, ir_mode, ir_window)
        if key in self._cache:
            return self._cache[key].copy()

        ic  = self.ic_series(method="pearson", robust=True)
        ric = self.rank_ic_series()

        if ir_mode == "expanding":
            ir_ic = _expanding_zscore(ic).rename("IR_from_IC")
            rir   = _expanding_zscore(ric).rename("RankIR_from_RankIC")
        elif ir_mode == "rolling":
            ir_ic = _rolling_zscore(ic, ir_window).rename("IR_from_IC")
            rir   = _rolling_zscore(ric, ir_window).rename("RankIR_from_RankIC")
        else:
            raise ValueError("ir_mode must be 'expanding' or 'rolling'")

        ls   = self.longshort_series(n_groups=n_groups)
        ir_ls = (_expanding_zscore(ls) if ir_mode == "expanding" else _rolling_zscore(ls, ir_window)).rename("IR_LS")
        tstat = self.tstat_topn_series(top_n=top_n).rename("tstat_topN")

        df = pd.concat([ic, ric, ir_ic, rir, ls, ir_ls, tstat], axis=1).sort_index()
        df["IC_cum"]   = df["IC"].fillna(0.0).cumsum()
        df["RankIC_cum"] = df["RankIC"].fillna(0.0).cumsum()
        df["IR_from_IC_cum"] = df["IR_from_IC"].fillna(0.0).cumsum()
        df["RankIR_from_RankIC_cum"] = df["RankIR_from_RankIC"].fillna(0.0).cumsum()
        df["IR_LS_cum"] = df["IR_LS"].fillna(0.0).cumsum()
        df["tstat_topN_cum"] = df["tstat_topN"].fillna(0.0).cumsum()

        self._cache[key] = df
        return df.copy()

    # ----------------- summary stats -----------------
    def ic_stats(self) -> pd.Series:
        ic = self.ic_series(method="pearson", robust=True).dropna()
        if ic.empty:
            return pd.Series(dtype=float)
        n = len(ic)
        mean = ic.mean(); std = ic.std(ddof=1)
        pos_ratio = (ic > 0).mean()
        t_value = mean / (std / np.sqrt(n)) if std > 0 else np.nan
        return pd.Series({
            "mean": mean, "std": std, "IR": mean / std if std > 0 else np.nan,
            "t": t_value, "pos_ratio": pos_ratio, "max": ic.max(), "min": ic.min(),
            "p5": ic.quantile(0.05), "p95": ic.quantile(0.95), "count": n
        })

    def rank_ic_stats(self) -> pd.Series:
        ric = self.rank_ic_series().dropna()
        if ric.empty:
            return pd.Series(dtype=float)
        n = len(ric)
        mean = ric.mean(); std = ric.std(ddof=1)
        pos_ratio = (ric > 0).mean()
        t_value = mean / (std / np.sqrt(n)) if std > 0 else np.nan
        return pd.Series({
            "mean": mean, "std": std, "RankIR": mean / std if std > 0 else np.nan,
            "t": t_value, "pos_ratio": pos_ratio, "max": ric.max(), "min": ric.min(),
            "p5": ric.quantile(0.05), "p95": ric.quantile(0.95), "count": n
        })

    # ----------------- utils -----------------
    def clear_cache(self):
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, int]:
        return {str(k): (isinstance(v, (pd.DataFrame, pd.Series)) and len(v) or 1)
                for k, v in self._cache.items()}
