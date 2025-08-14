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

def _corr_safe(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    """安全相关：样本不足、常数列 → NaN；不触发未来 pandas 的 groupby.apply 警告。"""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return np.nan
    xx, yy = x[m], y[m]
    if xx.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
        return np.nan
    try:
        return xx.corr(yy, method=method)
    except Exception:
        return np.nan

def _robust_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    """Winsorize 后再相关。默认不启用（robust=False）以保持与你原口径一致。"""
    return _corr_safe(_winsor_clip(x), _winsor_clip(y), method=method)

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
    std = s.expanding(min_periods=1).std(ddof=1).replace(0, np.nan)
    return ((s - m) / std).fillna(0.0)

def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=1).mean()
    std = s.rolling(window, min_periods=1).std(ddof=1).replace(0, np.nan)
    return ((s - m) / std).fillna(0.0)

def _calc_window_from_dates(start_date: str, end_date: str) -> int:
    tds = get_trading_dates(start_date=start_date, end_date=end_date)
    if not tds:
        return 256
    return len(tds) + 5

def _is_open_like(px: str) -> bool:
    return px in ("open", "high", "low")

def _is_close_like(px: str) -> bool:
    return px == "close"

def _holding_days(period: int, buy_price: str, sell_price: str) -> int:
    """
    A股T+1：buy=open 且 sell=close 时，period=1 实际占用 2 天。
    其它常见组合：
      - buy=close ⇒ K = period
      - buy=open/high/low:
            sell=close  -> K = period + 1
            sell=open   -> K = period
    """
    k = int(period)
    if _is_open_like(buy_price) and _is_close_like(sell_price):
        k += 1
    return max(k, 1)

def _period_to_daily_equiv(R_period, k_hold: int):
    """
    将“period 总收益”折算为“等效日收益”：(1 + R)^(1/K) - 1
    同时支持 Series 和 DataFrame。对 NaN 安全处理。
    """
    K = float(k_hold)
    if isinstance(R_period, pd.DataFrame):
        df = R_period.apply(pd.to_numeric, errors="coerce")
        return (1.0 + df).pow(1.0 / K) - 1.0
    else:
        s = pd.to_numeric(R_period, errors="coerce")
        return (1.0 + s).pow(1.0 / K) - 1.0


# ----------------- main evaluator -----------------
class FactorEvaluator:
    """
    - 支持 FactorBase 子类/实例 或直接传 factor_df/return_df
    - 分组列名：Q1(最高)…Qn(最低)，顺序=因子从高到低
    - 分组/LS/TopN：先把 period 收益折算为“日收益”(等效日)，再做 K 桶叠加（rolling(K).mean()）
      -> 再 cumprod 得到净值。彻底避免 period>1 时的收益高估。
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
        window: Optional[int] = None,
        **kwargs: Any,
    ):
        # ---- 时间口径（严格按你的要求） ----
        if end_date is None:
            raise ValueError("end_date 是必填参数")
        self.end_date = end_date

        if start_date is not None:
            self.start_date = start_date
            self.window = _calc_window_from_dates(start_date, end_date) if window is None else int(window)
        else:
            if window is None:
                raise ValueError("start_date 和 window 不能同时为空；至少给一个")
            self.window = int(window)
            tds = get_trading_dates(end_date=self.end_date, window=self.window)
            if len(tds) == 0:
                raise ValueError("get_trading_dates 返回为空，请检查数据源或参数")
            self.start_date = pd.to_datetime(tds[0]).strftime("%Y-%m-%d")

        # 其他配置
        self.period = int(period)
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.k_hold = _holding_days(self.period, self.buy_price, self.sell_price)  # 核心：持有天数K

        self.codes = list(codes) if codes is not None else None
        self.factor = factor() if isinstance(factor, type) else factor

        self.factor_df = factor_df
        self.return_df = return_df
        self.merged: Optional[pd.DataFrame] = None

        self._cache: Dict[tuple, Any] = {}
        self._group_returns_cache: Dict[tuple, pd.DataFrame] = {}

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

        # 读文件优先
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

        print(f"[load factor] {len(out)} rows in {time.time()-t0:.3f}s")
        return out.reset_index(drop=True)

    def _load_return_df(self) -> pd.DataFrame:
        """
        生成 future_return —— 与你原口径保持一致（信号日的“period 总收益”）：
          - buy=close: buy_shift=0
          - buy=open/high/low: buy_shift=-1  (下一交易日开盘买)
          - sell 使用 shift(-period + buy_shift)
        """
        t0 = time.time()
        self._ensure_codes()

        daily = get_daily_data(self.codes, self.end_date, self.window).sort_values(
            ["stock_code", "trade_date"]
        )
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        mask = (daily["trade_date"] >= self.start_date) & (daily["trade_date"] <= self.end_date)
        daily = daily.loc[mask].copy()

        g = daily.groupby("stock_code", sort=False)

        if self.buy_price == "close":
            buy = g[self.buy_price].shift(0)
            buy_shift = 0
        elif _is_open_like(self.buy_price):
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

        print(f"[load return] {len(ret_df)} rows in {time.time()-t0:.3f}s")
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

    # ----------------- IC / RankIC -----------------
    def ic_series(self, method: str = "pearson", robust: bool = False) -> pd.Series:
        """按日横截面相关（Series，index=date）。默认与旧口径一致：robust=False。"""
        if self.merged is None:
            raise ValueError("data not ready")
        key = _config_key("ic_series", method, robust)
        if key in self._cache:
            return self._cache[key]

        corr_fn = _robust_corr if robust else (lambda a, b, m=method: _corr_safe(a, b, m))
        out = []
        for d, g in self.merged.groupby("date", sort=True):
            val = corr_fn(g["value"], g["future_return"], method)
            out.append((d, val))
        ic = pd.Series(dict(out)).sort_index().rename("IC")
        self._cache[key] = ic
        return ic

    def rank_ic_series(self, robust: bool = False) -> pd.Series:
        return self.ic_series(method="spearman", robust=robust).rename("RankIC")

    # ----------------- 分组工具 -----------------
    @staticmethod
    def _group_labels(n_groups: int) -> List[str]:
        labels = [f"Q{i+1}" for i in range(n_groups)]
        labels[0] = "Q1(最高)"
        labels[-1] = f"Q{n_groups}(最低)"
        return labels

    @staticmethod
    def _qcode_by_value(values: pd.Series, n_groups: int) -> pd.Series:
        """按值分位：0=最小 … n-1=最大。"""
        rk = values.rank(method="first")
        return pd.qcut(rk, n_groups, labels=False)

    # ----------------- 分组 period → (等效)日 → K桶叠加 → NAV -----------------
    def group_period_returns(self, n_groups: int = 5,
                             weight_type: str = "equal",
                             weight_col: Optional[str] = None) -> pd.DataFrame:
        """
        各组 **period 总收益**（信号日口径）。列从“高到低”命名：Q1(最高)…Qn(最低)
        """
        if self.merged is None:
            raise ValueError("data not ready")
        key = _config_key("group_period", n_groups, weight_type, weight_col)
        if key in self._cache:
            return self._cache[key]

        rows: Dict[int, list] = {}
        for d, g in self.merged.groupby("date", sort=True):
            q = self._qcode_by_value(g["value"], n_groups)
            g = g.assign(q=q)

            for i in range(n_groups):
                gi = g[g.q == i]
                if gi.empty:
                    continue
                if weight_type == "equal":
                    r = gi["future_return"].mean()
                elif weight_type == "weighted":
                    if weight_col is not None and weight_col in gi:
                        w = gi[weight_col].astype(float).abs()
                    else:
                        w = gi["value"].astype(float).abs()
                    w = w.replace(0, np.nan)
                    if w.notna().sum() == 0:
                        r = gi["future_return"].mean()
                    else:
                        w = w / w.sum()
                        r = (gi["future_return"] * w).sum()
                else:
                    raise ValueError("weight_type 必须是 'equal' 或 'weighted'")
                rows.setdefault(i, []).append((d, float(r)))

        labels = self._group_labels(n_groups)
        order_idx = list(range(n_groups - 1, -1, -1))  # n-1…0 => 高到低

        df_parts = {}
        for j, i in enumerate(order_idx):
            series = pd.Series({d: v for d, v in rows.get(i, [])}).sort_index()
            df_parts[labels[j]] = series

        df = pd.DataFrame(df_parts)
        df.index.name = "date"
        self._cache[key] = df
        return df

    def group_daily_returns(self, n_groups: int = 5,
                            weight_type: str = "equal",
                            weight_col: Optional[str] = None,
                            overlap_buckets: Optional[int] = None) -> pd.DataFrame:
        """
        各组 **日收益**：
          step1 折算：r_day = (1 + R_period)^(1/K) - 1
          step2 叠加：daily = r_day.rolling(K).mean()   （K=真实持有天数，或你指定的 overlap_buckets）
        """
        per = self.group_period_returns(n_groups, weight_type, weight_col)
        k = overlap_buckets if overlap_buckets is not None else self.k_hold
        r_day = _period_to_daily_equiv(per, k)
        daily = r_day.rolling(k, min_periods=1).mean()
        daily.index.name = "date"
        return daily

    def calc_group_returns(self, n_groups: int = 5,
                           weight_type: str = 'equal',
                           weight_col: Optional[str] = None,
                           overlap_buckets: Optional[int] = None) -> pd.DataFrame:
        """
        各组 **累计净值**（从 1 开始），列名：Q1(最高)…Qn(最低)
        """
        cache_key = _config_key("group_nav", n_groups, weight_type, weight_col,
                                overlap_buckets, self.k_hold)
        if cache_key in self._group_returns_cache:
            return self._group_returns_cache[cache_key]

        daily_ret = self.group_daily_returns(n_groups=n_groups,
                                             weight_type=weight_type,
                                             weight_col=weight_col,
                                             overlap_buckets=overlap_buckets)
        if daily_ret.empty:
            out = pd.DataFrame()
        else:
            out = (1 + daily_ret.fillna(0.0)).cumprod()

        self._group_returns_cache[cache_key] = out
        return out

    # ----------------- Long–Short：period → 日（等效）→ 叠加 -----------------
    def longshort_period(self, n_groups: int = 5) -> pd.Series:
        """
        多空 **period 总收益**：Top 组均值 - Bottom 组均值（信号日口径）
        """
        if self.merged is None:
            raise ValueError("data not ready")
        key = _config_key("ls_period", n_groups)
        if key in self._cache:
            return self._cache[key]

        rows = []
        for d, g in self.merged.groupby("date", sort=True):
            q = self._qcode_by_value(g["value"], n_groups)
            g = g.assign(q=q)
            top = g.loc[g.q == n_groups - 1, "future_return"].mean()
            bot = g.loc[g.q == 0, "future_return"].mean()
            rows.append((d, float(top - bot)))
        ls = pd.Series(dict(rows)).sort_index().rename("LS_period")
        self._cache[key] = ls
        return ls

    def longshort_daily(self, n_groups: int = 5,
                        overlap_buckets: Optional[int] = None) -> pd.Series:
        """
        多空 **日收益**（严格先折算再叠加）：
          r_day = (1 + LS_period)^(1/K) - 1   →   LS_daily = r_day.rolling(K).mean()
        """
        k = overlap_buckets if overlap_buckets is not None else self.k_hold
        ls_per = self.longshort_period(n_groups)
        r_day = _period_to_daily_equiv(ls_per, k)
        daily = r_day.rolling(k, min_periods=1).mean().rename("LS_daily")
        return daily

    # ----------------- TopN：period → 日（等效）→ 叠加 → NAV -----------------
    def topn_period_returns(self, top_n: int = 5,
                            weight_type: str = "equal",
                            weight_col: Optional[str] = None) -> pd.Series:
        """
        信号日选 TopN（因子最大），该笔交易的 **period 总收益**（与 future_return 口径一致）。
        """
        if self.merged is None:
            raise ValueError("data not ready")
        key = _config_key("topn_period", top_n, weight_type, weight_col)
        if key in self._cache:
            return self._cache[key]

        rows = []
        for d, g in self.merged.groupby("date", sort=True):
            top = g.nlargest(top_n, "value").copy()
            if top.empty:
                continue
            if weight_type == "equal":
                r = top["future_return"].mean()
            elif weight_type == "weighted":
                if weight_col is not None and weight_col in top:
                    w = top[weight_col].astype(float).abs()
                else:
                    w = top["value"].astype(float).abs()
                w = w.replace(0, np.nan)
                if w.notna().sum() == 0:
                    r = top["future_return"].mean()
                else:
                    w = w / w.sum()
                    r = (top["future_return"] * w).sum()
            else:
                raise ValueError("weight_type 必须是 'equal' 或 'weighted'")
            rows.append((d, float(r)))

        s = pd.Series(dict(rows)).sort_index().rename(f"Top{top_n}_{weight_type}_period")
        self._cache[key] = s
        return s

    def topn_daily_returns(self, top_n: int = 5,
                           weight_type: str = "equal",
                           weight_col: Optional[str] = None,
                           overlap_buckets: Optional[int] = None) -> pd.Series:
        """
        TopN **日收益**（严格先折算再叠加）：
          r_day = (1 + R_period)^(1/K) - 1   →   daily = r_day.rolling(K).mean()
        """
        k = overlap_buckets if overlap_buckets is not None else self.k_hold
        s = self.topn_period_returns(top_n, weight_type, weight_col)
        r_day = _period_to_daily_equiv(s, k)
        daily = r_day.rolling(k, min_periods=1).mean().rename(s.name.replace("_period", "_daily"))
        return daily

    def topn_nav(self, top_n: int = 5,
                 weight_type: str = "equal",
                 weight_col: Optional[str] = None,
                 overlap_buckets: Optional[int] = None) -> pd.Series:
        """
        TopN 组合 **净值**（从 1 开始）
        """
        daily = self.topn_daily_returns(top_n, weight_type, weight_col, overlap_buckets).fillna(0.0)
        nav = (1 + daily).cumprod().rename(daily.name.replace("_daily", "_NAV"))
        return nav

    # ----------------- 一站式日度指标（含累计） -----------------
    def daily_metrics(
        self,
        n_groups: int = 5,
        top_n: int = 5,
        ir_mode: str = "expanding",  # "expanding" | "rolling"
        ir_window: int = 60,
        robust: bool = False,
        overlap_buckets: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        汇总：IC / RankIC（横截面） + LS_daily（K 桶叠加）→ IR，含 TopN 的横截面 t 统计（基于 period 收益）。
        所有 *_cum 都是从同一首日开始累计，避免前段空值的问题。
        """
        key = _config_key("daily_metrics", n_groups, top_n, ir_mode, ir_window, robust,
                          overlap_buckets, self.k_hold)
        if key in self._cache:
            return self._cache[key].copy()

        ic  = self.ic_series(method="pearson", robust=robust)
        ric = self.rank_ic_series(robust=robust)

        # IR_from_IC / RankIR_from_RankIC —— 直接对 IC/RankIC 做 z-score
        if ir_mode == "expanding":
            ir_ic = _expanding_zscore(ic).rename("IR_from_IC")
            rir   = _expanding_zscore(ric).rename("RankIR_from_RankIC")
        elif ir_mode == "rolling":
            ir_ic = _rolling_zscore(ic, ir_window).rename("IR_from_IC")
            rir   = _rolling_zscore(ric, ir_window).rename("RankIR_from_RankIC")
        else:
            raise ValueError("ir_mode must be 'expanding' or 'rolling'")

        # LS 日收益（K 桶叠加）→ IR_LS
        ls_daily = self.longshort_daily(n_groups=n_groups, overlap_buckets=overlap_buckets)
        if ir_mode == "expanding":
            ir_ls = _expanding_zscore(ls_daily).rename("IR_LS")
        else:
            ir_ls = _rolling_zscore(ls_daily, ir_window).rename("IR_LS")

        # TopN 横截面 t 值（按 period 收益）
        t_rows = []
        for d, g in self.merged.groupby("date", sort=True):
            top = g.nlargest(top_n, "value")
            if len(top) >= 2:
                sd = top["future_return"].std(ddof=1)
                mu = top["future_return"].mean()
                tval = np.nan if (sd is None or sd == 0 or np.isnan(sd)) else np.sqrt(len(top)) * mu / sd
            else:
                tval = np.nan
            t_rows.append((d, float(tval) if not np.isnan(tval) else np.nan))
        tstat = pd.Series(dict(t_rows)).sort_index().rename("tstat_topN")

        df = pd.concat([ic, ric, ir_ic, rir, ls_daily, ir_ls, tstat], axis=1).sort_index()

        # 所有累计列：从同一首日开始累计（遇 NaN 视为 0 再累计），与你看到的他家口径一致
        for col in ["IC", "RankIC", "IR_from_IC", "RankIR_from_RankIC", "LS_daily", "IR_LS", "tstat_topN"]:
            df[f"{col}_cum"] = df[col].fillna(0.0).cumsum()

        self._cache[key] = df
        return df.copy()

    # ----------------- summary stats -----------------
    def ic_stats(self, robust: bool = False) -> pd.Series:
        ic = self.ic_series(method="pearson", robust=robust).dropna()
        if ic.empty:
            return pd.Series(dtype=float)
        n = len(ic); mean = ic.mean(); std = ic.std(ddof=1)
        pos_ratio = (ic > 0).mean()
        t_value = mean / (std / np.sqrt(n)) if std > 0 else np.nan
        return pd.Series({
            "mean": mean, "std": std, "IR": mean / std if std > 0 else np.nan,
            "t": t_value, "pos_ratio": pos_ratio, "max": ic.max(), "min": ic.min(),
            "p5": ic.quantile(0.05), "p95": ic.quantile(0.95), "count": n
        })

    def rank_ic_stats(self, robust: bool = False) -> pd.Series:
        ric = self.rank_ic_series(robust=robust).dropna()
        if ric.empty:
            return pd.Series(dtype=float)
        n = len(ric); mean = ric.mean(); std = ric.std(ddof=1)
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
        self._group_returns_cache.clear()

    def get_cache_info(self) -> Dict[str, int]:
        return {str(k): (isinstance(v, (pd.DataFrame, pd.Series)) and len(v) or 1)
                for k, v in self._cache.items()}
