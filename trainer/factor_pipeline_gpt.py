# factor_pipeline.py  — leakage-safe, dual-branch lag, post-merge split
from __future__ import annotations
import os, json, joblib, duckdb
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

# 你已有的工具函数：不要改名
from utils import get_trading_dates
from data_reader import get_daily_basic_data  # 已存在于你的工程
# make_future_return 用你的原实现（下方直接调用，不重写）
from factor_pipeline import make_future_return  # 如果你的项目里函数就在本文件, 改成相对引用

# ---------------------------
# 两个轻量工具：winsor & 标准化
# ---------------------------
class XWinsorizer:
    """按列全局（训练段）winsor，再应用到 val/test；默认不启用。"""
    def __init__(self, limits: Optional[Tuple[float, float]] = None):
        self.limits = limits
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        if not self.limits:
            return self
        lo, hi = self.limits
        self.bounds_ = {}
        for c in feature_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            self.bounds_[c] = (s.quantile(lo), s.quantile(hi))
        return self

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        if not self.limits:
            return df
        out = df.copy()
        for c in feature_cols:
            if c in self.bounds_:
                lo, hi = self.bounds_[c]
                out[c] = pd.to_numeric(out[c], errors="coerce").clip(lo, hi)
        return out


class TSScalerPerCode:
    """仅用训练段拟合，按 code 做时序缩放（StandardScaler 或 RobustScaler）。"""
    def __init__(self, kind: Optional[str] = "standard"):
        self.kind = kind  # "standard"|"robust"|None
        self.scalers_: Dict[str, object] = {}

    def _new(self):
        from sklearn.preprocessing import StandardScaler, RobustScaler
        return StandardScaler() if self.kind == "standard" else RobustScaler()

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        if not self.kind:
            return self
        self.scalers_.clear()
        for code, g in tqdm(df.groupby("code", sort=False), desc="TSScaler.fit"):
            s = self._new()
            s.fit(pd.to_numeric(g[feature_cols], errors="coerce").values)
            self.scalers_[code] = s
        return self

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        if not self.kind:
            return df
        outs = []
        for code, g in df.groupby("code", sort=False):
            s = self.scalers_.get(code)
            if s is None:
                outs.append(g)  # 训练里没见过的 code，保持原样
            else:
                g2 = g.copy()
                g2[feature_cols] = s.transform(pd.to_numeric(g2[feature_cols], errors="coerce").values)
                outs.append(g2)
        return pd.concat(outs, ignore_index=True)


def xs_zscore_per_date(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """当日横截面 z-score，不涉及“fit”（当天已知信息）。"""
    out = df.copy()
    grp = out.groupby("date")[feature_cols]
    mu = grp.transform("mean")
    sd = grp.transform("std").replace(0, np.nan)
    out[feature_cols] = (out[feature_cols] - mu) / sd
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# ---------------------------
# 因子与 daily_basic 两路：只做 lag，不做缩放
# ---------------------------
def _add_lags(df: pd.DataFrame, feature_cols: List[str], lag_days: List[int]) -> pd.DataFrame:
    if not lag_days or lag_days == [0]:
        return df
    lag_days = sorted(set([l for l in lag_days if l > 0]))
    if not lag_days:
        return df
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    pieces = [df]
    for lag in lag_days:
        shifted = (
            df.groupby("code", sort=False)[feature_cols]
              .shift(lag)
              .add_suffix(f"_lag{lag}")
        )
        pieces.append(shifted)
    out = pd.concat(pieces, axis=1)
    return out


class FactorBlock:
    """因子块：读取 → 只做 lag（可选 winsor_x，但默认 None）"""
    def __init__(self, factor_dir: str, factor_names: Optional[List[str]] = None,
                 lag_days: List[int] = [0], winsor_x: Optional[Tuple[float,float]] = None):
        self.factor_dir = Path(factor_dir)
        self.factor_names = factor_names
        self.lag_days = lag_days
        self.winsor = XWinsorizer(winsor_x) if winsor_x else None
        self.feature_cols_: List[str] = []

    def _load(self, start: str, end: str) -> pd.DataFrame:
        files = sorted(self.factor_dir.glob("*.parquet"))
        if self.factor_names:
            names = set(self.factor_names)
            files = [f for f in files if f.stem in names]
        if not files:
            raise FileNotFoundError("未找到因子 parquet 文件")
        files_str = str([str(p) for p in files]).replace("'", '"')
        con = duckdb.connect()
        df = con.execute(f"""
            SELECT * FROM (
                SELECT code, date, factor, value
                FROM read_parquet({files_str})
                WHERE date BETWEEN '{start}' AND '{end}'
            )
            PIVOT (MAX(value) FOR factor IN (SELECT DISTINCT factor FROM read_parquet({files_str})))
        """).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def build(self, start: str, end: str) -> pd.DataFrame:
        base = self._load(start, end).sort_values(["code","date"]).reset_index(drop=True)
        all_cols = [c for c in base.columns if c not in ("code","date")]
        self.feature_cols_ = all_cols[:]  # 原始列名
        # lag（仅对因子列）
        out = _add_lags(base, all_cols, self.lag_days)
        # winsor_x（可选；仅对因子与其 lag 列）
        feat_cols = [c for c in out.columns if c not in ("code","date")]
        if self.winsor:
            # 注意：winsor 的“fit”放到 Pipeline 的 train split 上做
            self._all_feat_cols = feat_cols  # 暂存，Pipeline 内部再 fit
        return out


class BasicBlock:
    """daily_basic 块：读取 → 只做它自己的 lag（不缩放）"""
    def __init__(self, lag_days: List[int] = [0], rename_prefix: str = "basic_"):
        self.lag_days = lag_days
        self.rename_prefix = rename_prefix
        self.feature_cols_: List[str] = []

    def build(self, codes: List[str], end: str, window: int) -> pd.DataFrame:
        raw = get_daily_basic_data(codes, end, window)
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["code","date"])
        df = raw.rename(columns={"stock_code":"code","trade_date":"date"}).copy()
        df["date"] = pd.to_datetime(df["date"])
        keep = [c for c in df.columns if c not in ("code","date")]
        lagged = _add_lags(df[["code","date"]+keep], keep, self.lag_days)
        # 重命名避免与因子重名
        rename = {c: f"{self.rename_prefix}{c}" for c in lagged.columns if c not in ("code","date")}
        lagged = lagged.rename(columns=rename)
        self.feature_cols_ = [c for c in lagged.columns if c not in ("code","date")]
        return lagged


# ---------------------------
# 总 Pipeline（先合并 → 再切分 → 再缩放）
# ---------------------------
class FactorPipeline:
    def __init__(
        self,
        factor_dir: str,
        factor_names: Optional[List[str]] = None,
        start: str = "2018-01-01",
        end: str   = datetime.today().strftime("%Y-%m-%d"),

        # 两路（因子 / daily_basic）各自的 lag
        factor_lag_days: List[int] = [0],
        basic_lag_days:  List[int] = [0],

        # 训练段专用的 X 剪尾与时序缩放
        winsor_x: Optional[Tuple[float,float]] = None,   # None 表示不剪尾
        ts_scale: Optional[str] = "standard",             # "standard"|"robust"|None

        # 当日横截面标准化（不需要 fit）
        xs_norm: bool = True,

        # 切分
        train_ratio: float = 0.8,
        val_ratio:   float = 0.1,
        random_state: int  = 42,

        # 未来收益口径（不改你的 make_future_return）
        period: int = 1,
        buy: str = "open",
        sell: str = "open",

        # 价格读取器：callable(codes, end_date, window)->pd.DataFrame with OHLCV 等
        price_loader=None,

        # 缺失填充
        fillna: bool = False,
    ):
        self.factor_dir = factor_dir
        self.factor_names = factor_names
        self.start, self.end = start, end

        self.factor_lag_days = factor_lag_days
        self.basic_lag_days  = basic_lag_days

        self._winsor = XWinsorizer(winsor_x) if winsor_x else None
        self._ts_scaler = TSScalerPerCode(ts_scale)
        self._use_xs_norm = xs_norm

        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.random_state = random_state

        self.period, self.buy, self.sell = period, buy, sell
        if price_loader is None:
            raise ValueError("price_loader 不能为空：需要提供 callable(codes, end_date, window)->DataFrame")
        self.price_loader = price_loader

        self.fillna = fillna

        # 构建后的数据与特征清单
        self.df: Optional[pd.DataFrame] = None
        self.feature_cols_: List[str] = []

        # 运行中缓存
        self._dates_all: List[str] = get_trading_dates(self.start, self.end)

        # 两个 block
        self._fac_blk = FactorBlock(factor_dir=self.factor_dir,
                                    factor_names=self.factor_names,
                                    lag_days=self.factor_lag_days,
                                    winsor_x=None)  # 注意：winsor 在切分后、训练段 fit
        self._bsc_blk = BasicBlock(lag_days=self.basic_lag_days, rename_prefix="basic_")

    # --------- 工具：按日期切分（先合并后切分） ---------
    def _split_dates(self, uniq_dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        uniq_dates = np.array(sorted(pd.to_datetime(uniq_dates).astype("datetime64[D]").astype(str)))
        n = len(uniq_dates)
        if n < 10:
            raise ValueError("有效交易日太少，无法切分")
        tr_n = int(n * self.train_ratio)
        va_n = int(n * (self.train_ratio + self.val_ratio))
        train_dates = uniq_dates[:tr_n]
        val_dates   = uniq_dates[tr_n:va_n]
        test_dates  = uniq_dates[va_n:]
        return train_dates, val_dates, test_dates

    # --------- 主流程：构建 df & 三段数据 ---------
    def build_dataset(self):
        # 1) 读取两路特征（各自只做 lag，不做缩放）
        fac_df = self._fac_blk.build(self.start, self.end)  # 包含因子 & 因子 lag
        if fac_df.empty:
            raise RuntimeError("因子数据为空")
        codes = fac_df["code"].astype(str).unique().tolist()

        # daily_basic（其自己的 lag），再与因子左连接
        window = len(self._dates_all) + self.period + max(0, *(self.factor_lag_days or [0]), *(self.basic_lag_days or [0]))
        bsc_df = self._bsc_blk.build(codes=codes, end=self.end, window=window)
        if not bsc_df.empty:
            feat_df = fac_df.merge(bsc_df, on=["code","date"], how="left")
        else:
            feat_df = fac_df.copy()

        # 2) 未来收益（只根据价格，避免泄露）
        price_df = self.price_loader(codes, self.end, window)
        ret_df = make_future_return(price_df, period=self.period, buy_price=self.buy, sell_price=self.sell)

        # 3) 合并 y
        full = feat_df.merge(ret_df, on=["code","date"], how="inner").sort_values(["date","code"]).reset_index(drop=True)
        full["date"] = pd.to_datetime(full["date"])

        # 4) 切分（合并完后再切分，确保两路对齐且无重复 lag）
        uniq_dates = full["date"].drop_duplicates().values
        train_dates, val_dates, test_dates = self._split_dates(uniq_dates)

        df_tr = full[full["date"].isin(train_dates)].copy()
        df_va = full[full["date"].isin(val_dates)].copy()
        df_te = full[full["date"].isin(test_dates)].copy()

        # 5) 列出特征列（此刻已经一次性确定，确保后续 transform 顺序一致）
        self.feature_cols_ = [c for c in df_tr.columns if c not in ("code","date","future_return")]

        # 6) 训练段 fit：winsor（可选）+ TS 缩放
        if self._winsor:
            self._winsor.fit(df_tr, self.feature_cols_)
            df_tr = self._winsor.transform(df_tr, self.feature_cols_)
            df_va = self._winsor.transform(df_va, self.feature_cols_)
            df_te = self._winsor.transform(df_te, self.feature_cols_)

        # TS 缩放只用训练段拟合
        self._ts_scaler.fit(df_tr, self.feature_cols_)
        df_tr = self._ts_scaler.transform(df_tr, self.feature_cols_)
        df_va = self._ts_scaler.transform(df_va, self.feature_cols_)
        df_te = self._ts_scaler.transform(df_te, self.feature_cols_)

        # 7) 当日横截面标准化（不需要 fit；在各自 split 内按当日做 z-score）
        if self._use_xs_norm:
            df_tr = xs_zscore_per_date(df_tr, self.feature_cols_)
            df_va = xs_zscore_per_date(df_va, self.feature_cols_)
            df_te = xs_zscore_per_date(df_te, self.feature_cols_)

        # 8) 缺失处理
        if self.fillna:
            for d in (df_tr, df_va, df_te):
                d[self.feature_cols_] = d[self.feature_cols_].fillna(0.0)

        # 9) 汇总输出
        df_tr["split"] = "train"; df_va["split"] = "val"; df_te["split"] = "test"
        self.df = pd.concat([df_tr, df_va, df_te], ignore_index=True).sort_values(["date","code"]).reset_index(drop=True)

        print(f"[FactorPipeline] build done: train={len(df_tr)}, val={len(df_va)}, test={len(df_te)}, feats={len(self.feature_cols_)}")
        return self

    # --------- 取 X/y ---------
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.df is None: raise RuntimeError("先调用 build_dataset()")
        df = self.df[self.df["split"]=="train"]
        X = df[self.feature_cols_].values
        y = df["future_return"].values
        return X, y

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.df is None: raise RuntimeError("先调用 build_dataset()")
        df = self.df[self.df["split"]=="val"]
        X = df[self.feature_cols_].values
        y = df["future_return"].values
        return X, y

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.df is None: raise RuntimeError("先调用 build_dataset()")
        df = self.df[self.df["split"]=="test"]
        X = df[self.feature_cols_].values
        y = df["future_return"].values
        return X, y

    def get_all_splits(self):
        if self.df is None: raise RuntimeError("先调用 build_dataset()")
        tr = self.df[self.df["split"]=="train"].copy()
        va = self.df[self.df["split"]=="val"].copy()
        te = self.df[self.df["split"]=="test"].copy()
        return {
            "train": (tr[self.feature_cols_].values, tr["future_return"].values),
            "val":   (va[self.feature_cols_].values, va["future_return"].values),
            "test":  (te[self.feature_cols_].values, te["future_return"].values),
            "train_df": tr, "val_df": va, "test_df": te
        }

    def get_preprocessed_data(self) -> pd.DataFrame:
        if self.df is None: raise RuntimeError("先调用 build_dataset()")
        return self.df.copy()

    # --------- 推理：未来日期范围（仅 transform，不做任何 fit）---------
    def transform_future_factors(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """读取未来区间的因子与 basic（各自 lag），然后用已拟合的 winsor/ts_scale 进行 transform，再做（可选）当日 z-score。"""
        if start_date is None: start_date = self.start
        if end_date   is None: end_date   = self.end

        fac_df = self._fac_blk.build(start_date, end_date)
        if fac_df.empty:
            return pd.DataFrame(columns=["code","date"])

        codes = fac_df["code"].astype(str).unique().tolist()
        # 取足够窗口，保证 basic lag 可构造
        all_dates = get_trading_dates(start_date, end_date)
        window = len(all_dates) + self.period + max(0, *(self.factor_lag_days or [0]), *(self.basic_lag_days or [0]))
        bsc_df = self._bsc_blk.build(codes=codes, end=end_date, window=window)

        if not bsc_df.empty:
            df = fac_df.merge(bsc_df, on=["code","date"], how="left")
        else:
            df = fac_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        feat_cols = [c for c in df.columns if c not in ("code","date")]

        # 训练时拟合过的 winsor / ts_scale
        if self._winsor:
            # 注意：transform 只剪到训练段保存的边界
            df = self._winsor.transform(df, feat_cols)
        df = self._ts_scaler.transform(df, feat_cols)

        if self._use_xs_norm:
            df = xs_zscore_per_date(df, feat_cols)

        if self.fillna:
            df[feat_cols] = df[feat_cols].fillna(0.0)
        return df

    def predict_range(self, model, start_date: Optional[str]=None, end_date: Optional[str]=None) -> pd.DataFrame:
        """给定已训练模型，对未来区间做预测（仅依赖特征，不需要价格）。"""
        df = self.transform_future_factors(start_date, end_date)
        if df.empty:
            return pd.DataFrame(columns=["code","date","pred"])
        # 对齐训练时的列顺序（若未来出现新列，忽略；缺列补 0）
        f_now = [c for c in df.columns if c not in ("code","date")]
        missing = [c for c in self.feature_cols_ if c not in f_now]
        extra   = [c for c in f_now if c not in self.feature_cols_]
        use_df = df[["code","date"] + [c for c in self.feature_cols_ if c in f_now]].copy()
        for m in missing:
            use_df[m] = 0.0
        use_df = use_df[["code","date"] + self.feature_cols_]

        X = use_df[self.feature_cols_].values
        use_df["pred"] = model.predict(X)
        return use_df[["code","date","pred"]].sort_values(["date","code"]).reset_index(drop=True)

    # --------- 保存 / 加载（恢复可复现状态）---------
    def save(self, path: str):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        meta = {
            "factor_dir": self.factor_dir,
            "factor_names": self.factor_names,
            "start": self.start, "end": self.end,
            "factor_lag_days": self.factor_lag_days,
            "basic_lag_days": self.basic_lag_days,
            "winsor_x": self._winsor.limits if self._winsor else None,
            "ts_scale": self._ts_scaler.kind,
            "xs_norm": self._use_xs_norm,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "random_state": self.random_state,
            "period": self.period, "buy": self.buy, "sell": self.sell,
            "fillna": self.fillna,
            "feature_cols_": self.feature_cols_,
        }
        with open(path/"pipeline_meta.json","w",encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # scalers & winsor 边界
        if self._winsor:
            joblib.dump(self._winsor.bounds_, path/"winsor_bounds.joblib")
        else:
            if (path/"winsor_bounds.joblib").exists():
                os.remove(path/"winsor_bounds.joblib")

        joblib.dump(self._ts_scaler.scalers_, path/"ts_scalers.joblib")

        # 数据副本（可选）
        if self.df is not None:
            self.df.to_parquet(path/"dataset.parquet", index=False)

        print(f"[FactorPipeline] saved to {path}")

    @classmethod
    def load(cls, path: str, price_loader):
        path = Path(path)
        with open(path/"pipeline_meta.json","r",encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls(
            factor_dir=meta["factor_dir"],
            factor_names=meta["factor_names"],
            start=meta["start"], end=meta["end"],
            factor_lag_days=meta["factor_lag_days"],
            basic_lag_days=meta["basic_lag_days"],
            winsor_x=meta["winsor_x"],
            ts_scale=meta["ts_scale"],
            xs_norm=meta["xs_norm"],
            train_ratio=meta["train_ratio"],
            val_ratio=meta["val_ratio"],
            random_state=meta["random_state"],
            period=meta["period"], buy=meta["buy"], sell=meta["sell"],
            price_loader=price_loader,
            fillna=meta["fillna"],
        )
        obj.feature_cols_ = meta.get("feature_cols_", [])

        # 恢复 scalers & winsor
        if (path/"winsor_bounds.joblib").exists() and obj._winsor:
            obj._winsor.bounds_ = joblib.load(path/"winsor_bounds.joblib")
        if (path/"ts_scalers.joblib").exists():
            obj._ts_scaler.scalers_ = joblib.load(path/"ts_scalers.joblib")

        # 恢复数据
        if (path/"dataset.parquet").exists():
            obj.df = pd.read_parquet(path/"dataset.parquet")
        print(f"[FactorPipeline] loaded from {path}")
        return obj
