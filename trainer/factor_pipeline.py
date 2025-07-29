from __future__ import annotations
import os, json, joblib, duckdb
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from utils import get_trading_dates
# ---------- 1. Loader ----------
class FactorLoader:
    def __init__(
        self,
        factor_dir: str,
        factor_names: Optional[List[str]] = None,
        start: Optional[str] = "2000-01-01",
        end  : Optional[str] = datetime.today().strftime("%Y-%m-%d")
    ):
        self.factor_dir = Path(factor_dir)
        self.factor_names = factor_names  # None => 全部
        self.start, self.end = pd.to_datetime(start), pd.to_datetime(end)

    def load(self) -> pd.DataFrame:
        """duckdb‑pivot 一次成表"""
        files = sorted(self.factor_dir.glob("*.parquet"))
        if self.factor_names:
            files = [f for f in files if f.stem in self.factor_names]
        if not files:
            raise FileNotFoundError("未找到因子 parquet 文件")

        files_str = str([str(p) for p in files]).replace("'", '"')
        con = duckdb.connect()
        print(f"[FactorLoader] 读取 {len(files)} 个因子文件，Pivot …")
        sql = f"""
        WITH base AS (
            SELECT code, date, factor, value
            FROM read_parquet({files_str})
            WHERE date BETWEEN '{self.start.date()}' AND '{self.end.date()}'
        )
        SELECT *
        FROM base
        PIVOT ( MAX(value) FOR factor IN (SELECT DISTINCT factor FROM base) )
        """
        df = con.execute(sql).df()
        df["date"] = pd.to_datetime(df["date"])
        return df


# ---------- 2. Return ----------
def make_future_return(
    daily: pd.DataFrame,
    period: int = 1,
    buy_price: str = "open",
    sell_price: str = "open"
) -> pd.DataFrame:
    daily = daily.sort_values(["stock_code", "trade_date"]).copy()
    g = daily.groupby("stock_code", sort=False)

    buy_shift = 0 if buy_price == "close" else -1
    buy = g[buy_price].shift(buy_shift)
    sell = g[sell_price].shift(-period + buy_shift)
    ret = (sell - buy) / buy

    out = daily[["stock_code", "trade_date"]].copy()
    out["future_return"] = ret.values
    return out.dropna().rename(columns={"stock_code":"code", "trade_date":"date"})


# ---------- 3. Preprocessor ----------
class Preprocessor:
    def __init__(
        self,
        lag_days: List[int] = (0,),
        winsor: Tuple[float,float] | None = (0.01, 0.99),
        xs_norm: Optional[str] = "zscore",    # 横截面：zscore | robust | minmax | None
        ts_scale: Optional[str] = "standard"  # 时序：standard | robust | None
    ):
        self.lag_days, self.winsor, self.xs_norm, self.ts_scale = lag_days, winsor, xs_norm, ts_scale
        self.scalers: Dict[str, object] = {}   # 存每支股票每列 scaler
    
    # ----- static helpers -----
    @staticmethod
    def _winsorize(s: pd.Series, limits: Tuple[float,float]) -> pd.Series:
        lower, upper = s.quantile(limits[0]), s.quantile(limits[1])
        return s.clip(lower, upper)

    # ----- fit_transform -----
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["code","date"]).reset_index(drop=True)
        fac_cols = [c for c in df.columns if c not in ("code","date")]

        # 1. winsor
        if self.winsor:
            for c in fac_cols:
                df[c] = self._winsorize(df[c], self.winsor)

        # 2. lag
        for lag in self.lag_days:
            if lag>0:
                for c in fac_cols:
                    df[f"{c}_lag{lag}"] = df.groupby("code")[c].shift(lag)
        fac_cols = [c for c in df.columns if c not in ("code","date")]

        # 3. cross‑section normalize 日度
        if self.xs_norm:
            for d,g in tqdm(df.groupby("date"), desc="XS‑norm"):
                idx = g.index
                if self.xs_norm=="zscore":
                    df.loc[idx, fac_cols] = (g[fac_cols]-g[fac_cols].mean())/g[fac_cols].std(ddof=0)
                elif self.xs_norm=="robust":
                    df.loc[idx, fac_cols] = (g[fac_cols]-g[fac_cols].median())/g[fac_cols].mad()
                elif self.xs_norm=="minmax":
                    df.loc[idx, fac_cols] = (g[fac_cols]-g[fac_cols].min())/(g[fac_cols].max()-g[fac_cols].min())

        # 4. time‑series scaler per stock
        if self.ts_scale is not None:
            scaler_cls = StandardScaler if self.ts_scale=="standard" else RobustScaler
            out = []
            for code,g in tqdm(df.groupby("code", sort=False), desc="TS‑scale"):
                g2 = g.copy()
                scaler = scaler_cls().fit(g2[fac_cols])
                g2[fac_cols] = scaler.transform(g2[fac_cols])
                self.scalers[code] = scaler
                out.append(g2)
            df = pd.concat(out, ignore_index=True)

        return df.dropna().reset_index(drop=True)

    # ----- transform（预测时用） -----
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        fac_cols = [c for c in df.columns if c not in ("code","date")]
        if self.ts_scale:
            out = []
            for code,g in df.groupby("code", sort=False):
                scaler = self.scalers.get(code)
                if scaler is None:
                    out.append(g)  # 未见过的股票，直接原值
                else:
                    g2 = g.copy()
                    g2[fac_cols] = scaler.transform(g2[fac_cols])
                    out.append(g2)
            df = pd.concat(out, ignore_index=True)
        return df


# ---------- 4. Pipeline ----------
class FactorPipeline:
    def __init__(
        self,
        factor_dir: str,
        price_loader,      # callable(codes, end, window)->pd.DataFrame
        factor_names: Optional[List[str]] = None,
        period:int=2, buy:str="open", sell:str="open",
        start:str="2018-01-01", end:str=datetime.today().strftime("%Y-%m-%d"),
        lag_days:List[int]=[0], winsor=(0.01,0.99), xs_norm="zscore",
        ts_scale="standard",
        train_ratio=0.6,val_ratio=0.2,random_state=42
    ):
        self.factor_dir, self.factor_names = factor_dir, factor_names
        self.period, self.buy, self.sell = period, buy, sell
        self.start, self.end = start, end
        self.prep = Preprocessor(lag_days, winsor, xs_norm, ts_scale)
        self.train_ratio, self.val_ratio = train_ratio, val_ratio
        self.random_state = random_state
        self.price_loader = price_loader  # 注入型 (方便单测或替换)
        self.window = len(get_trading_dates(self.start, self.end)) + self.period + max(self.prep.lag_days)

        # 容器
        self.df: pd.DataFrame|None = None
        self.model: XGBRegressor|None = None

    # =========== 数据 ===========
    def build_dataset(self):
        # 1. 因子表
        fac_df = FactorLoader(self.factor_dir, self.factor_names,
                              self.start, self.end).load()

        # 2. 收益
        codes = fac_df["code"].unique().tolist()
        daily = self.price_loader(codes, self.end, self.window)
        ret_df = make_future_return(daily, self.period, self.buy, self.sell)

        # 3. merge
        merged = fac_df.merge(ret_df, on=["code","date"], how="inner")
        # 4. preprocess
        self.df = self.prep.fit_transform(merged)
        print(f"[Pipeline] 最终行数 {len(self.df)}, 特征 {self.df.shape[1]-3}")

    # =========== 划分 ===========
    def _split(self):
        if self.df is None: raise RuntimeError("先 build_dataset()")
        df = self.df.sort_values("date")
        n = len(df)
        tr=int(n*self.train_ratio); va=int(n*(self.train_ratio+self.val_ratio))
        return df.iloc[:tr], df.iloc[tr:va], df.iloc[va:]

    # =========== 训练 ===========
    def fit(self, **xgb_params):
        train,val,_ = self._split()
        X_tr,y_tr = self._xy(train); X_va,y_va = self._xy(val)

        default = dict(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1
        ); default.update(xgb_params)
        self.model = XGBRegressor(**default)
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_va,y_va)],
            eval_metric="rmse",
            verbose=False,
            early_stopping_rounds=50
        )
        print(f"[XGB] best_iteration={self.model.best_iteration_}  best_score={self.model.best_score}")

    # =========== 评估 ===========
    def evaluate(self):
        _,_,test = self._split()
        X_t,y_t = self._xy(test)
        preds = self.model.predict(X_t)
        print(f"Test RMSE {mean_squared_error(y_t,preds,squared=False):.4e}  R2 {r2_score(y_t,preds):.3f}")

    # =========== 预测 ===========
    def predict_range(
        self,
        factor_parquet_paths: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        给定若干因子 parquet 文件，读取指定日期范围内数据并进行预测

        Parameters:
            factor_parquet_paths: 每个因子的 parquet 文件路径列表
            start_date, end_date: 时间范围，若不指定默认取所有日期

        Returns:
            DataFrame 包含 code, date, pred
        """
        con = duckdb.connect()

        # 日期过滤条件
        date_filter = ""
        if start_date and end_date:
            date_filter = f"WHERE date >= '{start_date}' AND date <= '{end_date}'"
        elif start_date:
            date_filter = f"WHERE date >= '{start_date}'"
        elif end_date:
            date_filter = f"WHERE date <= '{end_date}'"

        files_str = str(factor_parquet_paths).replace("'", '"')

        # 构建宽表
        sql = f"""
            SELECT * FROM (
                SELECT code, date, factor, value
                FROM read_parquet({files_str})
                {date_filter}
            )
            PIVOT (
                MAX(value) FOR factor IN (
                    SELECT DISTINCT factor FROM read_parquet({files_str})
                )
            )
        """
        df = con.execute(sql).df()
        df["date"] = pd.to_datetime(df["date"])

        # 标准化预处理（transform，不要再 fit）
        df_prep = self.prep.transform(df)  # 你之前已经保存了 self.prep.scalers

        # 预测
        X = df_prep.drop(columns=["code", "date"]).values
        df_prep["pred"] = self.model.predict(X)

        return df_prep[["code", "date", "pred"]]

    # =========== 保存 / 载入 ===========
    def save(self, path: str):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path/"model.joblib")
        with open(path/"prep.json","w",encoding="utf8") as f:
            json.dump({"lag":self.prep.lag_days,
                       "winsor":self.prep.winsor,
                       "xs_norm":self.prep.xs_norm,
                       "ts_scale":self.prep.ts_scale}, f, ensure_ascii=False, indent=2)
        joblib.dump(self.prep.scalers, path/"scalers.joblib")
        print(f"[Pipeline] 已保存到 {path}")

    # -------------- utils --------------
    def _xy(self, df: pd.DataFrame) -> Tuple[np.ndarray,np.ndarray]:
        X = df.drop(columns=["code","date","future_return"]).values
        y = df["future_return"].values
        return X,y
