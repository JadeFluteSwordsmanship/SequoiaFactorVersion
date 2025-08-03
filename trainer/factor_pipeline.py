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
        
        # 先获取所有因子名称
        factor_list = con.execute(f"""
            SELECT DISTINCT factor FROM read_parquet({files_str})
            WHERE date BETWEEN '{self.start.date()}' AND '{self.end.date()}'
        """).fetchall()
        
        if not factor_list:
            return pd.DataFrame()
        
        # 构造 PIVOT 的列名
        factor_names = [f"'{row[0]}'" for row in factor_list]
        pivot_cols = ', '.join(factor_names)
        
        sql = f"""
            SELECT * FROM (
                SELECT code, date, factor, value
                FROM read_parquet({files_str})
                WHERE date BETWEEN '{self.start.date()}' AND '{self.end.date()}'
            )
            PIVOT ( MAX(value) FOR factor IN ({pivot_cols}) )
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
        ts_scale_type: Optional[str] = "standard"  # 时序：standard | robust | None
    ):
        self.lag_days, self.winsor, self.xs_norm, self.ts_scale_type = lag_days, winsor, xs_norm, ts_scale_type
        self.scalers: Dict[str, object] = {}   # 存每支股票每列 scaler
        
        # 存储训练时的统计信息，避免信息泄露
        self.winsor_limits: Dict[str, Tuple[float, float]] = {}  # 每列的winsor上下限
        self.factor_columns: List[str] = []  # 训练时的因子列名
    
    # ----- static helpers -----
    @staticmethod
    def _winsorize(s: pd.Series, limits: Tuple[float,float]) -> pd.Series:
        lower, upper = s.quantile(limits[0]), s.quantile(limits[1])
        return s.clip(lower, upper)

    # ----- 拆分的小函数 -----
    def winsorize(self, df: pd.DataFrame, is_fit: bool = True) -> pd.DataFrame:
        """去极值处理"""
        if not self.winsor:
            return df
            
        fac_cols = [c for c in df.columns if c not in ("code","date")]
        df = df.copy()
        
        if is_fit:
            # 训练时：计算并存储每列的上下限
            for c in fac_cols:
                lower, upper = df[c].quantile(self.winsor[0]), df[c].quantile(self.winsor[1])
                self.winsor_limits[c] = (lower, upper)
                df[c] = df[c].clip(lower, upper)
        else:
            # 预测时：使用训练时的上下限
            for c in fac_cols:
                if c in self.winsor_limits:
                    lower, upper = self.winsor_limits[c]
                    df[c] = df[c].clip(lower, upper)
                # 如果新列不在训练时，跳过（保持原值）
        print("完成去极值处理")
        return df

    def add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """安全的 lag 特征生成，处理缺失数据"""
        if not self.lag_days or self.lag_days == [0]:
            return df

        fac_cols = [c for c in df.columns if c not in ("code", "date")]
        lag_days = [l for l in self.lag_days if l > 0]
        if not lag_days:
            return df

        # 按股票分组处理，避免 reshape 问题
        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        
        lag_features = {}
        for lag in lag_days:
            for col in fac_cols:
                lag_features[f"{col}_lag{lag}"] = []
        
        # 按股票分组，每只股票单独处理
        for code, group in df.groupby("code"):
            group = group.sort_values("date").reset_index(drop=True)
            
            for lag in lag_days:
                for col in fac_cols:
                    # 使用 shift 生成 lag 特征
                    lag_col = group[col].shift(lag)
                    lag_features[f"{col}_lag{lag}"].extend(lag_col.tolist())
        
        # 创建 lag DataFrame
        lag_df = pd.DataFrame(lag_features)
        lag_df.index = df.index
        
        # 合并原始数据和 lag 特征
        df = pd.concat([df, lag_df], axis=1)
        print("完成滞后特征添加")
        return df

    def xs_normalize(self, df: pd.DataFrame, is_fit: bool = True) -> pd.DataFrame:
        """矢量化 cross‑section 标准化（date 分组）"""
        if not self.xs_norm:
            return df

        fac_cols = [c for c in df.columns if c not in ("code", "date")]

        if self.xs_norm == "zscore":
            means = df.groupby("date")[fac_cols].transform("mean")
            stds  = df.groupby("date")[fac_cols].transform("std").replace(0, np.nan)
            df[fac_cols] = (df[fac_cols] - means) / stds

        elif self.xs_norm == "robust":
            med  = df.groupby("date")[fac_cols].transform("median")
            mad  = df.groupby("date")[fac_cols].transform(
                lambda x: (x - x.median()).abs().mean()
            ).replace(0, np.nan)
            df[fac_cols] = (df[fac_cols] - med) / mad

        elif self.xs_norm == "minmax":
            mins = df.groupby("date")[fac_cols].transform("min")
            maxs = df.groupby("date")[fac_cols].transform("max")
            df[fac_cols] = (df[fac_cols] - mins) / (maxs - mins).replace(0, np.nan)

        df[fac_cols] = df[fac_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        print("完成横截面标准化")
        return df

    def ts_scale(self, df: pd.DataFrame, is_fit: bool = True) -> pd.DataFrame:
        """时序标准化"""
        if not self.ts_scale_type:
            return df
            
        fac_cols = [c for c in df.columns if c not in ("code","date")]
        
        if is_fit:
            # 训练时：为每支股票训练scaler
            scaler_cls = StandardScaler if self.ts_scale_type == "standard" else RobustScaler
            out = []
            for code, g in tqdm(df.groupby("code", sort=False), desc="TS-scale"):
                g2 = g.copy()
                scaler = scaler_cls().fit(g2[fac_cols])
                g2[fac_cols] = scaler.transform(g2[fac_cols])
                self.scalers[code] = scaler
                out.append(g2)
            df = pd.concat(out, ignore_index=True)
        else:
            # 预测时：使用训练时的scaler
            out = []
            for code, g in df.groupby("code", sort=False):
                scaler = self.scalers.get(code)
                if scaler is None:
                    out.append(g)  # 未见过的股票，直接原值
                else:
                    g2 = g.copy()
                    g2[fac_cols] = scaler.transform(g2[fac_cols])
                    out.append(g2)
            df = pd.concat(out, ignore_index=True)
        print("完成时序标准化")
        return df

    # ----- fit_transform -----
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["code","date"]).reset_index(drop=True)
        
        # 存储训练时的因子列名
        self.factor_columns = [c for c in df.columns if c not in ("code","date")]
        
        # 1. winsor
        df = self.winsorize(df, is_fit=True)
        
        # 2. lag
        df = self.add_lags(df)
        
        # 3. cross‑section normalize 日度（使用当天的横截面统计）
        df = self.xs_normalize(df, is_fit=True)
        
        # 4. time‑series scaler per stock
        df = self.ts_scale(df, is_fit=True)
        print("完成预处理")
        return df.dropna().reset_index(drop=True)

    # ----- transform（预测时用） -----
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["code","date"]).reset_index(drop=True)
        
        # 1. winsor
        df = self.winsorize(df, is_fit=False)
        
        # 2. lag
        df = self.add_lags(df)
        
        # 3. cross‑section normalize 日度（使用当天的横截面统计）
        df = self.xs_normalize(df, is_fit=False)
        
        # 4. time‑series scaler per stock
        df = self.ts_scale(df, is_fit=False)
        
        return df


# ---------- 4. Pipeline ----------
class FactorPipeline:
    def __init__(
        self,
        factor_dir: str = "",
        price_loader = None,      # callable(codes, end, window)->pd.DataFrame
        factor_names: Optional[List[str]] = None,
        period:int=2, buy:str="open", sell:str="open",
        start:str="2018-01-01", end:str=datetime.today().strftime("%Y-%m-%d"),
        lag_days:List[int]=[0], winsor=(0.01,0.99), xs_norm=None,
        ts_scale_type=None,
        train_ratio=0.8,val_ratio=0.11,random_state=42
    ):
        self.factor_dir, self.factor_names = factor_dir, factor_names
        self.period, self.buy, self.sell = period, buy, sell
        self.start, self.end = start, end
        self.prep = Preprocessor(lag_days, winsor, xs_norm, ts_scale_type)
        self.train_ratio, self.val_ratio = train_ratio, val_ratio
        self.random_state = random_state
        self.price_loader = price_loader  # 注入型 (方便单测或替换)
        self.window = len(get_trading_dates(self.start, self.end)) + self.period + max(self.prep.lag_days)

        # 容器
        self.df: pd.DataFrame|None = None
        self.model = None  # 移除XGBRegressor类型限制

    # =========== 数据 ===========
    def build_dataset(self):
        # 1. 因子表
        fac_df = FactorLoader(self.factor_dir, self.factor_names,
                              self.start, self.end).load()
        print("完成因子加载")
        # 2. 预处理因子（不包含return）
        fac_df_prep = self.prep.fit_transform(fac_df)
        print("完成因子预处理")
        # 3. 收益
        codes = fac_df["code"].unique().tolist()
        daily = self.price_loader(codes, self.end, self.window)
        ret_df = make_future_return(daily, self.period, self.buy, self.sell)
        print("完成收益计算")
        # 4. merge
        self.df = fac_df_prep.merge(ret_df, on=["code","date"], how="inner")
        print(f"[Pipeline] 最终行数 {len(self.df)}, 特征 {self.df.shape[1]-3}")

    # =========== 划分 ===========
    def _split(self):
        if self.df is None: raise RuntimeError("先 build_dataset()")
        df = self.df.sort_values("date")
        
        # 按日期分组，确保同一天的数据不会被分割
        unique_dates = df['date'].unique()
        n_dates = len(unique_dates)
        
        # 按日期比例切分
        tr_dates = int(n_dates * self.train_ratio)
        va_dates = int(n_dates * (self.train_ratio + self.val_ratio))
        
        train_dates = unique_dates[:tr_dates]
        val_dates = unique_dates[tr_dates:va_dates]
        test_dates = unique_dates[va_dates:]
        
        # 根据日期切分数据
        train = df[df['date'].isin(train_dates)]
        val = df[df['date'].isin(val_dates)]
        test = df[df['date'].isin(test_dates)]
        
        return train, val, test

    # =========== 数据获取方法 ===========
    def get_train_data(self):
        """获取训练数据"""
        train, _, _ = self._split()
        return self._xy(train)
    
    def get_val_data(self):
        """获取验证数据"""
        _, val, _ = self._split()
        return self._xy(val)
    
    def get_test_data(self):
        """获取测试数据"""
        _, _, test = self._split()
        return self._xy(test)
    
    def get_all_splits(self):
        """获取所有数据分割"""
        train, val, test = self._split()
        return {
            'train': self._xy(train),
            'val': self._xy(val), 
            'test': self._xy(test),
            'train_df': train,
            'val_df': val,
            'test_df': test
        }
    
    def get_preprocessed_data(self):
        """获取预处理后的完整数据（未分割）"""
        if self.df is None:
            raise RuntimeError("先调用 build_dataset()")
        return self.df
    
    # =========== 模型训练（可选，保持向后兼容） ===========
    def fit_xgb(self, **xgb_params):
        """训练XGBoost模型（保持向后兼容）"""
        train,val,_ = self._split()
        X_tr,y_tr = self._xy(train); X_va,y_va = self._xy(val)

        default = dict(
            n_estimators=1200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=150,
            random_state=self.random_state,
            n_jobs=-1
        ); default.update(xgb_params)
        self.model = XGBRegressor(**default)
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_va,y_va)],
            verbose=False
        )
        # 安全地获取best_iteration和best_score
        best_iter = getattr(self.model, 'best_iteration', None)
        best_score = getattr(self.model, 'best_score', None)
        print(f"[XGB] best_iteration={best_iter}  best_score={best_score}")
        
        return self.model

    # =========== 评估 ===========
    def evaluate(self, model=None):
        """评估模型性能"""
        if model is None:
            model = self.model
        if model is None:
            raise RuntimeError("模型未训练，请先训练模型或传入model参数")
            
        _,_,test = self._split()
        X_t,y_t = self._xy(test)
        preds = model.predict(X_t)
        mse = mean_squared_error(y_t,preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_t,preds)
        print(f"Test RMSE {rmse:.4e}  R2 {r2:.3f}")
        
        # 返回评估结果DataFrame
        result_df = test[["code", "date"]].copy()
        result_df["y"] = y_t
        result_df["pred"] = preds
        
        return result_df, {"rmse": rmse, "r2": r2}
    
    def evaluate_model(self, model, data_type='test'):
        """评估指定模型在指定数据集上的性能"""
        if data_type == 'train':
            X, y = self.get_train_data()
            df = self._split()[0]
        elif data_type == 'val':
            X, y = self.get_val_data()
            df = self._split()[1]
        elif data_type == 'test':
            X, y = self.get_test_data()
            df = self._split()[2]
        else:
            raise ValueError("data_type 必须是 'train', 'val', 或 'test'")
            
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, preds)
        
        print(f"{data_type.capitalize()} RMSE {rmse:.4e}  R2 {r2:.3f}")
        
        result_df = df[["code", "date"]].copy()
        result_df["y"] = y
        result_df["pred"] = preds
        
        return result_df, {"rmse": rmse, "r2": r2}

    # =========== 预测 ===========
    def predict_range(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model=None
    ) -> pd.DataFrame:
        """
        使用训练时的因子数据源进行预测
        自动补齐因子数据起始日期以支持 lag 特征
        """
        if model is None:
            model = self.model
        if model is None:
            raise RuntimeError("模型未训练，请先训练模型或传入model参数")
        
        # 1. 处理日期范围，自动补齐 lag 需要的历史数据
        lag_max = max(self.prep.lag_days) if self.prep.lag_days else 0
        # 允许用户不传入日期，默认用训练时的
        _start = start_date or self.start
        _end = end_date or self.end

        # 获取所有可用交易日
        all_dates = get_trading_dates("2000-01-01", _end)
        if _end not in all_dates:
            raise ValueError(f"指定的 end_date {_end} 不在交易日历中")
        # 目标预测区间的索引
        end_idx = all_dates.index(_end)
        # 需要往前补 lag_max 天
        if lag_max > 0:
            # 如果只预测一天，自动往前补
            if _start == _end:
                if end_idx - lag_max < 0:
                    raise ValueError(f"历史数据不足，无法补齐 lag={lag_max} 天的因子数据")
                _start_adj = all_dates[end_idx - lag_max]
                print(f"[Pipeline] 预测 { _end }，自动补齐因子数据起始日期为 {_start_adj}（以支持 lag 特征）")
                _start = _start_adj
            else:
                # 如果用户指定了区间，也要判断是否足够
                start_idx = all_dates.index(_start)
                if start_idx - lag_max < 0:
                    raise ValueError(f"start_date={_start} 离 end_date={_end} 太近，无法补齐 lag={lag_max} 天的因子数据")
                # 自动往前补
                _start_adj = all_dates[start_idx - lag_max]
                if _start_adj != _start:
                    print(f"[Pipeline] 自动将因子数据起始日期从 {_start} 提前到 {_start_adj}（以支持 lag 特征）")
                    _start = _start_adj

        # 2. 加载因子数据
        fac_df = FactorLoader(self.factor_dir, self.factor_names, _start, _end).load()
        
        if fac_df.empty:
            print("[Pipeline] 预测期间无因子数据")
            return pd.DataFrame(columns=["code", "date", "pred"])

        # 3. 标准化预处理（transform，不要再 fit）
        df_prep = self.prep.transform(fac_df)

        # 4. 只保留用户实际想要预测的日期区间
        # 只保留 date 在 [start_date, end_date] 区间的数据
        mask = (df_prep["date"] >= pd.to_datetime(start_date or self.start)) & (df_prep["date"] <= pd.to_datetime(end_date or self.end))
        df_prep = df_prep[mask]

        # 5. 预测
        X = df_prep.drop(columns=["code", "date"]).values
        df_prep["pred"] = model.predict(X)

        return df_prep[["code", "date", "pred"]]

    # =========== 保存 / 载入 ===========
    def save(self, path: str, model=None):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        if model is None:
            model = self.model
        if model is not None:
            joblib.dump(model, path/"model.joblib")
        else:
            print("警告：没有模型可保存")
        
        # 保存预处理参数
        with open(path/"prep.json","w",encoding="utf8") as f:
            json.dump({
                "lag":self.prep.lag_days,
                "winsor":self.prep.winsor,
                "xs_norm":self.prep.xs_norm,
                "ts_scale_type":self.prep.ts_scale_type,
                "factor_dir": self.factor_dir,
                "factor_names": self.factor_names,
                "period": self.period,
                "buy": self.buy,
                "sell": self.sell,
                "start": self.start,
                "end": self.end,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "random_state": self.random_state
            }, f, ensure_ascii=False, indent=2)
        
        # 保存预处理器状态
        joblib.dump(self.prep.scalers, path/"scalers.joblib")
        joblib.dump(self.prep.winsor_limits, path/"winsor_limits.joblib")
        joblib.dump(self.prep.factor_columns, path/"factor_columns.joblib")
        
        print(f"[Pipeline] 已保存到 {path}")

    def load(self, path: str):
        """加载训练好的模型和预处理管道 - 不需要传任何参数"""
        path = Path(path)
        
        # 加载模型
        self.model = joblib.load(path/"model.joblib")
        
        # 加载预处理参数
        with open(path/"prep.json","r",encoding="utf8") as f:
            params = json.load(f)
        
        # 恢复pipeline参数
        self.factor_dir = params["factor_dir"]
        self.factor_names = params["factor_names"]
        self.period = params["period"]
        self.buy = params["buy"]
        self.sell = params["sell"]
        self.start = params["start"]
        self.end = params["end"]
        self.train_ratio = params["train_ratio"]
        self.val_ratio = params["val_ratio"]
        self.random_state = params["random_state"]
        
        # 重建预处理器
        self.prep = Preprocessor(
            params["lag"], params["winsor"], 
            params["xs_norm"], params["ts_scale_type"]
        )
        
        # 加载预处理器状态
        self.prep.scalers = joblib.load(path/"scalers.joblib")
        self.prep.winsor_limits = joblib.load(path/"winsor_limits.joblib")
        self.prep.factor_columns = joblib.load(path/"factor_columns.joblib")
        
        # 重新计算 window，保证和 init 行为一致
        self.window = len(get_trading_dates(self.start, self.end)) + self.period + max(self.prep.lag_days)
        
        print(f"[Pipeline] 已从 {path} 加载模型和预处理管道")

    def summary(self, model=None) -> str:
        """输出模型摘要信息"""
        if model is None:
            model = self.model
        if model is None:
            return "模型未训练或加载"
        
        summary_lines = [
            "=" * 50,
            "FactorPipeline 模型摘要",
            "=" * 50,
            f"因子目录: {self.factor_dir}",
            f"因子列表: {self.factor_names if self.factor_names else '全部'}",
            f"训练期间: {self.start} 到 {self.end}",
            f"预测周期: {self.period} 天",
            f"买入价格: {self.buy}",
            f"卖出价格: {self.sell}",
            "",
            "预处理参数:",
            f"  滞后天数: {self.prep.lag_days}",
            f"  去极值: {self.prep.winsor}",
            f"  横截面标准化: {self.prep.xs_norm}",
            f"  时序标准化: {self.prep.ts_scale_type}",
            "",
            "模型信息:",
            f"  模型类型: {type(model).__name__}",
            f"  最佳迭代: {getattr(model, 'best_iteration', 'N/A')}",
            f"  最佳分数: {getattr(model, 'best_score', 'N/A')}",
            f"  特征数量: {len(self.prep.factor_columns)}",
            f"  训练股票数: {len(self.prep.scalers)}",
            "",
            "=" * 50
        ]
        
        return "\n".join(summary_lines)

    # -------------- utils --------------
    def _xy(self, df: pd.DataFrame) -> Tuple[np.ndarray,np.ndarray]:
        X = df.drop(columns=["code","date","future_return"]).values
        y = df["future_return"].values
        return X,y

