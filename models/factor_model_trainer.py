#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
因子模型训练器 - 高效版本
使用duckdb一次性读取所有因子数据，SQL pivot操作构建宽表
"""

import pandas as pd
import numpy as np
import duckdb
import os
import glob
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from data_reader import get_daily_data
from utils import get_trading_dates
from settings import config


class FactorModelTrainer:
    """
    因子模型训练器 - 高效版本
    
    功能：
    1. 使用duckdb一次性读取所有因子数据
    2. SQL pivot操作构建宽表
    3. 支持训练/验证/测试集划分
    4. 训练多种机器学习模型
    5. 模型评估和预测
    """
    
    def __init__(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
        codes: Optional[List[str]] = None,
        period: int = 1,
        buy_price: str = "close",
        sell_price: str = "close",
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        random_state: int = 42
    ):
        """
        初始化因子模型训练器
        
        Args:
            factor_names: 因子名称列表，如['Alpha001', 'Alpha004', 'Alpha005']
            start_date: 开始日期
            end_date: 结束日期
            codes: 股票代码列表，默认全市场
            period: 收益计算周期
            buy_price: 买入价字段
            sell_price: 卖出价字段
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子
        """
        self.factor_names = factor_names
        self.start_date = start_date
        self.end_date = end_date
        self.codes = codes
        self.period = period
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # 验证比例总和
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
        
        # 数据存储
        self.wide_df = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.models = {}
        
    def build_wide_table(self) -> pd.DataFrame:
        """
        使用duckdb高效构建因子宽表
        
        Returns:
            DataFrame: 宽表，包含code, date, factor1, factor2, ..., return
        """
        print(f"[构建宽表] 开始构建 {len(self.factor_names)} 个因子的宽表...")
        
        # 获取factors目录路径
        data_dir = config.get('data_dir', 'E:/data')
        factors_dir = os.path.join(data_dir, 'factors')
        
        # 构建所有因子文件的路径列表
        factor_files = []
        for factor_name in self.factor_names:
            factor_path = os.path.join(factors_dir, f"{factor_name}.parquet")
            if os.path.exists(factor_path):
                factor_files.append(factor_path)
            else:
                print(f"[构建宽表] 警告: 因子文件不存在 {factor_path}")
        
        if not factor_files:
            raise ValueError("没有找到任何因子文件")
        
        print(f"[构建宽表] 找到 {len(factor_files)} 个因子文件")
        
        # 使用duckdb一次性读取所有因子数据
        con = duckdb.connect(':memory:')
        
        # 构建SQL查询，一次性读取所有因子数据
        files_str = str(factor_files).replace("'", '"')
        
        # 使用UNION ALL合并所有因子数据
        union_queries = []
        for i, factor_path in enumerate(factor_files):
            factor_name = self.factor_names[i]
            union_queries.append(f"""
                SELECT 
                    code, 
                    date, 
                    '{factor_name}' as factor_name, 
                    value
                FROM read_parquet('{factor_path}')
                WHERE date >= '{self.start_date}' AND date <= '{self.end_date}'
            """)
        
        union_sql = " UNION ALL ".join(union_queries)
        
        # 执行UNION查询
        print(f"[构建宽表] 读取因子数据...")
        con.execute(f"CREATE TABLE all_factors AS {union_sql}")
        
        # 使用PIVOT操作构建宽表
        print(f"[构建宽表] 执行PIVOT操作...")
        pivot_columns = ", ".join([f"'{name}'" for name in self.factor_names])
        
        pivot_sql = f"""
        SELECT 
            code, 
            date,
            {', '.join([f"MAX(CASE WHEN factor_name = '{name}' THEN value END) as {name}" for name in self.factor_names])}
        FROM all_factors
        GROUP BY code, date
        """
        
        wide_df = con.execute(pivot_sql).df()
        
        print(f"[构建宽表] PIVOT完成，数据形状: {wide_df.shape}")
        
        # 计算未来收益率
        print(f"[构建宽表] 计算未来收益率 (period={self.period}, buy={self.buy_price}, sell={self.sell_price})...")
        
        # 获取价格数据
        if self.codes is None:
            from data_reader import list_available_stocks
            self.codes = list_available_stocks('daily')
        
        # 获取足够的历史数据来计算收益率
        window = 255 + self.period + 5
        price_df = get_daily_data(self.codes, self.end_date, window)
        price_df = price_df.sort_values(['stock_code', 'trade_date'])
        
        # 过滤时间范围
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'])
        mask = (price_df['trade_date'] >= self.start_date) & (price_df['trade_date'] <= self.end_date)
        price_df = price_df[mask].copy()
        
        # 计算未来收益率
        g = price_df.groupby('stock_code')
        
        # 买入价shift
        if self.buy_price == 'close':
            buy = g[self.buy_price].shift(0)
            buy_shift = 0
        elif self.buy_price in ['open', 'high', 'low']:
            buy = g[self.buy_price].shift(-1)
            buy_shift = -1
        else:
            raise ValueError("buy_price仅支持'close'或'open'/'high'/'low'")
        
        sell = g[self.sell_price].shift(-self.period + buy_shift)
        future_return = (sell - buy) / buy
        
        # 构建收益率DataFrame
        return_df = price_df[['stock_code', 'trade_date']].copy()
        return_df['future_return'] = future_return.values
        return_df = return_df.rename(columns={'stock_code': 'code', 'trade_date': 'date'})
        return_df = return_df.dropna(subset=['future_return'])
        
        # 合并因子数据和收益率数据
        wide_df = wide_df.merge(return_df, on=['code', 'date'], how='inner')
        
        # 删除包含NaN的行
        wide_df = wide_df.dropna()
        
        print(f"[构建宽表] 完成! 最终数据形状: {wide_df.shape}")
        print(f"[构建宽表] 时间范围: {wide_df['date'].min()} 到 {wide_df['date'].max()}")
        print(f"[构建宽表] 股票数量: {wide_df['code'].nunique()}")
        
        self.wide_df = wide_df
        return wide_df
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序划分训练/验证/测试集
        
        Returns:
            Tuple: (train_data, val_data, test_data)
        """
        if self.wide_df is None:
            raise ValueError("请先调用 build_wide_table() 构建数据")
        
        print(f"[数据划分] 开始划分数据...")
        
        # 按时间排序
        df_sorted = self.wide_df.sort_values('date').reset_index(drop=True)
        
        # 计算分割点
        total_rows = len(df_sorted)
        train_end = int(total_rows * self.train_ratio)
        val_end = int(total_rows * (self.train_ratio + self.val_ratio))
        
        # 划分数据
        self.train_data = df_sorted.iloc[:train_end].copy()
        self.val_data = df_sorted.iloc[train_end:val_end].copy()
        self.test_data = df_sorted.iloc[val_end:].copy()
        
        print(f"[数据划分] 完成!")
        print(f"[数据划分] 训练集: {len(self.train_data)} 行 ({self.train_data['date'].min()} 到 {self.train_data['date'].max()})")
        print(f"[数据划分] 验证集: {len(self.val_data)} 行 ({self.val_data['date'].min()} 到 {self.val_data['date'].max()})")
        print(f"[数据划分] 测试集: {len(self.test_data)} 行 ({self.test_data['date'].min()} 到 {self.test_data['date'].max()})")
        
        return self.train_data, self.val_data, self.test_data
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和标签
        
        Args:
            df: 包含因子和收益率的DataFrame
            
        Returns:
            Tuple: (X, y) 特征矩阵和标签
        """
        # 特征列（排除code, date, future_return）
        feature_cols = [col for col in df.columns if col not in ['code', 'date', 'future_return']]
        X = df[feature_cols].values
        y = df['future_return'].values
        
        return X, y
    
    def train_models(self, model_types: List[str] = None) -> Dict[str, Any]:
        """
        训练多种模型
        
        Args:
            model_types: 模型类型列表，支持 ['rf', 'gbm', 'linear', 'ridge', 'lasso']
            
        Returns:
            Dict: 训练好的模型字典
        """
        if self.train_data is None:
            raise ValueError("请先调用 split_data() 划分数据")
        
        if model_types is None:
            model_types = ['rf', 'gbm', 'linear', 'ridge', 'lasso']
        
        print(f"[模型训练] 开始训练 {len(model_types)} 个模型...")
        
        # 准备训练数据
        X_train, y_train = self.prepare_features(self.train_data)
        X_val, y_val = self.prepare_features(self.val_data)
        
        # 特征列名
        feature_cols = [col for col in self.train_data.columns if col not in ['code', 'date', 'future_return']]
        
        # 定义模型
        model_configs = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        
        # 训练模型
        for model_type in model_types:
            if model_type not in model_configs:
                print(f"[模型训练] 警告: 不支持的模型类型 {model_type}")
                continue
                
            print(f"[模型训练] 训练 {model_type}...")
            
            model = model_configs[model_type]
            model.fit(X_train, y_train)
            
            # 预测和评估
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # 计算指标
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            
            # 特征重要性
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                importance = None
            
            # 存储结果
            self.models[model_type] = {
                'model': model,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'feature_importance': importance,
                'feature_names': feature_cols
            }
            
            print(f"[模型训练] {model_type} 完成 - 训练R²: {train_r2:.4f}, 验证R²: {val_r2:.4f}")
        
        print(f"[模型训练] 所有模型训练完成!")
        return self.models
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        评估所有模型在测试集上的表现
        
        Returns:
            DataFrame: 评估结果
        """
        if not self.models:
            raise ValueError("请先调用 train_models() 训练模型")
        
        if self.test_data is None:
            raise ValueError("请先调用 split_data() 划分数据")
        
        print(f"[模型评估] 在测试集上评估模型...")
        
        X_test, y_test = self.prepare_features(self.test_data)
        
        results = []
        
        for model_type, model_info in self.models.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            
            # 计算指标
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 计算IC（信息系数）
            ic = np.corrcoef(y_test, y_pred)[0, 1]
            
            results.append({
                'model': model_type,
                'test_mse': mse,
                'test_mae': mae,
                'test_r2': r2,
                'test_ic': ic,
                'train_r2': model_info['train_r2'],
                'val_r2': model_info['val_r2']
            })
        
        results_df = pd.DataFrame(results)
        print(f"[模型评估] 评估完成!")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def get_feature_importance(self, model_type: str = 'rf') -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            model_type: 模型类型
            
        Returns:
            DataFrame: 特征重要性排序
        """
        if model_type not in self.models:
            raise ValueError(f"模型 {model_type} 不存在")
        
        model_info = self.models[model_type]
        importance = model_info['feature_importance']
        feature_names = model_info['feature_names']
        
        if importance is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, model_type: str, df: pd.DataFrame) -> np.ndarray:
        """
        使用指定模型进行预测
        
        Args:
            model_type: 模型类型
            df: 包含因子的DataFrame
            
        Returns:
            np.ndarray: 预测结果
        """
        if model_type not in self.models:
            raise ValueError(f"模型 {model_type} 不存在")
        
        X, _ = self.prepare_features(df)
        model = self.models[model_type]['model']
        return model.predict(X) 