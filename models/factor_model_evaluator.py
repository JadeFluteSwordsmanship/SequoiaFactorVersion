#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
因子模型评估器
提供详细的模型性能分析和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FactorModelEvaluator:
    """
    因子模型评估器
    
    功能：
    1. 详细的模型性能分析
    2. 时间序列性能分析
    3. 特征重要性分析
    4. 预测结果可视化
    5. 回测策略评估
    """
    
    def __init__(self, trainer):
        """
        初始化评估器
        
        Args:
            trainer: FactorModelTrainer实例
        """
        self.trainer = trainer
        self.results = {}
        
    def comprehensive_evaluation(self) -> Dict[str, pd.DataFrame]:
        """
        综合评估所有模型
        
        Returns:
            Dict: 包含各种评估结果的字典
        """
        print("=== 开始综合模型评估 ===")
        
        # 基础性能评估
        basic_results = self.trainer.evaluate_models()
        
        # 时间序列分析
        time_series_results = self.analyze_time_series_performance()
        
        # 特征重要性分析
        feature_importance_results = self.analyze_feature_importance()
        
        # 预测准确性分析
        prediction_accuracy_results = self.analyze_prediction_accuracy()
        
        # 策略回测结果
        strategy_results = self.backtest_strategy()
        
        self.results = {
            'basic': basic_results,
            'time_series': time_series_results,
            'feature_importance': feature_importance_results,
            'prediction_accuracy': prediction_accuracy_results,
            'strategy': strategy_results
        }
        
        return self.results
    
    def analyze_time_series_performance(self) -> pd.DataFrame:
        """
        分析模型在不同时间段的表现
        
        Returns:
            DataFrame: 时间序列性能分析结果
        """
        print("[时间序列分析] 分析模型在不同时间段的表现...")
        
        if self.trainer.test_data is None:
            raise ValueError("请先调用 split_data() 划分数据")
        
        # 按月份分组分析
        test_data = self.trainer.test_data.copy()
        test_data['year_month'] = pd.to_datetime(test_data['date']).dt.to_period('M')
        
        results = []
        
        for model_type, model_info in self.trainer.models.items():
            model = model_info['model']
            
            # 按月份计算性能
            for period, group in test_data.groupby('year_month'):
                if len(group) < 10:  # 样本太少跳过
                    continue
                    
                X, y = self.trainer.prepare_features(group)
                y_pred = model.predict(X)
                
                # 计算该月的性能指标
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                ic = np.corrcoef(y, y_pred)[0, 1] if len(y) > 1 else np.nan
                
                results.append({
                    'model': model_type,
                    'period': str(period),
                    'sample_count': len(group),
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'ic': ic
                })
        
        results_df = pd.DataFrame(results)
        print(f"[时间序列分析] 完成，分析了 {len(results_df)} 个时间段")
        
        return results_df
    
    def analyze_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        分析所有模型的特征重要性
        
        Returns:
            Dict: 每个模型的特征重要性DataFrame
        """
        print("[特征重要性分析] 分析所有模型的特征重要性...")
        
        importance_dict = {}
        
        for model_type in self.trainer.models.keys():
            importance_df = self.trainer.get_feature_importance(model_type)
            if len(importance_df) > 0:
                importance_dict[model_type] = importance_df
                print(f"[特征重要性分析] {model_type}: 最重要的特征是 {importance_df.iloc[0]['feature']}")
        
        return importance_dict
    
    def analyze_prediction_accuracy(self) -> pd.DataFrame:
        """
        分析预测准确性，包括不同收益区间的表现
        
        Returns:
            DataFrame: 预测准确性分析结果
        """
        print("[预测准确性分析] 分析不同收益区间的预测准确性...")
        
        if self.trainer.test_data is None:
            raise ValueError("请先调用 split_data() 划分数据")
        
        test_data = self.trainer.test_data.copy()
        
        # 将实际收益分为几个区间
        test_data['return_quantile'] = pd.qcut(test_data['future_return'], q=5, labels=['很低', '低', '中等', '高', '很高'])
        
        results = []
        
        for model_type, model_info in self.trainer.models.items():
            model = model_info['model']
            
            for quantile, group in test_data.groupby('return_quantile'):
                if len(group) < 5:  # 样本太少跳过
                    continue
                    
                X, y = self.trainer.prepare_features(group)
                y_pred = model.predict(X)
                
                # 计算该区间的性能指标
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                ic = np.corrcoef(y, y_pred)[0, 1] if len(y) > 1 else np.nan
                
                results.append({
                    'model': model_type,
                    'return_quantile': quantile,
                    'sample_count': len(group),
                    'avg_actual_return': y.mean(),
                    'avg_predicted_return': y_pred.mean(),
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'ic': ic
                })
        
        results_df = pd.DataFrame(results)
        print(f"[预测准确性分析] 完成，分析了 {len(results_df)} 个收益区间")
        
        return results_df
    
    def backtest_strategy(self, top_n: int = 100) -> pd.DataFrame:
        """
        回测基于模型预测的投资策略
        
        Args:
            top_n: 每期选择预测收益最高的前N只股票
            
        Returns:
            DataFrame: 策略回测结果
        """
        print(f"[策略回测] 回测基于模型预测的投资策略 (top_n={top_n})...")
        
        if self.trainer.test_data is None:
            raise ValueError("请先调用 split_data() 划分数据")
        
        test_data = self.trainer.test_data.copy()
        
        results = []
        
        for model_type, model_info in self.trainer.models.items():
            model = model_info['model']
            
            # 按日期分组回测
            strategy_returns = []
            
            for date, group in test_data.groupby('date'):
                if len(group) < top_n:
                    continue
                
                # 获取预测值
                X, _ = self.trainer.prepare_features(group)
                predictions = model.predict(X)
                
                # 将预测值添加到组数据中
                group_with_pred = group.copy()
                group_with_pred['predicted_return'] = predictions
                
                # 选择预测收益最高的top_n只股票
                top_stocks = group_with_pred.nlargest(top_n, 'predicted_return')
                
                # 计算策略收益（等权重）
                strategy_return = top_stocks['future_return'].mean()
                strategy_returns.append({
                    'date': date,
                    'strategy_return': strategy_return,
                    'market_return': group['future_return'].mean(),  # 市场平均收益
                    'stock_count': len(top_stocks)
                })
            
            if strategy_returns:
                strategy_df = pd.DataFrame(strategy_returns)
                
                # 计算策略统计指标
                total_return = (1 + strategy_df['strategy_return']).prod() - 1
                market_return = (1 + strategy_df['market_return']).prod() - 1
                excess_return = total_return - market_return
                
                # 计算夏普比率（假设无风险利率为0）
                strategy_sharpe = strategy_df['strategy_return'].mean() / strategy_df['strategy_return'].std() if strategy_df['strategy_return'].std() > 0 else 0
                market_sharpe = strategy_df['market_return'].mean() / strategy_df['market_return'].std() if strategy_df['market_return'].std() > 0 else 0
                
                # 计算胜率
                win_rate = (strategy_df['strategy_return'] > strategy_df['market_return']).mean()
                
                results.append({
                    'model': model_type,
                    'total_return': total_return,
                    'market_return': market_return,
                    'excess_return': excess_return,
                    'strategy_sharpe': strategy_sharpe,
                    'market_sharpe': market_sharpe,
                    'win_rate': win_rate,
                    'trading_days': len(strategy_df),
                    'avg_daily_return': strategy_df['strategy_return'].mean(),
                    'volatility': strategy_df['strategy_return'].std()
                })
        
        results_df = pd.DataFrame(results)
        print(f"[策略回测] 完成，分析了 {len(results_df)} 个模型")
        
        return results_df
    
    def plot_performance_comparison(self, save_path: Optional[str] = None):
        """
        绘制模型性能对比图
        
        Args:
            save_path: 图片保存路径
        """
        if not self.results:
            raise ValueError("请先调用 comprehensive_evaluation() 进行评估")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('因子模型性能对比分析', fontsize=16)
        
        # 1. 基础性能对比
        basic_results = self.results['basic']
        ax1 = axes[0, 0]
        x = np.arange(len(basic_results))
        width = 0.35
        
        ax1.bar(x - width/2, basic_results['test_r2'], width, label='测试集R²', alpha=0.8)
        ax1.bar(x + width/2, basic_results['test_ic'], width, label='测试集IC', alpha=0.8)
        ax1.set_xlabel('模型')
        ax1.set_ylabel('性能指标')
        ax1.set_title('基础性能对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(basic_results['model'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 策略回测结果
        strategy_results = self.results['strategy']
        ax2 = axes[0, 1]
        x = np.arange(len(strategy_results))
        
        ax2.bar(x - width/2, strategy_results['total_return'], width, label='策略总收益', alpha=0.8)
        ax2.bar(x + width/2, strategy_results['excess_return'], width, label='超额收益', alpha=0.8)
        ax2.set_xlabel('模型')
        ax2.set_ylabel('收益率')
        ax2.set_title('策略回测结果')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategy_results['model'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 特征重要性（选择最好的模型）
        best_model = basic_results.loc[basic_results['test_r2'].idxmax(), 'model']
        feature_importance = self.results['feature_importance'].get(best_model, pd.DataFrame())
        
        if len(feature_importance) > 0:
            ax3 = axes[1, 0]
            top_features = feature_importance.head(10)
            ax3.barh(range(len(top_features)), top_features['importance'])
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'])
            ax3.set_xlabel('重要性')
            ax3.set_title(f'{best_model} 特征重要性 (Top 10)')
            ax3.grid(True, alpha=0.3)
        
        # 4. 时间序列性能
        time_series_results = self.results['time_series']
        if len(time_series_results) > 0:
            ax4 = axes[1, 1]
            for model in time_series_results['model'].unique():
                model_data = time_series_results[time_series_results['model'] == model]
                ax4.plot(model_data['period'], model_data['r2'], marker='o', label=model, alpha=0.7)
            
            ax4.set_xlabel('时间段')
            ax4.set_ylabel('R²')
            ax4.set_title('时间序列性能')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[可视化] 图片已保存到: {save_path}")
        
        plt.show()
    
    def plot_prediction_vs_actual(self, model_type: str = 'rf', save_path: Optional[str] = None):
        """
        绘制预测值vs实际值的散点图
        
        Args:
            model_type: 模型类型
            save_path: 图片保存路径
        """
        if model_type not in self.trainer.models:
            raise ValueError(f"模型 {model_type} 不存在")
        
        if self.trainer.test_data is None:
            raise ValueError("请先调用 split_data() 划分数据")
        
        # 获取预测值
        X_test, y_test = self.trainer.prepare_features(self.trainer.test_data)
        model = self.trainer.models[model_type]['model']
        y_pred = model.predict(X_test)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_test, y_pred, alpha=0.6, s=20)
        
        # 添加对角线
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='理想预测线')
        
        plt.xlabel('实际收益率')
        plt.ylabel('预测收益率')
        plt.title(f'{model_type} 模型: 预测值 vs 实际值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加R²和IC信息
        r2 = r2_score(y_test, y_pred)
        ic = np.corrcoef(y_test, y_pred)[0, 1]
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nIC = {ic:.4f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[可视化] 图片已保存到: {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        生成详细的评估报告
        
        Args:
            save_path: 报告保存路径
            
        Returns:
            str: 报告内容
        """
        if not self.results:
            raise ValueError("请先调用 comprehensive_evaluation() 进行评估")
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("因子模型评估报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 基本信息
        report_lines.append("1. 基本信息")
        report_lines.append("-" * 20)
        report_lines.append(f"因子数量: {len(self.trainer.factor_names)}")
        report_lines.append(f"因子列表: {', '.join(self.trainer.factor_names)}")
        report_lines.append(f"时间范围: {self.trainer.start_date} 到 {self.trainer.end_date}")
        report_lines.append(f"训练集比例: {self.trainer.train_ratio:.1%}")
        report_lines.append(f"验证集比例: {self.trainer.val_ratio:.1%}")
        report_lines.append(f"测试集比例: {self.trainer.test_ratio:.1%}")
        report_lines.append("")
        
        # 基础性能
        report_lines.append("2. 基础性能评估")
        report_lines.append("-" * 20)
        basic_results = self.results['basic']
        for _, row in basic_results.iterrows():
            report_lines.append(f"{row['model']}:")
            report_lines.append(f"  测试集R²: {row['test_r2']:.4f}")
            report_lines.append(f"  测试集IC: {row['test_ic']:.4f}")
            report_lines.append(f"  测试集MSE: {row['test_mse']:.6f}")
            report_lines.append("")
        
        # 策略回测
        report_lines.append("3. 策略回测结果")
        report_lines.append("-" * 20)
        strategy_results = self.results['strategy']
        for _, row in strategy_results.iterrows():
            report_lines.append(f"{row['model']}:")
            report_lines.append(f"  策略总收益: {row['total_return']:.4f}")
            report_lines.append(f"  超额收益: {row['excess_return']:.4f}")
            report_lines.append(f"  策略夏普比率: {row['strategy_sharpe']:.4f}")
            report_lines.append(f"  胜率: {row['win_rate']:.4f}")
            report_lines.append("")
        
        # 特征重要性
        report_lines.append("4. 特征重要性 (Top 5)")
        report_lines.append("-" * 20)
        for model_type, importance_df in self.results['feature_importance'].items():
            if len(importance_df) > 0:
                report_lines.append(f"{model_type}:")
                for _, row in importance_df.head(5).iterrows():
                    report_lines.append(f"  {row['feature']}: {row['importance']:.4f}")
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"[报告] 报告已保存到: {save_path}")
        
        return report_content 