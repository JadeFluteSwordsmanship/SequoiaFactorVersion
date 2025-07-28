#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
因子模型训练和评估使用示例 - 高效版本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.factor_model_trainer import FactorModelTrainer
from models.factor_model_evaluator import FactorModelEvaluator


def example_efficient_training():
    """高效训练示例"""
    print("=== 高效因子模型训练示例 ===\n")
    
    # 1. 初始化训练器
    factor_names = ['Alpha001', 'Alpha004', 'Alpha005', 'Custom003', 'Custom014']
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    
    trainer = FactorModelTrainer(
        factor_names=factor_names,
        start_date=start_date,
        end_date=end_date,
        period=1,  # 预测1日后收益
        buy_price='close',
        sell_price='close',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    # 2. 高效构建宽表（使用duckdb + SQL pivot）
    print("步骤1: 高效构建因子宽表")
    wide_df = trainer.build_wide_table()
    print(f"宽表形状: {wide_df.shape}")
    print(f"特征列: {[col for col in wide_df.columns if col not in ['code', 'date', 'future_return']]}")
    print()
    
    # 3. 划分数据
    print("步骤2: 划分训练/验证/测试集")
    train_data, val_data, test_data = trainer.split_data()
    print()
    
    # 4. 训练模型
    print("步骤3: 训练模型")
    models = trainer.train_models(['rf', 'gbm', 'linear'])
    print()
    
    # 5. 评估模型
    print("步骤4: 评估模型")
    evaluation_results = trainer.evaluate_models()
    print()
    
    # 6. 查看特征重要性
    print("步骤5: 查看特征重要性")
    importance_df = trainer.get_feature_importance('rf')
    print("随机森林特征重要性 (Top 5):")
    print(importance_df.head())
    print()
    
    return trainer


def example_comprehensive_evaluation(trainer):
    """综合评估示例"""
    print("=== 综合模型评估 ===\n")
    
    # 创建评估器
    evaluator = FactorModelEvaluator(trainer)
    
    # 执行综合评估
    results = evaluator.comprehensive_evaluation()
    
    # 生成可视化
    print("生成性能对比图...")
    evaluator.plot_performance_comparison()
    
    # 生成预测vs实际散点图
    print("生成预测vs实际散点图...")
    evaluator.plot_prediction_vs_actual('rf')
    
    # 生成报告
    print("生成评估报告...")
    report = evaluator.generate_report('model_evaluation_report.txt')
    print("报告已保存到 model_evaluation_report.txt")
    
    return evaluator


def example_factor_combination_comparison():
    """不同因子组合对比示例"""
    print("=== 因子组合对比分析 ===\n")
    
    # 使用不同的因子组合
    factor_combinations = [
        ['Alpha001', 'Alpha004', 'Alpha005'],  # Alpha因子组合
        ['Custom003', 'Custom004', 'Custom005'],  # 高频因子组合
        ['Custom014', 'Custom015', 'Custom016'],  # 成交量因子组合
    ]
    
    results_summary = []
    
    for i, factors in enumerate(factor_combinations):
        print(f"分析因子组合 {i+1}: {factors}")
        
        trainer = FactorModelTrainer(
            factor_names=factors,
            start_date='2023-01-01',
            end_date='2024-12-31',
            period=1
        )
        
        # 快速训练和评估
        try:
            trainer.build_wide_table()
            trainer.split_data()
            trainer.train_models(['rf'])  # 只训练随机森林
            eval_results = trainer.evaluate_models()
            
            # 记录结果
            rf_result = eval_results[eval_results['model'] == 'rf'].iloc[0]
            results_summary.append({
                'factor_combination': ', '.join(factors),
                'test_r2': rf_result['test_r2'],
                'test_ic': rf_result['test_ic'],
                'test_mse': rf_result['test_mse']
            })
            
            print(f"  测试集R²: {rf_result['test_r2']:.4f}")
            print(f"  测试集IC: {rf_result['test_ic']:.4f}")
            print()
            
        except Exception as e:
            print(f"  错误: {e}")
            print()
    
    # 总结结果
    if results_summary:
        import pandas as pd
        summary_df = pd.DataFrame(results_summary)
        print("因子组合性能对比:")
        print(summary_df.to_string(index=False))


def main():
    """主函数"""
    print("高效因子模型训练和评估系统示例")
    print("=" * 50)
    
    try:
        # 高效训练示例
        trainer = example_efficient_training()
        
        # 综合评估
        evaluator = example_comprehensive_evaluation(trainer)
        
        # 因子组合对比
        example_factor_combination_comparison()
        
        print("\n=== 示例完成 ===")
        print("所有功能演示完毕！")
        
    except Exception as e:
        print(f"示例执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 