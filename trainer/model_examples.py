"""
模型训练示例：展示如何使用重构后的FactorPipeline训练不同类型的模型
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path

from trainer.factor_pipeline import FactorPipeline
from data_reader import get_daily_data
from settings import config

def train_linear_models(pipeline):
    """训练线性模型示例"""
    print("=== 训练线性模型 ===")
    
    # 获取数据
    X_train, y_train = pipeline.get_train_data()
    X_val, y_val = pipeline.get_val_data()
    
    # 标准化特征（线性模型通常需要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n训练 {name}...")
        model.fit(X_train_scaled, y_train)
        
        # 评估
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"  Train RMSE: {train_rmse:.6f}, R2: {train_r2:.4f}")
        print(f"  Val RMSE: {val_rmse:.6f}, R2: {val_r2:.4f}")
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2
        }
    
    return results

def train_tree_models(pipeline):
    """训练树模型示例"""
    print("\n=== 训练树模型 ===")
    
    X_train, y_train = pipeline.get_train_data()
    X_val, y_val = pipeline.get_val_data()
    
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n训练 {name}...")
        model.fit(X_train, y_train)
        
        # 评估
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"  Train RMSE: {train_rmse:.6f}, R2: {train_r2:.4f}")
        print(f"  Val RMSE: {val_rmse:.6f}, R2: {val_r2:.4f}")
        
        results[name] = {
            'model': model,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2
        }
    
    return results

def train_neural_network(pipeline):
    """训练神经网络示例"""
    print("\n=== 训练神经网络 ===")
    
    X_train, y_train = pipeline.get_train_data()
    X_val, y_val = pipeline.get_val_data()
    
    # 标准化特征（神经网络需要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 创建神经网络
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )
    
    print("训练神经网络...")
    model.fit(X_train_scaled, y_train)
    
    # 评估
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"  Train RMSE: {train_rmse:.6f}, R2: {train_r2:.4f}")
    print(f"  Val RMSE: {val_rmse:.6f}, R2: {val_r2:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2
    }

def compare_models(pipeline, all_results):
    """比较所有模型性能"""
    print("\n=== 模型性能比较 ===")
    
    comparison_data = []
    for model_name, result in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'Train_RMSE': result['train_rmse'],
            'Val_RMSE': result['val_rmse'],
            'Train_R2': result['train_r2'],
            'Val_R2': result['val_r2']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Val_RMSE')
    
    print(comparison_df.to_string(index=False, float_format='%.6f'))
    
    # 找到最佳模型
    best_model_name = comparison_df.iloc[0]['Model']
    best_result = all_results[best_model_name]
    
    print(f"\n最佳模型: {best_model_name}")
    print(f"验证集 RMSE: {best_result['val_rmse']:.6f}")
    print(f"验证集 R2: {best_result['val_r2']:.4f}")
    
    return best_model_name, best_result

def save_best_model(pipeline, best_model_name, best_result, save_dir="models"):
    """保存最佳模型"""
    save_path = Path(save_dir) / best_model_name.lower().replace(' ', '_')
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model = best_result['model']
    joblib.dump(model, save_path / "model.joblib")
    
    # 如果有scaler，也保存
    if 'scaler' in best_result:
        joblib.dump(best_result['scaler'], save_path / "scaler.joblib")
    
    # 保存pipeline（包含预处理信息）
    pipeline.save(str(save_path), model=model)
    
    print(f"\n最佳模型已保存到: {save_path}")
    
    return save_path

def main():
    """主函数：完整的模型训练流程"""
    # 1. 构建数据pipeline
    factor_dir = f"{config.get('data_dir','E:/data')}/factors"
    
    pipeline = FactorPipeline(
        factor_dir=factor_dir,
        price_loader=get_daily_data,
        factor_names=['Alpha001', 'Alpha002', 'Custom100'],  # 示例因子
        start="2023-01-01",
        end="2024-12-31",
        period=2,
        buy="open",
        sell="close",
        lag_days=[0, 1, 2],
        winsor=(0.01, 0.99),
        xs_norm="zscore",
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # 构建数据集
    pipeline.build_dataset()
    
    # 2. 训练不同类型的模型
    all_results = {}
    
    # 线性模型
    linear_results = train_linear_models(pipeline)
    all_results.update(linear_results)
    
    # 树模型
    tree_results = train_tree_models(pipeline)
    all_results.update(tree_results)
    
    # 神经网络
    nn_result = train_neural_network(pipeline)
    all_results['NeuralNetwork'] = nn_result
    
    # 3. 比较模型性能
    best_model_name, best_result = compare_models(pipeline, all_results)
    
    # 4. 保存最佳模型
    save_path = save_best_model(pipeline, best_model_name, best_result)
    
    # 5. 使用最佳模型进行预测
    print(f"\n=== 使用最佳模型进行预测 ===")
    best_model = best_result['model']
    
    # 在测试集上评估
    test_result_df, test_metrics = pipeline.evaluate_model(best_model, 'test')
    print(f"测试集性能: RMSE={test_metrics['rmse']:.6f}, R2={test_metrics['r2']:.4f}")
    
    # 预测未来数据
    predictions = pipeline.predict_range(
        start_date="2024-12-01",
        end_date="2024-12-31",
        model=best_model
    )
    print(f"预测结果形状: {predictions.shape}")
    print(predictions.head())

if __name__ == "__main__":
    main() 