"""
因子模型训练和回测模块

包含以下功能：
- FactorModelTrainer: 因子模型训练器
- FactorModelEvaluator: 因子模型评估器
- FactorModelPredictor: 因子模型预测器
"""

from .factor_model_trainer import FactorModelTrainer
from .factor_model_evaluator import FactorModelEvaluator
# from .factor_model_predictor import FactorModelPredictor

__all__ = [
    'FactorModelTrainer',
    'FactorModelEvaluator', 
    'FactorModelPredictor'
] 