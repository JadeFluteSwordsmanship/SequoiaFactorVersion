from trainer.factor_pipeline import FactorPipeline
from data_reader import get_daily_data                     # 你已有实现
from settings import config

factor_dir = f"{config.get('data_dir','E:/data')}/factors"

# =========== 训练阶段 ===========
print("=== 开始训练模型 ===")
pipe = FactorPipeline(
    factor_dir      = factor_dir,
    price_loader    = get_daily_data,
    factor_names    = None,           # None=全部
    start           = "2021-01-01",
    end             = "2025-06-30",
    period          = 2,
    buy             = "open",         # t+1 开盘买
    sell            = "close",        # t+1 收盘卖
    lag_days        = [0,1,5],
    winsor          = (0.01,0.99)
)

pipe.build_dataset()
pipe.fit()
pipe.evaluate()
pipe.save("models/alpha_all_xgb")

# =========== 预测阶段（模拟重启后） ===========
print("\n=== 模拟重启后加载模型进行预测 ===")
# 创建空的pipeline实例（不需要传任何参数）
new_pipe = FactorPipeline()

# 加载训练好的模型和预处理管道（所有参数都会从文件读取）
new_pipe.load("models/alpha_all_xgb")

# 查看模型摘要
print(new_pipe.summary())

# 使用加载的模型进行预测（使用训练时的因子数据源）
pred_df = new_pipe.predict_range(
    start_date="2024-07-01",  # 预测未来数据
    end_date="2024-07-31"
)
print("预测结果:")
print(pred_df.head())
print(f"预测了 {len(pred_df)} 条记录")

# 也可以预测最新数据（不指定日期范围）
latest_pred = new_pipe.predict_range()
print(f"\n最新预测结果（共 {len(latest_pred)} 条记录）:")
print(latest_pred.head())
