from trainer.factor_pipeline import FactorPipeline
from data_reader import get_daily_data                     # 你已有实现
from settings import config

factor_dir = f"{config.get('data_dir','E:/data')}/factors"

pipe = FactorPipeline(
    factor_dir      = factor_dir,
    price_loader    = get_daily_data,
    factor_names    = None,           # None=全部
    start           = "2021-01-01",
    end             = "2024-06-30",
    period          = 1,
    buy             = "open",         # t+1 开盘买
    sell            = "close",        # t+1 收盘卖
    lag_days        = [0,1,5],
    winsor          = (0.01,0.99)
)

pipe.build_dataset()
pipe.fit()
pipe.evaluate()
pipe.save("models/alpha_all_xgb")

# ---- 预测 ----
today_files = [
    f"{factor_dir}/Alpha001.parquet",   # 假设这些 parquet 里已包含最新 1 天
    f"{factor_dir}/Alpha005.parquet"
]
pred_df = pipe.predict_latest(today_files)
print(pred_df.head())
