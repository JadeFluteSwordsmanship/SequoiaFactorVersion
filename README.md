## SequoiaFactorVersion — 数据驱动的因子研究与选股框架

本项目围绕“数据更新 → 因子计算 → 评估与可视化 → 训练与预测”的完整闭环，提供：
- 数据层：分钟/日线/daily_basic/资金流/分红/沪深股通十大成交股等的增量与初始化更新
- 因子层：统一的`FactorBase`抽象与便捷的全量/增量写盘
- 评估层：标准化的因子评估指标与Plotly可视化
- 训练层：基于DuckDB的因子宽表加载与可选的XGBoost训练、预测管道

项目默认以本地`config.yaml`作为全局配置，执行入口为`main.py`（单次/定时）。

### 目录概览（节选）
- `main.py`：按配置执行单次或定时任务（数据更新 + 因子增量）
- `work_flow.py`：对外暴露`prepare(today, today_ymd)`，串联数据与因子更新
- `daily_data_fetcher.py`：数据更新任务注册与执行（分钟、日线、daily_basic、moneyflow、分红、沪深股通）
- `data_reader.py`：统一的数据读取接口（parquet）
- `factors/factor_base.py`：因子基类及读写工具，统一注册机制. factors中仅提供部分因子。
- `evaluation/factor_evaluator.py`：因子评估（IC/RankIC、分组、TopN、多空等）
- `evaluation/factor_plots_plotly.py`：评估结果的Plotly图表与落盘
- `trainer/factor_pipeline.py`：因子宽表加载、预处理、训练/预测流水线

### 快速开始
1) 准备环境
```
conda create -n sequoia39 python=3.9
conda activate sequoia39
```

2) 安装依赖（推荐）
```
pip install -r requirements.txt
```
如未提供requirements，可按需安装常用包：
```
pip install akshare tushare duckdb plotly kaleido tqdm xgboost scikit-learn joblib schedule pyarrow pandas numpy
```

3) 生成配置
```
copy config.yaml.example config.yaml   # Windows
# 或
cp config.yaml.example config.yaml     # macOS/Linux
```

`config.yaml`关键项示例：
```yaml
cron: false            # 是否以定时任务模式持续运行
data_dir: "E:/data"    # 数据根目录（自定义）
end_date:              # 可用于回测/评估的截止日期

push:                  # 可选：邮件推送（如不需要可禁用）
  enable: true
  smtp:
    server: "smtp.example.com"
    port: 465
    username: "your_email@example.com"
    password: "your_auth_code"
    from_email: "your_email@example.com"
    to_email: "recipient@example.com"

daily_snapshot_workers: 16
tushare_token: 'your_tushare_pro_token_here'
```

### 运行
- 本地单次执行（自动判断是否交易日）：
```
python main.py
```
- 定时任务：将`config.yaml`中的`cron`设为`true`后，`main.py`将常驻，并在每天`21:40`运行（见`main.py`）。

日志输出位于`logs/`目录（每日滚动）。

### 数据目录结构（默认位于`data_dir`下）
- `minute/*.parquet`：A股分钟线
- `daily/*.parquet`：不复权日线
- `daily_qfq/*.parquet`：前复权日线（快照）
- `daily_basic/*.parquet`：Tushare daily_basic
- `moneyflow/*.parquet`：Tushare 资金流
- `dividend/*.parquet`：Tushare 分红
- `other/hsgt_top10.parquet`：沪深股通十大成交股
- `basics/*.parquet`：基础信息（如`stock_basic.parquet`、行业成分等）
- `factors/*.parquet`：各因子结果，命名为`<因子名>.parquet`

数据由`daily_data_fetcher.py`中的任务集中更新，包括但不限于：
- `update_minute_data`
- `update_daily_qfq_data_snapshot`
- `update_daily_data`
- `update_daily_basic_data`
- `update_moneyflow_data`
- `update_dividend_data`
- `update_hsgt_top10_data`

### 因子开发与存储
- 因子需继承`factors/factor_base.py`中的`FactorBase`：
  - 必填类属性：`name`、`direction`（1 或 -1）、`description`、`data_requirements`
  - 实现方法：`_compute_impl(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame`
  - 推荐输出列：`['code','date','factor','value']`
- 存储路径：自动根据`settings.config['data_dir']/factors/<name>.parquet`生成
- 批量初始化：
  - 所有因子：`from factors.utils import initialize_all_factors`
  - 单个因子：`FactorBase.initialize_all(force=False)`（类方法）
- 增量更新：
  - 所有因子：`from factors.utils import update_all_factors_daily`
  - 单个因子：`FactorBase.update_daily(end_date=..., length=1)`（类方法）
- 方向统一：评估阶段会自动按因子`direction`将`value`乘以`±1`，正向越大越好。

### 因子评估（IC/分组/多空/TopN）示例
```python
from evaluation.factor_evaluator import FactorEvaluator
from factors.factor_base import FactorBase

# 列出并选择一个因子
print(FactorBase.list_all_factors().head())
fac = FactorBase.get_factor_by_name('<你的因子类名>')  # 例如 'FactorMoneyflow'，以实际名称为准

e = FactorEvaluator(
    factor=fac,           # 传入因子实例或类
    end_date='2024-12-31',
    window=256,           # 或配 start_date
    period=2,             # 未来收益周期
    buy_price='close',    # 收益口径
    sell_price='close'
)

# IC 序列与汇总
ic = e.ic_series(); print(ic.tail())
print(e.ic_stats())

# 分组日收益 → 净值
grp_daily = e.group_daily_returns(n_groups=5)

# 可视化（自动保存到 config.save_dir/<因子名>/）
from evaluation.factor_plots_plotly import fig_ir_distribution_panel, fig_group_navs
ir_df = e.daily_metrics()
fig_ir_distribution_panel(ir_df, factor_name=fac.name)
fig_group_navs(grp_daily, factor_name=fac.name)
```

### 训练/预测流水线（可选）
```python
import os
from trainer.factor_pipeline import FactorPipeline
from data_reader import get_daily_data
from settings import config

factor_dir = os.path.join(config['data_dir'], 'factors')
pipe = FactorPipeline(
    factor_dir=factor_dir,
    factor_names=None,     # None=使用全部因子，也可传列表
    period=2, buy='open', sell='open',
    start='2019-01-01', end='2024-12-31',
    lag_days=[0,1,5], winsor=(0.01,0.99), xs_norm='zscore', ts_scale_type='standard',
    train_ratio=0.8, val_ratio=0.1, random_state=42,
    add_daily_basic=True, daily_basic_lag_days=[0,1],
    fillna=False
)
pipe.build_dataset()
model = pipe.fit_xgb()
result_df, metrics = pipe.evaluate(model)
print(metrics)
```

### 常见问题
- 未配置`tushare_token`：会跳过依赖Tushare的数据更新任务
- 未安装`duckdb`/`kaleido`：DuckDB聚合或PNG导出会被跳过（HTML仍可用）
- `data_dir`为空：请先执行一次`python main.py`或在`work_flow.prepare()`中跑完更新
- 交易日日历：默认通过`akshare.tool_trade_date_hist_sina()`获取

### 备注
- Windows/macOS/Linux均可运行；Windows下复制配置文件可使用`copy`命令
- 若仅做回测/评估，可在`config.yaml`中设置`end_date`并直接使用`FactorEvaluator`

