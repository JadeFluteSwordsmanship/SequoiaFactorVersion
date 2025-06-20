import pandas as pd
from factors.factor_base import factor_registry

@factor_registry.register(window=20)
def vol_20d(df: pd.DataFrame) -> pd.DataFrame:
    """
    20日波动率因子
    """
    out = df.copy()
    out["vol_20d"] = out["close"].rolling(20).std()
    return out[["code", "trade_date", "vol_20d"]] 