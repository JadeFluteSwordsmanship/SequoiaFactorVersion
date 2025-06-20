import pandas as pd
from typing import List, Dict, Any, Type
from abc import ABC, abstractmethod

class FactorMeta(type):
    registry: Dict[str, Type['FactorBase']] = {}

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if name != "FactorBase":
            mcs.registry[cls.__name__] = cls
        return cls

    @classmethod
    def get_factors(mcs):
        return mcs.registry

class FactorBase(ABC, metaclass=FactorMeta):
    name: str = ""
    data_requirements: Dict[str, Any] = {}  # 例如 {'daily': {'window': 60}}

    def __init__(self):
        pass

    def fetch_data(self, codes: List[str], end_date: str) -> Dict[str, pd.DataFrame]:
        data = {}
        for dtype, req in self.data_requirements.items():
            window = req.get('window', 1)
            if dtype == 'daily':
                data[dtype] = self.read_daily_data(codes, end_date, window)
            elif dtype == 'minute':
                data[dtype] = self.read_minute_data(codes, end_date, window)
            elif dtype == 'lhb':
                data[dtype] = self.read_lhb_data(codes, end_date, window)
            # 你可以继续扩展更多数据类型
            else:
                raise ValueError(f"Unknown data type: {dtype}")
        return data

    def read_daily_data(self, codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        # TODO: 实现日线数据读取
        from data_reader import get_daily_data
        return get_daily_data(codes, end_date, window)

    def read_minute_data(self, codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        # TODO: 实现分钟线数据读取
        from data_reader import get_minute_data
        return get_minute_data(codes, end_date, window)

    def read_lhb_data(self, codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        # TODO: 实现龙虎榜数据读取
        from data_reader import get_lhb_data
        return get_lhb_data(codes, end_date, window)

    @abstractmethod
    def compute(self, codes: List[str], end_date: str) -> pd.DataFrame:
        pass

    @classmethod
    def get_all_factors(cls):
        return FactorMeta.get_factors() 