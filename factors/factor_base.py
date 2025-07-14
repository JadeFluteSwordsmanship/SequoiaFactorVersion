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
    description: str = ""  # 因子描述，用于查找和说明
    data_requirements: Dict[str, Any] = {}  # 例如 {'daily_qfq': {'window': 60}}
    # direction: 1=正向，-1=反向，子类必须显式声明

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'direction'):
            raise NotImplementedError(f"{cls.__name__} 必须声明 direction = 1 或 -1，表示因子方向性")

    def __init__(self):
        pass

    def read_daily_qfq_data(self, codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        # TODO: 实现日线数据读取
        from data_reader import get_daily_qfq_data
        return get_daily_qfq_data(codes, end_date, window)

    def read_daily_data(self, codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        # TODO: 实现不复权日线数据读取
        from data_reader import get_daily_data
        return get_daily_data(codes, end_date, window)

    def read_daily_basic_data(self, codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        # TODO: 实现daily_basic数据读取
        from data_reader import get_daily_basic_data
        return get_daily_basic_data(codes, end_date, window)

    def read_minute_data(self, codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        # TODO: 实现分钟线数据读取
        from data_reader import get_minute_data
        return get_minute_data(codes, end_date, window)

    def read_hsgt_top10_data(self, codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        # TODO: 实现沪深股通前十成交量数据读取
        from data_reader import get_hsgttop10_data
        return get_hsgttop10_data(codes, end_date, window)

    def read_moneyflow_data(self, codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        # TODO: 实现资金流向数据读取
        from data_reader import get_moneyflow_data
        return get_moneyflow_data(codes, end_date, window)

    def read_dividend_data(self, codes: List[str], end_date: str, window: int) -> pd.DataFrame:
        # TODO: 实现分红数据读取
        from data_reader import get_dividend_data
        return get_dividend_data(codes, end_date, window)

    def fetch_data(self, codes: List[str], end_date: str) -> Dict[str, pd.DataFrame]:
        """
        获取增量更新所需的数据（使用window参数）
        Args:
            codes: 股票代码列表
            end_date: 计算日期
        Returns:
            包含所有数据类型的字典
        """
        data = {}
        for dtype, req in self.data_requirements.items():
            window = req.get('window', 1)
            if dtype == 'daily_qfq':
                data[dtype] = self.read_daily_qfq_data(codes, end_date, window)
            elif dtype == 'daily':
                data[dtype] = self.read_daily_data(codes, end_date, window)
            elif dtype == 'daily_basic':
                data[dtype] = self.read_daily_basic_data(codes, end_date, window)
            elif dtype == 'minute':
                data[dtype] = self.read_minute_data(codes, end_date, window)
            elif dtype == 'hsgt_top10':
                data[dtype] = self.read_hsgt_top10_data(codes, end_date, window)
            elif dtype == 'moneyflow':
                data[dtype] = self.read_moneyflow_data(codes, end_date, window)
            elif dtype == 'dividend':
                data[dtype] = self.read_dividend_data(codes, end_date, window)
            else:
                raise ValueError(f"Unknown data type: {dtype}")
        return data

    def fetch_data_batch(self, codes: List[str], end_date: str) -> Dict[str, pd.DataFrame]:
        """
        批量获取所有历史数据，用于回填历史因子值
        Args:
            codes: 股票代码列表
            end_date: 计算日期
        Returns:
            包含所有数据类型的字典
        """
        data = {}
        for dtype, req in self.data_requirements.items():
            # 回填时获取所有历史数据，使用很大的window值
            window = 241 * 255  # 足够大的值来获取所有历史数据
            if dtype == 'daily_qfq':
                data[dtype] = self.read_daily_qfq_data(codes, end_date, window)
            elif dtype == 'daily':
                data[dtype] = self.read_daily_data(codes, end_date, window)
            elif dtype == 'daily_basic':
                data[dtype] = self.read_daily_basic_data(codes, end_date, window)
            elif dtype == 'minute':
                data[dtype] = self.read_minute_data(codes, end_date, window)
            elif dtype == 'hsgt_top10':
                data[dtype] = self.read_hsgt_top10_data(codes, end_date, window)
            elif dtype == 'moneyflow':
                data[dtype] = self.read_moneyflow_data(codes, end_date, window)
            elif dtype == 'dividend':
                data[dtype] = self.read_dividend_data(codes, end_date, window)
            else:
                raise ValueError(f"Unknown data type: {dtype}")
        return data


    @abstractmethod
    def compute(self, codes: List[str], end_date: str) -> pd.DataFrame:
        """
        计算单日因子值（增量计算）
        Args:
            codes: 股票代码列表
            end_date: 计算日期
        Returns:
            DataFrame，包含股票代码、日期和因子值
        """
        data = self.fetch_data(codes, end_date)
        return self._compute_impl(data, end_date, batch=False)
    

    def compute_batch(self, codes: List[str], end_date: str) -> pd.DataFrame:
        """
        批量计算因子值（历史回填）
        Args:
            codes: 股票代码列表
            end_date: 计算日期
        Returns:
            DataFrame，包含股票代码、日期和因子值
        """
        # 获取所有历史数据
        data = self.fetch_data_batch(codes, end_date)
        
        # 调用具体的批量计算实现
        return self._compute_impl(data, end_date, batch=True)

    def _compute_impl(self, data: Dict[str, pd.DataFrame], end_date: str, batch: bool = False) -> pd.DataFrame:
        """
        批量计算的具体实现，子类必须重写此方法
        使用所有历史数据一次性计算所有历史因子值
        """
        raise NotImplementedError("子类必须实现_compute_impl此方法")

    @classmethod
    def get_all_factors(cls):
        return FactorMeta.get_factors() 

    @classmethod
    def get_factor_by_name(cls, name: str) -> 'FactorBase':
        """根据因子名称获取因子实例"""
        factors = cls.get_all_factors()
        if name in factors:
            return factors[name]()
        else:
            raise ValueError(f"因子 {name} 不存在")

    @classmethod
    def search_factors_by_description(cls, keyword: str) -> List['FactorBase']:
        """
        根据描述关键词搜索因子，支持多关键词搜索
        例如：搜索"涨停"会找到包含涨停的因子
        搜索"日内 波动"会找到同时包含"日内"和"波动"的因子
        """
        factors = cls.get_all_factors()
        matching_factors = []
        
        # 将搜索关键词按空格分割
        keywords = keyword.lower().split()
        
        for name, factor_class in factors.items():
            factor_instance = factor_class()
            description = factor_instance.description.lower()
            
            # 检查所有关键词是否都在描述中
            if all(kw in description for kw in keywords):
                matching_factors.append(factor_instance)
        
        return matching_factors

    @classmethod
    def list_all_factors(cls) -> pd.DataFrame:
        """列出所有因子的信息，返回DataFrame"""
        factors = cls.get_all_factors()
        factor_info = []
        
        for name, factor_class in factors.items():
            factor_instance = factor_class()
            factor_info.append({
                'name': name,
                'description': factor_instance.description,
                'data_requirements': str(factor_instance.data_requirements)
            })
        
        return pd.DataFrame(factor_info) 