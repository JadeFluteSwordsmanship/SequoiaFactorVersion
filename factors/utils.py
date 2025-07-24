from factors.factor_base import FactorBase

def initialize_all_factors(codes=None, end_date=None, force=False):
    """
    初始化并批量计算所有注册因子，写入parquet。
    Args:
        codes: 股票代码列表，默认全市场
        end_date: 截止日期，默认今天
        force: 是否强制重算覆盖
    """
    from tqdm import tqdm
    factors = FactorBase.get_all_factors()
    for name, factor_class in tqdm(list(factors.items()), desc="初始化因子", ncols=80):
        print(f"\n[initialize_all_factors] 初始化因子: {name}")
        factor_class.initialize_all(codes=codes, end_date=end_date, force=force) 