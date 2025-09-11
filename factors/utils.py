from factors.factor_base import FactorBase
import time
import logging
import sys
from utils import setup_logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from collections import defaultdict
import os


def _load_data_by_type(dtype, codes, end_date, window):
    """
    统一的数据加载函数，根据数据类型调用对应的读取函数
    
    Args:
        dtype: 数据类型
        codes: 股票代码列表
        end_date: 结束日期
        window: 数据窗口大小
    
    Returns:
        DataFrame: 加载的数据
    """
    from data_reader import (get_daily_qfq_data, get_daily_data, get_minute_data, 
                           get_daily_basic_data, get_moneyflow_data, get_dividend_data,
                           get_stock_basic_data, get_industry_member_data, get_company_info_data,
                           get_index_daily_data, get_index_basic_data)
    
    # 根据数据类型调用对应的读取函数
    if dtype == 'daily_qfq':
        return get_daily_qfq_data(codes, end_date, window)
    elif dtype == 'daily':
        return get_daily_data(codes, end_date, window)
    elif dtype == 'minute':
        return get_minute_data(codes, end_date, window)
    elif dtype == 'daily_basic':
        return get_daily_basic_data(codes, end_date, window)
    elif dtype == 'moneyflow':
        return get_moneyflow_data(codes, end_date, window)
    elif dtype == 'dividend':
        return get_dividend_data(codes, end_date, window)
    elif dtype == 'stock_basic':
        return get_stock_basic_data(codes, end_date, window)
    elif dtype == 'industry_member':
        return get_industry_member_data(codes, end_date, window)
    elif dtype == 'company_info':
        return get_company_info_data(codes, end_date, window)
    elif dtype == 'index_daily':
        return get_index_daily_data(codes, end_date, window)
    elif dtype == 'index_basic':
        return get_index_basic_data(codes, end_date, window)
    else:
        logging.warning(f"未知数据类型: {dtype}")
        return pd.DataFrame()

def _load_data_for_requirements(data_requirements):
    """
    根据data_requirements加载数据，避免重复IO
    注意：批量回填时使用超大window获取所有历史数据
    """
    from data_reader import list_available_stocks
    from datetime import datetime
    
    data = {}
    
    for dtype in data_requirements.keys():  # 只关注数据类型，不关注window
        # 获取该数据类型的所有可用股票
        codes = list_available_stocks(dtype)
        
        # 设置结束日期为今天
        end_date = datetime.now().strftime('%Y-%m-%d 23:59:59')
        
        # 批量回填使用超大window获取所有历史数据
        if dtype == 'minute':
            # 分钟数据：一天241行，需要特别大的window
            window = 241 * 255 * 10  # 约10年的分钟数据
        elif dtype == 'dividend':
            # 分红数据：使用年数，获取20年的分红记录
            window = 50  # 20年
        else:
            # 其他数据：使用超大window
            window = 241 * 255  # 约241年的数据
        
        # 使用统一的数据加载函数
        df = _load_data_by_type(dtype, codes, end_date, window)
        if not df.empty:
            data[dtype] = df
            logging.info(f"已加载 {dtype} 数据，形状: {df.shape}")
    
    return data

def _estimate_memory_usage(data):
    """
    估算数据占用的内存大小（MB）
    """
    total_memory = 0
    for dtype, df in data.items():
        if df is not None and not df.empty:
            # 估算DataFrame内存使用
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            total_memory += memory_mb
            logging.info(f"{dtype} 数据内存使用: {memory_mb:.1f} MB")
    
    logging.info(f"总内存使用: {total_memory:.1f} MB")
    return total_memory

def _calculate_optimal_workers(data, max_workers, target_memory_per_worker=None):
    """
    根据数据大小和可用内存计算最优进程数
    
    Args:
        data: 加载的数据
        max_workers: 最大进程数
        target_memory_per_worker: 每个进程的计算开销内存（MB），包括：
            - pandas/numpy 计算过程中的临时变量
            - talib 技术指标计算的中间结果
            - numba 编译和运行时的内存开销
            - 因子计算过程中的临时DataFrame和Series
            - Python解释器本身的内存开销
    
    Returns:
        最优进程数
    """
    import psutil
    
    # 估算数据内存使用
    data_memory = _estimate_memory_usage(data)
    
    # 获取系统可用内存
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    logging.info(f"系统可用内存: {available_memory:.1f} MB")
    
    # 动态计算计算开销
    if target_memory_per_worker is None:
        # 根据数据大小动态调整计算开销
        if data_memory < 1000:  # 小数据
            target_memory_per_worker = 1000
        elif data_memory < 5000:  # 中等数据
            target_memory_per_worker = 2500
        else:  # 大数据
            target_memory_per_worker = 3500
    
    # 计算每个进程的内存需求
    # 数据内存：每个进程都会复制一份数据（multiprocessing的特性）
    # 计算开销：因子计算过程中的临时变量、中间结果等
    memory_per_worker = data_memory + target_memory_per_worker
    
    # 计算理论最大进程数
    theoretical_max = int(available_memory / memory_per_worker)
    
    # 取最小值，确保不超过系统限制
    optimal_workers = min(max_workers, theoretical_max, psutil.cpu_count())
    
    # 确保至少有1个进程
    optimal_workers = max(1, optimal_workers)
    
    logging.info(f"数据内存: {data_memory:.1f} MB, 每进程需求: {memory_per_worker:.1f} MB")
    logging.info(f"理论最大进程数: {theoretical_max}, 实际使用: {optimal_workers}")
    
    return optimal_workers

def _load_data_for_update(data_requirements, date, length=150):
    """
    根据data_requirements加载增量更新所需的数据
    注意：增量更新时使用data_requirements中的实际window值
    
    Args:
        data_requirements: 数据需求字典，如 {'daily': {'window': 20}}
        date: 更新日期
        length: 需要计算的天数（含date）
    """
    from data_reader import list_available_stocks
    
    data = {}
    
    for dtype, config in data_requirements.items():
        # 获取该数据类型的所有可用股票
        codes = list_available_stocks(dtype)
        
        # 使用data_requirements中的实际window值
        window = config.get('window', length)
        
        # 使用统一的数据加载函数
        df = _load_data_by_type(dtype, codes, date, window)
        if not df.empty:
            data[dtype] = df
            logging.info(f"已加载 {dtype} 增量数据，window: {window}，形状: {df.shape}")
    
    return data

def _compute_factor_batch(factor_class, data, force=False):
    """
    计算单个因子的批量数据
    """
    try:
        start_time = time.time()
        log_msg = f"[并行计算] 开始计算因子: {factor_class.name}"
        
        # 检查文件是否已存在
        factor_path = factor_class.get_factor_path()
        if os.path.exists(factor_path) and not force:
            log_msg = f"[并行计算] {factor_class.name} 已存在，跳过"
            return factor_class.name, 0, log_msg
        
        # 创建因子实例并计算
        factor_instance = factor_class()
        result_df = factor_instance._compute_impl(data)
        
        # 写入文件
        result_df.to_parquet(factor_path, index=False)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        log_msg = f"[并行计算] {factor_class.name} 计算完成，耗时 {elapsed:.2f} 秒，结果形状: {result_df.shape}"
        return factor_class.name, elapsed, log_msg
        
    except Exception as e:
        log_msg = f"[并行计算] {factor_class.name} 计算失败: {e}"
        return factor_class.name, -1, log_msg

def _update_factor_daily(factor_class, data, date):
    """
    计算单个因子的增量更新数据
    """
    try:
        start_time = time.time()
        log_msg = f"[并行更新] 开始更新因子: {factor_class.name} {date}"
        
        # 创建因子实例并计算
        factor_instance = factor_class()
        df_new = factor_instance._compute_impl(data)
        
        # 读取旧数据并合并
        factor_path = factor_class.get_factor_path()
        if os.path.exists(factor_path):
            try:
                df_old = pd.read_parquet(factor_path)
                df = pd.concat([df_old, df_new], ignore_index=True)
                # 保留旧数据，只更新新部分
                df = df.drop_duplicates(subset=['code', 'date', 'factor'], keep='last')
            except Exception as e:
                log_msg = f"[并行更新] 读取旧数据失败，将仅写入新数据: {e}"
                df = df_new
        else:
            df = df_new
        
        # 写入文件
        df.to_parquet(factor_path, index=False)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        log_msg = f"[并行更新] {factor_class.name} {date} 更新完成，耗时 {elapsed:.2f} 秒，新增数据形状: {df_new.shape}"
        return factor_class.name, elapsed, log_msg
        
    except Exception as e:
        log_msg = f"[并行更新] {factor_class.name} {date} 更新失败: {e}"
        return factor_class.name, -1, log_msg

def initialize_all_factors(codes=None, end_date=None, force=False, max_workers=4):
    """
    优化后的因子初始化函数：数据聚类 + 并行处理
    
    Args:
        codes: 股票代码列表，默认全市场
        end_date: 截止日期，默认今天
        force: 是否强制重算覆盖
        max_workers: 并行进程数
    """
    
    # 设置日志
    setup_logging('factors_init')
    
    # 获取所有因子
    factors = FactorBase.get_all_factors()
    logging.info(f"发现 {len(factors)} 个因子")
    
    # 按数据需求聚类因子（只关注数据类型，不关注window）
    factor_groups = defaultdict(list)
    for name, factor_class in factors.items():
        if hasattr(factor_class, 'data_requirements') and factor_class.data_requirements:
            # 只按数据类型分组，不关注window配置
            req_key = tuple(sorted(factor_class.data_requirements.keys()))
            factor_groups[req_key].append(factor_class)
        else:
            # 默认使用daily数据
            default_req = ('daily',)
            factor_groups[default_req].append(factor_class)
    
    logging.info(f"因子分组完成，共 {len(factor_groups)} 个数据需求组")
    
    total_start_time = time.time()
    
    # 按组处理因子
    for req_key, factor_classes in factor_groups.items():
        data_requirements = {dtype: {} for dtype in req_key}  # 创建空的配置字典
        logging.info(f"处理数据需求组: {list(data_requirements.keys())}，包含 {len(factor_classes)} 个因子")
        # 加载该组所需的数据（只加载一次）
        logging.info(f"开始加载数据: {list(data_requirements.keys())}")
        data = _load_data_for_requirements(data_requirements)
        logging.info(f"数据加载完成，共 {len(data)} 个数据类型")
        
        # 动态计算最优进程数
        optimal_workers = _calculate_optimal_workers(data, max_workers)
        
        # 并行计算该组的所有因子
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # 提交所有任务
            future_to_factor = {
                executor.submit(_compute_factor_batch, factor_class, data, force): factor_class
                for factor_class in factor_classes
            }
            
            # 收集结果
            completed_factors = []
            for future in as_completed(future_to_factor):
                factor_name, elapsed, log_msg = future.result()
                # 在主进程中输出日志
                logging.info(log_msg)
                if elapsed >= 0:
                    completed_factors.append((factor_name, elapsed))
                else:
                    logging.error(f"因子 {factor_name} 计算失败")
        
        # 输出该组的统计信息
        if completed_factors:
            total_time = sum(elapsed for _, elapsed in completed_factors)
            avg_time = total_time / len(completed_factors)
            successful_count = len(completed_factors)
            failed_count = len(factor_classes) - successful_count
            logging.info(f"数据需求组 {list(data_requirements.keys())} 完成，"
                        f"成功: {successful_count} 个因子，失败: {failed_count} 个因子，"
                        f"平均耗时 {avg_time:.2f} 秒，总耗时 {total_time:.2f} 秒")
            
            # 输出每个因子的详细耗时
            for factor_name, elapsed in sorted(completed_factors, key=lambda x: x[1], reverse=True):
                logging.info(f"  - {factor_name}: {elapsed:.2f} 秒")
    
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    
    # 统计总体结果
    total_factors = sum(len(factor_classes) for factor_classes in factor_groups.values())
    logging.info(f"所有因子初始化完成，总耗时 {total_elapsed:.2f} 秒")
    logging.info(f"总计处理 {total_factors} 个因子，平均每个因子耗时 {total_elapsed/total_factors:.2f} 秒")

def update_all_factors_daily(date, codes=None, length=1, max_workers=4):
    """
    优化后的因子增量更新函数：数据聚类 + 并行处理
    
    Args:
        date: 需要更新的日期（如'2024-06-01'）
        codes: 股票代码列表，默认当前可交易股票
        length: 需要计算的天数（含date）
        max_workers: 并行进程数
        log_level: 日志级别，支持'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    # 智能判断是否需要设置日志
    # 如果已经有日志处理器，说明是从main等地方调用的，使用现有日志
    # 如果没有日志处理器，说明是单独调用，需要设置新的日志
    
    # 获取所有因子
    factors = FactorBase.get_all_factors()
    logging.info(f"发现 {len(factors)} 个因子，开始增量更新 {date}")
    
    # 按数据需求组合分组，并计算每个组的最大window值
    factor_groups = defaultdict(list)
    group_requirements = {}  # 记录每个组的数据需求
    
    for name, factor_class in factors.items():
        if hasattr(factor_class, 'data_requirements') and factor_class.data_requirements:
            # 将数据需求转换为可哈希的元组，用于分组（只使用数据类型，不包含window值）
            req_key = tuple(sorted(factor_class.data_requirements.keys()))
            factor_groups[req_key].append(factor_class)
            
            # 记录该组的数据需求，并计算最大window值
            if req_key not in group_requirements:
                group_requirements[req_key] = {}
                for dtype, config in factor_class.data_requirements.items():
                    group_requirements[req_key][dtype] = {'window': config.get('window', 1)}
            else:
                # 更新最大window值
                for dtype, config in factor_class.data_requirements.items():
                    current_window = group_requirements[req_key][dtype]['window']
                    new_window = config.get('window', 1)
                    group_requirements[req_key][dtype]['window'] = max(current_window, new_window)
        else:
            # 默认使用daily数据
            default_req = ('daily',)
            factor_groups[default_req].append(factor_class)
            if default_req not in group_requirements:
                group_requirements[default_req] = {'daily': {'window': 1}}
    
    logging.info(f"因子分组完成，共 {len(factor_groups)} 个数据需求组")
    
    total_start_time = time.time()
    
    # 按数据需求组处理因子
    for req_key, factor_classes in factor_groups.items():
        data_requirements = group_requirements[req_key]
        logging.info(f"处理数据需求组: {list(data_requirements.keys())}，包含 {len(factor_classes)} 个因子")
        print(f"print:处理数据需求组: {list(data_requirements.keys())}，包含 {len(factor_classes)} 个因子")
        logging.info(f"该组数据需求: {data_requirements}")
        
        # 加载该组所需的数据（使用该组的最大window值）
        logging.info(f"开始加载数据: {list(data_requirements.keys())}")
        data = _load_data_for_update(data_requirements, date, length)
        logging.info(f"数据加载完成，共 {len(data)} 个数据类型")
        
        # 动态计算最优进程数
        optimal_workers = _calculate_optimal_workers(data, max_workers)
        
        # 并行更新该组的所有因子
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # 提交所有任务
            future_to_factor = {
                executor.submit(_update_factor_daily, factor_class, data, date): factor_class
                for factor_class in factor_classes
            }
            
            # 收集结果
            completed_factors = []
            for future in as_completed(future_to_factor):
                factor_name, elapsed, log_msg = future.result()
                # 在主进程中输出日志
                logging.info(log_msg)
                if elapsed >= 0:
                    completed_factors.append((factor_name, elapsed))
                else:
                    logging.error(f"因子 {factor_name} 更新失败")
        
        # 输出该组的统计信息
        if completed_factors:
            total_time = sum(elapsed for _, elapsed in completed_factors)
            avg_time = total_time / len(completed_factors)
            successful_count = len(completed_factors)
            failed_count = len(factor_classes) - successful_count
            logging.info(f"数据需求组 {list(data_requirements.keys())} 更新完成，"
                        f"成功: {successful_count} 个因子，失败: {failed_count} 个因子，"
                        f"平均耗时 {avg_time:.2f} 秒，总耗时 {total_time:.2f} 秒")
            
            # 输出每个因子的详细耗时
            for factor_name, elapsed in sorted(completed_factors, key=lambda x: x[1], reverse=True):
                logging.info(f"  - {factor_name}: {elapsed:.2f} 秒")
    
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    
    # 统计总体结果
    total_factors = sum(len(factor_classes) for factor_classes in factor_groups.values())
    logging.info(f"所有因子增量更新完成，总耗时 {total_elapsed:.2f} 秒")
    logging.info(f"总计处理 {total_factors} 个因子，平均每个因子耗时 {total_elapsed/total_factors:.2f} 秒") 