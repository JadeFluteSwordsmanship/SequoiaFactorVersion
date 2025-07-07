# -*- encoding: UTF-8 -*-
"""
数据清理工具
用于清理和整理股票数据库中的异常数据
"""

import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import akshare as ak
from utils import setup_logging, get_trading_dates
from settings import config

def check_non_trading_days_in_stock(stock_code: str, data_type: str = 'daily_qfq', 
                                   trading_dates: Optional[List[str]] = None) -> Tuple[int, List[str]]:
    """
    检查单只股票数据中的非交易日
    Args:
        stock_code: 股票代码
        data_type: 数据类型，'daily_qfq' 或 'minute'
        trading_dates: 交易日列表，如果为None则自动获取
    Returns:
        (非交易日数量, 非交易日列表)
    """
    data_dir = config.get('data_dir', 'E:/data')
    file_path = os.path.join(data_dir, data_type, f'{stock_code}.parquet')
    
    if not os.path.exists(file_path):
        logging.warning(f"文件不存在: {file_path}")
        return 0, []
    
    try:
        df = pd.read_parquet(file_path)
        
        # 确定日期列名
        if data_type == 'daily_qfq':
            date_col = '日期' if '日期' in df.columns else 'trade_date'
        else:
            date_col = '时间' if '时间' in df.columns else 'datetime'
        
        if date_col not in df.columns:
            logging.warning(f"股票 {stock_code} 未找到日期列，可用列: {list(df.columns)}")
            return 0, []
        
        # 转换日期格式
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        if df.empty:
            return 0, []
        
        # 获取数据的时间范围
        min_date = df[date_col].min().strftime('%Y-%m-%d')
        max_date = df[date_col].max().strftime('%Y-%m-%d')
        
        # 如果没有提供交易日列表，则获取
        if trading_dates is None:
            trading_dates = get_trading_dates(min_date, max_date)
        
        if not trading_dates:
            logging.warning(f"无法获取交易日历，跳过股票 {stock_code}")
            return 0, []
        
        # 转换为set提高查找效率
        trading_dates_set = set(trading_dates)
        
        # 检查非交易日
        df['date_str'] = df[date_col].dt.strftime('%Y-%m-%d')
        non_trading_dates = df[~df['date_str'].isin(trading_dates_set)]['date_str'].unique().tolist()
        
        return len(non_trading_dates), non_trading_dates
        
    except Exception as e:
        logging.error(f"检查股票 {stock_code} 非交易日失败: {e}")
        return 0, []

def clean_non_trading_days_from_stock(stock_code: str, data_type: str = 'daily_qfq',
                                    trading_dates: Optional[List[str]] = None,
                                    backup: bool = True) -> Dict:
    """
    清理单只股票数据中的非交易日数据
    Args:
        stock_code: 股票代码
        data_type: 数据类型，'daily_qfq' 或 'minute'
        trading_dates: 交易日列表，如果为None则自动获取
        backup: 是否备份原文件
    Returns:
        清理结果字典
    """
    data_dir = config.get('data_dir', 'E:/data')
    file_path = os.path.join(data_dir, data_type, f'{stock_code}.parquet')
    
    if not os.path.exists(file_path):
        return {'status': 'error', 'message': f'文件不存在: {file_path}'}
    
    try:
        # 读取数据
        df = pd.read_parquet(file_path)
        original_count = len(df)
        
        # 确定日期列名
        if data_type == 'daily_qfq':
            date_col = '日期' if '日期' in df.columns else 'trade_date'
        else:
            date_col = '时间' if '时间' in df.columns else 'datetime'
        
        if date_col not in df.columns:
            return {'status': 'error', 'message': f'未找到日期列，可用列: {list(df.columns)}'}
        
        # 转换日期格式
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        if df.empty:
            return {'status': 'success', 'removed_count': 0, 'remaining_count': 0}
        
        # 获取数据的时间范围
        min_date = df[date_col].min().strftime('%Y-%m-%d')
        max_date = df[date_col].max().strftime('%Y-%m-%d')
        
        # 如果没有提供交易日列表，则获取
        if trading_dates is None:
            trading_dates = get_trading_dates(min_date, max_date)
        
        if not trading_dates:
            return {'status': 'error', 'message': '无法获取交易日历'}
        
        # 转换为set提高查找效率
        trading_dates_set = set(trading_dates)
        
        # 过滤非交易日数据
        df['date_str'] = df[date_col].dt.strftime('%Y-%m-%d')
        df_cleaned = df[df['date_str'].isin(trading_dates_set)].copy()
        df_cleaned = df_cleaned.drop('date_str', axis=1)
        
        removed_count = original_count - len(df_cleaned)
        
        # 备份原文件
        if backup and removed_count > 0:
            backup_path = file_path.replace('.parquet', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet')
            df.to_parquet(backup_path, index=False)
            logging.info(f"已备份原文件到: {backup_path}")
        
        # 保存清理后的数据
        if removed_count > 0:
            df_cleaned.to_parquet(file_path, index=False)
            logging.info(f"股票 {stock_code} 清理完成，删除了 {removed_count} 条非交易日数据")
        
        return {
            'status': 'success',
            'original_count': original_count,
            'removed_count': removed_count,
            'remaining_count': len(df_cleaned),
            'backup_path': backup_path if backup and removed_count > 0 else None
        }
        
    except Exception as e:
        logging.error(f"清理股票 {stock_code} 非交易日数据失败: {e}")
        return {'status': 'error', 'message': str(e)}

def scan_all_stocks_for_non_trading_days(data_type: str = 'daily_qfq') -> Dict:
    """
    扫描所有股票数据，检查非交易日情况
    Args:
        data_type: 数据类型，'daily_qfq' 或 'minute'
    Returns:
        扫描结果字典
    """
    data_dir = config.get('data_dir', 'E:/data')
    target_dir = os.path.join(data_dir, data_type)
    
    if not os.path.exists(target_dir):
        return {'error': f'目录不存在: {target_dir}'}
    
    # 获取所有股票文件
    stock_files = [f for f in os.listdir(target_dir) if f.endswith('.parquet')]
    stock_codes = [f.replace('.parquet', '') for f in stock_files]
    
    logging.info(f"开始扫描 {len(stock_codes)} 只股票的{data_type}数据")
    
    # 获取全局交易日历（用于所有股票）
    if stock_codes:
        # 读取第一只股票获取时间范围
        sample_file = os.path.join(target_dir, stock_files[0])
        sample_df = pd.read_parquet(sample_file)
        
        if data_type == 'daily_qfq':
            date_col = '日期' if '日期' in sample_df.columns else 'trade_date'
        else:
            date_col = '时间' if '时间' in sample_df.columns else 'datetime'
        
        if date_col in sample_df.columns:
            sample_df[date_col] = pd.to_datetime(sample_df[date_col], errors='coerce')
            min_date = sample_df[date_col].min().strftime('%Y-%m-%d')
            max_date = sample_df[date_col].max().strftime('%Y-%m-%d')
            trading_dates = get_trading_dates(min_date, max_date)
        else:
            trading_dates = None
    else:
        trading_dates = None
    
    # 并行检查所有股票
    results = {}
    stocks_with_issues = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_stock = {
            executor.submit(check_non_trading_days_in_stock, code, data_type, trading_dates): code 
            for code in stock_codes
        }
        
        pbar = tqdm(as_completed(future_to_stock), total=len(stock_codes), desc=f"扫描{data_type}数据")
        for future in pbar:
            stock = future_to_stock[future]
            try:
                non_trading_count, non_trading_dates = future.result()
                results[stock] = {
                    'non_trading_count': non_trading_count,
                    'non_trading_dates': non_trading_dates
                }
                if non_trading_count > 0:
                    stocks_with_issues.append(stock)
            except Exception as e:
                logging.error(f"检查股票 {stock} 失败: {e}")
                results[stock] = {'error': str(e)}
            finally:
                pbar.set_postfix(issues=len(stocks_with_issues))
    
    # 统计结果
    total_issues = sum(r.get('non_trading_count', 0) for r in results.values() if isinstance(r, dict) and 'non_trading_count' in r)
    
    return {
        'total_stocks': len(stock_codes),
        'stocks_with_issues': len(stocks_with_issues),
        'total_issues': total_issues,
        'stocks_with_issues_list': stocks_with_issues,
        'detailed_results': results
    }

def clean_all_stocks_non_trading_days(data_type: str = 'daily', 
                                    backup: bool = True,
                                    dry_run: bool = False) -> Dict:
    """
    清理所有股票数据中的非交易日数据
    Args:
        data_type: 数据类型，'daily' 或 'minute'
        backup: 是否备份原文件
        dry_run: 是否仅预览，不实际修改文件
    Returns:
        清理结果字典
    """
    data_dir = config.get('data_dir', 'E:/data')
    target_dir = os.path.join(data_dir, data_type)
    
    if not os.path.exists(target_dir):
        return {'error': f'目录不存在: {target_dir}'}
    
    # 获取所有股票文件
    stock_files = [f for f in os.listdir(target_dir) if f.endswith('.parquet')]
    stock_codes = [f.replace('.parquet', '') for f in stock_files]
    
    logging.info(f"开始清理 {len(stock_codes)} 只股票的{data_type}数据")
    if dry_run:
        logging.info("DRY RUN模式：仅预览，不会修改文件")
    
    # 获取全局交易日历
    if stock_codes:
        sample_file = os.path.join(target_dir, stock_files[0])
        sample_df = pd.read_parquet(sample_file)
        
        if data_type == 'daily':
            date_col = '日期' if '日期' in sample_df.columns else 'trade_date'
        else:
            date_col = '时间' if '时间' in sample_df.columns else 'datetime'
        
        if date_col in sample_df.columns:
            sample_df[date_col] = pd.to_datetime(sample_df[date_col], errors='coerce')
            min_date = sample_df[date_col].min().strftime('%Y-%m-%d')
            max_date = sample_df[date_col].max().strftime('%Y-%m-%d')
            trading_dates = get_trading_dates(min_date, max_date)
        else:
            trading_dates = None
    else:
        trading_dates = None
    
    # 并行清理所有股票
    results = {}
    total_removed = 0
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:  # 减少并发数，避免文件冲突
        future_to_stock = {
            executor.submit(clean_non_trading_days_from_stock, code, data_type, trading_dates, backup): code 
            for code in stock_codes
        }
        
        pbar = tqdm(as_completed(future_to_stock), total=len(stock_codes), desc=f"清理{data_type}数据")
        for future in pbar:
            stock = future_to_stock[future]
            try:
                result = future.result()
                results[stock] = result
                
                if result['status'] == 'success':
                    success_count += 1
                    removed_count = result.get('removed_count', 0)
                    total_removed += removed_count
                else:
                    error_count += 1
                    
            except Exception as e:
                logging.error(f"清理股票 {stock} 失败: {e}")
                results[stock] = {'status': 'error', 'message': str(e)}
                error_count += 1
            finally:
                pbar.set_postfix(
                    success=success_count, 
                    error=error_count, 
                    removed=total_removed
                )
    
    return {
        'total_stocks': len(stock_codes),
        'success_count': success_count,
        'error_count': error_count,
        'total_removed': total_removed,
        'detailed_results': results
    }

    """
    分析数据质量，包括重复数据、缺失值、异常值等
    Args:
        data_type: 数据类型，'daily' 或 'minute'
    Returns:
        数据质量分析结果
    """
    data_dir = config.get('data_dir', 'E:/data')
    target_dir = os.path.join(data_dir, data_type)
    
    if not os.path.exists(target_dir):
        return {'error': f'目录不存在: {target_dir}'}
    
    stock_files = [f for f in os.listdir(target_dir) if f.endswith('.parquet')]
    stock_codes = [f.replace('.parquet', '') for f in stock_files]
    
    quality_report = {
        'total_stocks': len(stock_codes),
        'sample_analysis': {},
        'common_issues': {}
    }
    
    # 分析前10只股票作为样本
    sample_codes = stock_codes[:10] if len(stock_codes) >= 10 else stock_codes
    
    for code in sample_codes:
        file_path = os.path.join(target_dir, f'{code}.parquet')
        try:
            df = pd.read_parquet(file_path)
            
            # 基本统计
            quality_report['sample_analysis'][code] = {
                'total_rows': len(df),
                'duplicate_rows': len(df[df.duplicated()]),
                'null_counts': df.isnull().sum().to_dict(),
                'date_range': None,
                'price_range': None
            }
            
            # 日期范围
            if data_type == 'daily':
                date_col = '日期' if '日期' in df.columns else 'trade_date'
            else:
                date_col = '时间' if '时间' in df.columns else 'datetime'
            
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                valid_dates = df[date_col].dropna()
                if not valid_dates.empty:
                    quality_report['sample_analysis'][code]['date_range'] = {
                        'min': valid_dates.min().strftime('%Y-%m-%d'),
                        'max': valid_dates.max().strftime('%Y-%m-%d')
                    }
            
            # 价格范围
            price_cols = ['开盘', '收盘', '最高', '最低'] if data_type == 'daily' else ['开盘', '收盘', '最高', '最低']
            price_data = []
            for col in price_cols:
                if col in df.columns:
                    price_data.extend(df[col].dropna().tolist())
            
            if price_data:
                quality_report['sample_analysis'][code]['price_range'] = {
                    'min': min(price_data),
                    'max': max(price_data)
                }
                
        except Exception as e:
            logging.error(f"分析股票 {code} 数据质量失败: {e}")
            quality_report['sample_analysis'][code] = {'error': str(e)}
    
    return quality_report

if __name__ == "__main__":
    # 设置日志
    setup_logging('data_cleaner')
    
    print("=== 数据清理工具 ===")
    
    
    