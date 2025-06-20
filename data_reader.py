# -*- encoding: UTF-8 -*-
"""
股票数据读取工具
支持读取Parquet格式的股票数据，自动推断数据类型，提供灵活的查询接口
"""

import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import yaml
from typing import List, Dict, Optional, Union
from utils import setup_logging

# 设置日志
setup_logging('data_reader')

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logging.warning("DuckDB未安装，相关功能不可用。请运行: pip install duckdb")

def load_config():
    """加载配置文件"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        return {'data_dir': 'E:/data'}

def infer_stock_data_schema(parquet_path: str) -> Dict:
    """
    推断Parquet文件的数据结构
    Args:
        parquet_path: Parquet文件路径
    Returns:
        包含列名、数据类型、数据统计的字典
    """
    try:
        # 读取Parquet文件的schema
        df = pd.read_parquet(parquet_path)
        
        schema_info = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'sample_data': {}
        }
        
        # 为每列添加样本数据
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                # 字符串类型，显示前几个非空值
                sample_values = df[col].dropna().head(3).tolist()
            elif df[col].dtype in ['datetime64[ns]', 'datetime64']:
                # 时间类型，显示时间范围
                sample_values = {
                    'min': df[col].min().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(df[col].min()) else None,
                    'max': df[col].max().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(df[col].max()) else None
                }
            else:
                # 数值类型，显示统计信息
                sample_values = {
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                }
            
            schema_info['sample_data'][col] = sample_values
        
        return schema_info
        
    except Exception as e:
        logging.error(f"推断文件 {parquet_path} 结构失败: {e}")
        return {}

def list_available_stocks(data_type: str = 'daily') -> List[str]:
    """
    列出可用的股票代码
    Args:
        data_type: 'daily' 或 'minute'
    Returns:
        股票代码列表
    """
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    target_dir = os.path.join(data_dir, data_type)
    
    if not os.path.exists(target_dir):
        logging.warning(f"目录不存在: {target_dir}")
        return []
    
    stock_files = [f for f in os.listdir(target_dir) if f.endswith('.parquet')]
    stock_codes = [f.replace('.parquet', '') for f in stock_files]
    
    logging.info(f"找到 {len(stock_codes)} 只股票的{data_type}数据")
    return sorted(stock_codes)

def read_stock_data(stock_code: str, data_type: str = 'daily', 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    读取指定股票的数据
    Args:
        stock_code: 股票代码
        data_type: 'daily' 或 'minute'
        start_date: 开始日期 (格式: '2023-01-01' 或 '2023-01-01 09:30:00')
        end_date: 结束日期
        columns: 指定要读取的列名
    Returns:
        DataFrame
    """
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    parquet_path = os.path.join(data_dir, data_type, f'{stock_code}.parquet')
    
    if not os.path.exists(parquet_path):
        logging.error(f"文件不存在: {parquet_path}")
        return pd.DataFrame()
    
    try:
        # 读取数据
        if columns:
            df = pd.read_parquet(parquet_path, columns=columns)
        else:
            df = pd.read_parquet(parquet_path)
        
        # 确定时间列名
        time_col = None
        if data_type == 'daily':
            time_col = '日期' if '日期' in df.columns else '时间'
        else:
            time_col = '时间' if '时间' in df.columns else '日期'
        
        if time_col not in df.columns:
            logging.error(f"未找到时间列，可用列: {list(df.columns)}")
            return df
        
        # 过滤时间范围
        if start_date or end_date:
            df[time_col] = pd.to_datetime(df[time_col])
            
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df[time_col] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df[time_col] <= end_dt]
        
        logging.info(f"成功读取股票 {stock_code} 的{data_type}数据，共 {len(df)} 条记录")
        return df
        
    except Exception as e:
        logging.error(f"读取股票 {stock_code} 数据失败: {e}")
        return pd.DataFrame()

def query_multiple_stocks(stock_codes: List[str], data_type: str = 'daily',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    批量读取多只股票的数据
    Args:
        stock_codes: 股票代码列表
        data_type: 'daily' 或 'minute'
        start_date: 开始日期
        end_date: 结束日期
        columns: 指定列名
    Returns:
        股票代码到DataFrame的字典
    """
    results = {}
    failed_stocks = []
    
    for stock_code in stock_codes:
        df = read_stock_data(stock_code, data_type, start_date, end_date, columns)
        if not df.empty:
            results[stock_code] = df
        else:
            failed_stocks.append(stock_code)
    
    if failed_stocks:
        logging.warning(f"以下股票读取失败: {failed_stocks}")
    
    logging.info(f"成功读取 {len(results)} 只股票的数据")
    return results

def get_data_summary(data_type: str = 'daily') -> Dict:
    """
    获取数据概览
    Args:
        data_type: 'daily' 或 'minute'
    Returns:
        数据概览信息
    """
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    target_dir = os.path.join(data_dir, data_type)
    
    if not os.path.exists(target_dir):
        return {'error': f'目录不存在: {target_dir}'}
    
    stock_files = [f for f in os.listdir(target_dir) if f.endswith('.parquet')]
    
    summary = {
        'data_type': data_type,
        'total_stocks': len(stock_files),
        'total_size_mb': 0,
        'sample_schema': None
    }
    
    # 计算总大小
    for file in stock_files:
        file_path = os.path.join(target_dir, file)
        summary['total_size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
    
    # 获取样本schema
    if stock_files:
        sample_file = os.path.join(target_dir, stock_files[0])
        summary['sample_schema'] = infer_stock_data_schema(sample_file)
    
    return summary

def query_with_duckdb(sql_query: str, data_type: str = 'daily') -> pd.DataFrame:
    """
    使用DuckDB查询数据（如果安装了duckdb）
    Args:
        sql_query: SQL查询语句
        data_type: 'daily' 或 'minute'
    Returns:
        DataFrame
    """
    if not DUCKDB_AVAILABLE:
        logging.error("DuckDB未安装，请运行: pip install duckdb")
        return pd.DataFrame()
    
    try:
        config = load_config()
        data_dir = config.get('data_dir', 'E:/data')
        target_dir = os.path.join(data_dir, data_type)
        
        # 创建DuckDB连接
        con = duckdb.connect(':memory:')
        
        # 注册目录中的所有Parquet文件
        parquet_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.parquet')]
        
        if not parquet_files:
            logging.error(f"未找到{data_type}数据文件")
            return pd.DataFrame()
        
        # 创建视图
        con.execute(f"CREATE VIEW stock_data AS SELECT * FROM read_parquet({parquet_files})")
        
        # 执行查询
        result = con.execute(sql_query).fetchdf()
        con.close()
        
        logging.info(f"DuckDB查询完成，返回 {len(result)} 条记录")
        return result
        
    except Exception as e:
        logging.error(f"DuckDB查询失败: {e}")
        return pd.DataFrame()

def export_to_csv(stock_codes: List[str], data_type: str = 'daily',
                  output_dir: str = 'exports',
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> List[str]:
    """
    导出股票数据到CSV文件
    Args:
        stock_codes: 股票代码列表
        data_type: 'daily' 或 'minute'
        output_dir: 输出目录
        start_date: 开始日期
        end_date: 结束日期
    Returns:
        导出的文件路径列表
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    exported_files = []
    
    for stock_code in stock_codes:
        df = read_stock_data(stock_code, data_type, start_date, end_date)
        if not df.empty:
            output_file = os.path.join(output_dir, f'{stock_code}_{data_type}.csv')
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            exported_files.append(output_file)
            logging.info(f"导出 {stock_code} 到 {output_file}")
    
    return exported_files

def get_daily_data(codes: List[str], end_date: str, window: int) -> pd.DataFrame:
    """
    读取codes对应的日线数据，每只股票返回end_date之前最新window行，拼接为一个DataFrame。
    返回的DataFrame列名为：['trade_date', 'ts_code', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'chg', 'turnover']
    - 'trade_date': 交易日期，pd.Timestamp
    - 'ts_code': 股票代码，str
    其他为常见行情字段。
    """
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    daily_dir = os.path.join(data_dir, 'daily')
    COLUMN_MAP = {
        '日期': 'trade_date',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount',
        '振幅': 'amplitude',
        '涨跌幅': 'pct_chg',
        '涨跌额': 'chg',
        '换手率': 'turnover',
    }
    target_cols = ['trade_date', 'ts_code', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'chg', 'turnover']
    end_dt = pd.to_datetime(end_date)
    dfs = []
    for code in codes:
        file_path = os.path.join(daily_dir, f'{code}.parquet')
        if not os.path.exists(file_path):
            logging.warning(f"日线数据文件不存在: {file_path}")
            continue
        try:
            df = pd.read_parquet(file_path)
            # 列名映射
            df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
            # 股票代码列处理，只保留ts_code
            if '股票代码' in df.columns:
                df['ts_code'] = df['股票代码'].astype(str)
            elif '代码' in df.columns:
                df['ts_code'] = df['代码'].astype(str)
            elif 'symbol' in df.columns:
                df['ts_code'] = df['symbol'].astype(str)
            else:
                df['ts_code'] = str(code)
            # 日期列处理
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            else:
                raise ValueError(f"未找到日期列: {file_path}")
            # 只保留end_date之前的数据
            df = df[df['trade_date'] <= end_dt]
            # 只保留目标列
            for col in target_cols:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[target_cols]
            df = df.sort_values('trade_date').tail(window)
            dfs.append(df)
        except Exception as e:
            logging.error(f"读取日线数据失败: {file_path}, {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=target_cols)

def get_minute_data(codes: List[str], end_date: str, window: int) -> pd.DataFrame:
    """
    读取codes对应的分钟线数据，每只股票返回end_date之前最新window行，拼接为一个DataFrame。
    返回的DataFrame列名为：['datetime', 'open', 'close', 'high', 'low', 'volume', 'amount', 'avg_price', 'ts_code']
    - 'datetime': 分钟时间，pd.Timestamp
    - 'ts_code': 股票代码，str
    其他为常见行情字段。
    """
    config = load_config()
    data_dir = config.get('data_dir', 'E:/data')
    minute_dir = os.path.join(data_dir, 'minute')
    COLUMN_MAP = {
        '时间': 'datetime',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount',
        '均价': 'avg_price',
    }
    target_cols = ['datetime', 'open', 'close', 'high', 'low', 'volume', 'amount', 'avg_price', 'ts_code']
    end_dt = pd.to_datetime(end_date)
    dfs = []
    for code in codes:
        file_path = os.path.join(minute_dir, f'{code}.parquet')
        if not os.path.exists(file_path):
            logging.warning(f"分钟数据文件不存在: {file_path}")
            continue
        try:
            df = pd.read_parquet(file_path)
            # 列名映射
            df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
            # 股票代码列处理，只保留ts_code
            if '股票代码' in df.columns:
                df['ts_code'] = df['股票代码'].astype(str)
            elif '代码' in df.columns:
                df['ts_code'] = df['代码'].astype(str)
            elif 'symbol' in df.columns:
                df['ts_code'] = df['symbol'].astype(str)
            else:
                df['ts_code'] = str(code)
            # 时间列处理
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            else:
                raise ValueError(f"未找到时间列: {file_path}")
            # 只保留end_date之前的数据
            df = df[df['datetime'] <= end_dt]
            # 只保留目标列
            for col in target_cols:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[target_cols]
            df = df.sort_values('datetime').tail(window)
            dfs.append(df)
        except Exception as e:
            logging.error(f"读取分钟数据失败: {file_path}, {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=target_cols)

def get_lhb_data(codes: List[str], end_date: str, window: int) -> pd.DataFrame:
    """
    读取codes在end_date之前window天的龙虎榜数据。
    返回DataFrame，需包含['code', 'trade_date', ...]
    """
    # TODO: 实现具体读取逻辑
    raise NotImplementedError

if __name__ == "__main__":
    # 示例用法
    logging.info("=== 股票数据读取工具 ===")
    
    # 1. 查看可用股票
    daily_stocks = list_available_stocks('daily')
    minute_stocks = list_available_stocks('minute')
    
    print(f"日线数据股票数量: {len(daily_stocks)}")
    print(f"分钟数据股票数量: {len(minute_stocks)}")
    
    # 2. 获取数据概览
    daily_summary = get_data_summary('daily')
    print(f"日线数据概览: {daily_summary}")
    
    # 3. 读取单只股票数据
    if daily_stocks:
        sample_stock = daily_stocks[0]
        df = read_stock_data(sample_stock, 'minute', start_date='2023-01-01')
        print(f"股票 {sample_stock} 数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        if not df.empty:
            print(f"前5行数据:\n{df.head()}")
    
    # 4. 批量读取数据
    if len(daily_stocks) >= 3:
        sample_stocks = daily_stocks[:3]
        multi_df = query_multiple_stocks(sample_stocks, 'daily', start_date='2023-01-01')
        print(f"批量读取了 {len(multi_df)} 只股票的数据")
    
    # 5. 使用DuckDB查询（如果可用）
    if DUCKDB_AVAILABLE:
        duckdb_result = query_with_duckdb(
            "SELECT 代码, COUNT(*) as record_count FROM stock_data GROUP BY 代码 LIMIT 5",
            'daily'
        )
        if not duckdb_result.empty:
            print(f"DuckDB查询结果:\n{duckdb_result}")
    else:
        print("DuckDB查询不可用") 