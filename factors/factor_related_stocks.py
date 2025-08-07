import pandas as pd
import numpy as np
from .factor_base import FactorBase
from .numba_utils import ts_rank_numba, rolling_corr_numba

class IndustryRank20(FactorBase):
    """
    IndustryRank20：行业排名因子。
    计算股票在其所属行业中的相对表现排名（百分比）。
    公式：行业排名 = ts_rank(股票收益率, 行业内所有股票, pct=True)
    方向（direction=1）：行业排名越高，未来收益可能越高。
    值域：[0, 1]，0.15表示行业前15%。
    """
    name = "IndustryRank20"
    direction = 1  # 行业排名高，未来收益可能高
    description = (
        "IndustryRank20：行业排名因子。\n"
        "计算股票在其所属行业中的相对表现排名（百分比）。\n"
        "公式：行业排名 = ts_rank(股票收益率, 行业内所有股票, pct=True)\n"
        "方向（direction=1）：行业排名越高，未来收益可能越高。\n"
        "值域：[0, 1]，0.15表示行业前15%。"
    )
    data_requirements = {
        'daily': {'window': 20},  # 需要20天数据计算收益率
        'stock_basic': {'window': 1}  # 行业分类数据
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        stock_basic_df = data['stock_basic'].copy()
        
        # 使用已有的pct_chg字段计算20日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并行业数据
        df = daily_df.merge(stock_basic_df[['stock_code', 'industry']], 
                           on=['stock_code'], how='left')
        
        # 按日期和行业分组计算排名
        df['value'] = df.groupby(['trade_date', 'industry'])['returns_20d'].rank(method='average', pct=True)
        df['factor'] = self.name
        
        # 只保留有效数据
        result = df.dropna(subset=['value', 'industry'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)
    
class IndustryRank8(FactorBase):
    """
    IndustryRank8：行业排名因子。
    计算股票在其所属行业中的相对表现排名（百分比）。
    公式：行业排名 = ts_rank(股票收益率, 行业内所有股票, pct=True)
    方向（direction=1）：行业排名越高，未来收益可能越高。
    值域：[0, 1]，0.15表示行业前15%。
    """
    name = "IndustryRank8"
    direction = 1  # 行业排名高，未来收益可能高
    description = (
        "IndustryRank8：行业排名因子。\n"
        "计算股票在其所属行业中的相对表现排名（百分比）。\n"
        "公式：行业排名 = ts_rank(股票收益率, 行业内所有股票, pct=True)\n"
        "方向（direction=1）：行业排名越高，未来收益可能越高。\n"
        "值域：[0, 1]，0.15表示行业前15%。"
    )
    data_requirements = {
        'daily': {'window': 8},  # 需要8天数据计算收益率
        'stock_basic': {'window': 1}  # 行业分类数据
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        stock_basic_df = data['stock_basic'].copy()
        
        # 使用已有的pct_chg字段计算8日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_8d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(8).sum().reset_index(0, drop=True)
        
        # 合并行业数据
        df = daily_df.merge(stock_basic_df[['stock_code', 'industry']], 
                           on=['stock_code'], how='left')
        
        # 按日期和行业分组计算排名
        df['value'] = df.groupby(['trade_date', 'industry'])['returns_8d'].rank(method='average', pct=True)
        df['factor'] = self.name
        
        # 只保留有效数据
        result = df.dropna(subset=['value', 'industry'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class IndustryRank2(FactorBase):
    """
    IndustryRank2：行业排名因子。
    计算股票在其所属行业中的相对表现排名（百分比）。
    公式：行业排名 = ts_rank(股票收益率, 行业内所有股票, pct=True)
    方向（direction=1）：行业排名越高，未来收益可能越高。
    值域：[0, 1]，0.15表示行业前15%。
    """
    name = "IndustryRank2"
    direction = 1  # 行业排名高，未来收益可能高
    description = (
        "IndustryRank2：行业排名因子。\n"
        "计算股票在其所属行业中的相对表现排名（百分比）。\n"
        "公式：行业排名 = ts_rank(股票收益率, 行业内所有股票, pct=True)\n"
        "方向（direction=1）：行业排名越高，未来收益可能越高。\n"
        "值域：[0, 1]，0.15表示行业前15%。"
    )
    data_requirements = {
        'daily': {'window': 3},  # 需要3天数据计算收益率
        'stock_basic': {'window': 1}  # 行业分类数据
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        stock_basic_df = data['stock_basic'].copy()
        
        # 使用已有的pct_chg字段计算2日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_2d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(2).sum().reset_index(0, drop=True)
        
        # 合并行业数据
        df = daily_df.merge(stock_basic_df[['stock_code', 'industry']], 
                           on=['stock_code'], how='left')
        
        # 按日期和行业分组计算排名
        df['value'] = df.groupby(['trade_date', 'industry'])['returns_2d'].rank(method='average', pct=True)
        df['factor'] = self.name
        
        # 只保留有效数据
        result = df.dropna(subset=['value', 'industry'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class RegionalRank20(FactorBase):
    """
    RegionalRank20：地域排名因子。
    计算股票在其所属地域（省份）中的相对表现排名（百分比）。
    公式：地域排名 = ts_rank(股票收益率, 地域内所有股票, pct=True)
    方向（direction=1）：地域排名越高，未来收益可能越高。
    值域：[0, 1]，0.15表示地域前15%。
    """
    name = "RegionalRank20"
    direction = 1  # 地域排名高，未来收益可能高
    description = (
        "RegionalRank20：地域排名因子。\n"
        "计算股票在其所属地域（省份）中的相对表现排名（百分比）。\n"
        "公式：地域排名 = ts_rank(股票收益率, 地域内所有股票, pct=True)\n"
        "方向（direction=1）：地域排名越高，未来收益可能越高。\n"
        "值域：[0, 1]，0.15表示地域前15%。"
    )
    data_requirements = {
        'daily': {'window': 20},  # 需要20天数据计算收益率
        'company_info': {'window': 1}  # 使用province作为地域分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        company_info_df = data['company_info'].copy()
        
        # 使用已有的pct_chg字段计算20日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并地域数据（使用province作为地域分类）
        df = daily_df.merge(company_info_df[['stock_code', 'province']], 
                           on=['stock_code'], how='left')
        
        # 按日期和地域分组计算排名
        df['value'] = df.groupby(['trade_date', 'province'])['returns_20d'].rank(method='average', pct=True)
        df['factor'] = self.name
        
        # 只保留有效数据
        result = df.dropna(subset=['value', 'province'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class RegionalRank8(FactorBase):
    """
    RegionalRank8：地域排名因子。
    计算股票在其所属地域（省份）中的相对表现排名（百分比）。
    公式：地域排名 = ts_rank(股票收益率, 地域内所有股票, pct=True)
    方向（direction=1）：地域排名越高，未来收益可能越高。
    值域：[0, 1]，0.15表示地域前15%。
    """
    name = "RegionalRank8"
    direction = 1  # 地域排名高，未来收益可能高
    description = (
        "RegionalRank8：地域排名因子。\n"
        "计算股票在其所属地域（省份）中的相对表现排名（百分比）。\n"
        "公式：地域排名 = ts_rank(股票收益率, 地域内所有股票, pct=True)\n"
        "方向（direction=1）：地域排名越高，未来收益可能越高。\n"
        "值域：[0, 1]，0.15表示地域前15%。"
    )
    data_requirements = {
        'daily': {'window': 8},  # 需要8天数据计算收益率
        'company_info': {'window': 1}  # 使用province作为地域分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        company_info_df = data['company_info'].copy()
        
        # 使用已有的pct_chg字段计算8日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_8d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(8).sum().reset_index(0, drop=True)
        
        # 合并地域数据（使用province作为地域分类）
        df = daily_df.merge(company_info_df[['stock_code', 'province']], 
                           on=['stock_code'], how='left')
        
        # 按日期和地域分组计算排名
        df['value'] = df.groupby(['trade_date', 'province'])['returns_8d'].rank(method='average', pct=True)
        df['factor'] = self.name
        
        # 只保留有效数据
        result = df.dropna(subset=['value', 'province'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class RegionalRank2(FactorBase):
    """
    RegionalRank2：地域排名因子。
    计算股票在其所属地域（省份）中的相对表现排名（百分比）。
    公式：地域排名 = ts_rank(股票收益率, 地域内所有股票, pct=True)
    方向（direction=1）：地域排名越高，未来收益可能越高。
    值域：[0, 1]，0.15表示地域前15%。
    """
    name = "RegionalRank2"
    direction = 1  # 地域排名高，未来收益可能高
    description = (
        "RegionalRank2：地域排名因子。\n"
        "计算股票在其所属地域（省份）中的相对表现排名（百分比）。\n"
        "公式：地域排名 = ts_rank(股票收益率, 地域内所有股票, pct=True)\n"
        "方向（direction=1）：地域排名越高，未来收益可能越高。\n"
        "值域：[0, 1]，0.15表示地域前15%。"
    )
    data_requirements = {
        'daily': {'window': 3},  # 需要3天数据计算收益率
        'company_info': {'window': 1}  # 使用province作为地域分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        company_info_df = data['company_info'].copy()
        
        # 使用已有的pct_chg字段计算2日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_2d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(2).sum().reset_index(0, drop=True)
        
        # 合并地域数据（使用province作为地域分类）
        df = daily_df.merge(company_info_df[['stock_code', 'province']], 
                           on=['stock_code'], how='left')
        
        # 按日期和地域分组计算排名
        df['value'] = df.groupby(['trade_date', 'province'])['returns_2d'].rank(method='average', pct=True)
        df['factor'] = self.name
        
        # 只保留有效数据
        result = df.dropna(subset=['value', 'province'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class SectorMomentum20(FactorBase):
    """
    SectorMomentum20：板块动量因子。
    计算股票所属板块的整体动量表现。
    公式：板块动量 = mean(板块内所有股票20日收益率)
    方向（direction=1）：板块动量越强，未来收益可能越高。
    """
    name = "SectorMomentum20"
    direction = 1  # 板块动量强，未来收益可能高
    description = (
        "SectorMomentum20：板块动量因子。\n"
        "计算股票所属板块的整体动量表现。\n"
        "公式：板块动量 = mean(板块内所有股票20日收益率)\n"
        "方向（direction=1）：板块动量越强，未来收益可能越高。"
    )
    data_requirements = {
        'daily': {'window': 20},
        'industry_member': {'window': 1}  # 使用l2_name作为板块分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        industry_member_df = data['industry_member'].copy()
        
        # 使用已有的pct_chg字段计算20日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并板块数据（使用l2_name作为板块分类）
        df = daily_df.merge(industry_member_df[['stock_code', 'l2_name']], 
                           on=['stock_code'], how='left')
        
        # 计算板块动量
        sector_momentum = df.groupby(['l2_name', 'trade_date'])['returns_20d'].mean().reset_index()
        sector_momentum = sector_momentum.rename(columns={'returns_20d': 'sector_momentum'})
        
        # 合并回原数据
        result = df.merge(sector_momentum, on=['l2_name', 'trade_date'], how='left')
        result['factor'] = self.name
        result['value'] = result['sector_momentum']
        
        # 只保留有效数据
        result = result.dropna(subset=['value', 'l2_name'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)


class SectorMomentum8(FactorBase):
    """
    SectorMomentum8：板块动量因子。
    计算股票所属板块的整体动量表现。
    公式：板块动量 = mean(板块内所有股票8日收益率)
    方向（direction=1）：板块动量越强，未来收益可能越高。
    """
    name = "SectorMomentum8"
    direction = 1  # 板块动量强，未来收益可能高
    description = (
        "SectorMomentum8：板块动量因子。\n"
        "计算股票所属板块的整体动量表现。\n"
        "公式：板块动量 = mean(板块内所有股票8日收益率)\n"
        "方向（direction=1）：板块动量越强，未来收益可能越高。"
    )
    data_requirements = {
        'daily': {'window': 8},
        'industry_member': {'window': 1}  # 使用l2_name作为板块分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        industry_member_df = data['industry_member'].copy()
        
        # 使用已有的pct_chg字段计算8日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_8d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(8).sum().reset_index(0, drop=True)
        
        # 合并板块数据（使用l2_name作为板块分类）
        df = daily_df.merge(industry_member_df[['stock_code', 'l2_name']], 
                           on=['stock_code'], how='left')
        
        # 计算板块动量
        sector_momentum = df.groupby(['l2_name', 'trade_date'])['returns_8d'].mean().reset_index()
        sector_momentum = sector_momentum.rename(columns={'returns_8d': 'sector_momentum'})
        
        # 合并回原数据
        result = df.merge(sector_momentum, on=['l2_name', 'trade_date'], how='left')
        result['factor'] = self.name
        result['value'] = result['sector_momentum']
        
        # 只保留有效数据
        result = result.dropna(subset=['value', 'l2_name'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class RegionalStrength20(FactorBase):
    """
    RegionalStrength20：地域强度因子。
    计算股票所属地域的整体表现强度。
    公式：地域强度 = mean(地域内股票20日收益率)
    方向（direction=1）：地域强度越高，未来收益可能越高。
    """
    name = "RegionalStrength20"
    direction = 1  # 地域强度高，未来收益可能高
    description = (
        "RegionalStrength20：地域强度因子。\n"
        "计算股票所属地域的整体表现强度。\n"
        "公式：地域强度 = mean(地域内股票20日收益率)\n"
        "方向（direction=1）：地域强度越高，未来收益可能越高。"
    )
    data_requirements = {
        'daily': {'window': 20},
        'company_info': {'window': 1}  # 使用province作为地域分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        company_info_df = data['company_info'].copy()
        
        # 使用已有的pct_chg字段计算20日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并地域数据（使用province作为地域分类）
        df = daily_df.merge(company_info_df[['stock_code', 'province']], 
                           on=['stock_code'], how='left')
        
        # 计算地域强度
        regional_strength = df.groupby(['province', 'trade_date'])['returns_20d'].mean().reset_index()
        regional_strength = regional_strength.rename(columns={'returns_20d': 'regional_strength'})
        
        # 合并回原数据
        result = df.merge(regional_strength, on=['province', 'trade_date'], how='left')
        result['factor'] = self.name
        result['value'] = result['regional_strength']
        
        # 只保留有效数据
        result = result.dropna(subset=['value', 'province'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class RegionalStrength2(FactorBase):
    """
    RegionalStrength2：地域强度因子。
    计算股票所属地域的整体表现强度。
    公式：地域强度 = mean(地域内股票2日收益率)
    方向（direction=1）：地域强度越高，未来收益可能越高。
    """
    name = "RegionalStrength2"
    direction = 1  # 地域强度高，未来收益可能高
    description = (
        "RegionalStrength2：地域强度因子。\n"
        "计算股票所属地域的整体表现强度。\n"
        "公式：地域强度 = mean(地域内股票2日收益率)\n"
        "方向（direction=1）：地域强度越高，未来收益可能越高。"
    )
    data_requirements = {
        'daily': {'window': 3},
        'company_info': {'window': 1}  # 使用province作为地域分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        company_info_df = data['company_info'].copy()
        
        # 使用已有的pct_chg字段计算2日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_2d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(2).sum().reset_index(0, drop=True)
        
        # 合并地域数据（使用province作为地域分类）
        df = daily_df.merge(company_info_df[['stock_code', 'province']], 
                           on=['stock_code'], how='left')
        
        # 计算地域强度
        regional_strength = df.groupby(['province', 'trade_date'])['returns_2d'].mean().reset_index()
        regional_strength = regional_strength.rename(columns={'returns_2d': 'regional_strength'})
        
        # 合并回原数据
        result = df.merge(regional_strength, on=['province', 'trade_date'], how='left')
        result['factor'] = self.name
        result['value'] = result['regional_strength']
        
        # 只保留有效数据
        result = result.dropna(subset=['value', 'province'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class IndustryCorrelation20(FactorBase):
    """
    IndustryCorrelation20：行业相关性因子。
    计算股票与其所属行业平均收益的相关性。
    公式：行业相关性 = corr(股票收益率, 行业平均收益率, 20)
    方向（direction=1）：相关性越高，跟随行业趋势越强。
    """
    name = "IndustryCorrelation20"
    direction = 1  # 相关性高，跟随行业趋势强
    description = (
        "IndustryCorrelation20：行业相关性因子。\n"
        "计算股票与其所属行业平均收益的相关性。\n"
        "公式：行业相关性 = corr(股票收益率, 行业平均收益率, 20)\n"
        "方向（direction=1）：相关性越高，跟随行业趋势越强。"
    )
    data_requirements = {
        'daily': {'window': 20},
        'stock_basic': {'window': 1}  # 行业分类数据
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        stock_basic_df = data['stock_basic'].copy()
        
        # 使用已有的pct_chg字段
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns'] = daily_df['pct_chg']
        
        # 合并行业数据
        df = daily_df.merge(stock_basic_df[['stock_code', 'industry']], 
                           on=['stock_code'], how='left')
        
        # 计算行业平均收益率
        industry_avg_returns = df.groupby(['industry', 'trade_date'])['returns'].mean().reset_index()
        industry_avg_returns = industry_avg_returns.rename(columns={'returns': 'industry_avg_returns'})
        
        # 合并行业平均收益率
        df = df.merge(industry_avg_returns, on=['industry', 'trade_date'], how='left')
        
        # 按股票计算相关性
        result_list = []
        for stock_code, stock_data in df.groupby('stock_code'):
            stock_data = stock_data.sort_values('trade_date')
            
            if len(stock_data) >= 20:
                # 计算相关性
                stock_returns = stock_data['returns'].values
                industry_returns = stock_data['industry_avg_returns'].values
                
                correlation = rolling_corr_numba(stock_returns, industry_returns, 20)
                
                # 只保留有效相关性
                valid_mask = ~np.isnan(correlation)
                if valid_mask.any():
                    valid_corr = correlation[valid_mask]
                    valid_dates = stock_data.iloc[19:]['trade_date'].values  # 从第20天开始
                    
                    for i, date in enumerate(valid_dates):
                        result_list.append({
                            'code': stock_code,
                            'date': date,
                            'factor': self.name,
                            'value': valid_corr[i]
                        })
        
        # 创建结果DataFrame
        if result_list:
            return pd.DataFrame(result_list)
        else:
            return pd.DataFrame(columns=['code', 'date', 'factor', 'value'])

class SectorRotation20(FactorBase):
    """
    SectorRotation20：板块轮动因子。
    计算板块相对强度变化，识别板块轮动机会。
    公式：板块轮动 = ts_rank(板块动量, 所有板块) - ts_rank(板块动量, 所有板块, 20)
    方向（direction=1）：轮动强度越高，板块可能处于上升期。
    """
    name = "SectorRotation20"
    direction = 1  # 轮动强度高，板块可能处于上升期
    description = (
        "SectorRotation20：板块轮动因子。\n"
        "计算板块相对强度变化，识别板块轮动机会。\n"
        "公式：板块轮动 = ts_rank(板块动量, 所有板块) - ts_rank(板块动量, 所有板块, 20)\n"
        "方向（direction=1）：轮动强度越高，板块可能处于上升期。"
    )
    data_requirements = {
        'daily': {'window': 40},
        'industry_member': {'window': 1}  # 使用l2_name作为板块分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        industry_member_df = data['industry_member'].copy()
        
        # 使用已有的pct_chg字段计算20日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并板块数据（使用l2_name作为板块分类）
        df = daily_df.merge(industry_member_df[['stock_code', 'l2_name']], 
                           on=['stock_code'], how='left')
        
        # 计算板块动量
        sector_momentum = df.groupby(['l2_name', 'trade_date'])['returns_20d'].mean().reset_index()
        sector_momentum = sector_momentum.rename(columns={'returns_20d': 'sector_momentum'})
        
        # 计算板块轮动
        sector_momentum = sector_momentum.sort_values(['l2_name', 'trade_date'])
        
        # 计算当前排名和20天前排名
        sector_momentum['current_rank'] = sector_momentum.groupby('trade_date')['sector_momentum'].rank(method='average')
        sector_momentum['past_rank'] = sector_momentum.groupby('l2_name')['current_rank'].shift(20)
        
        # 计算轮动强度
        sector_momentum['rotation_strength'] = sector_momentum['current_rank'] - sector_momentum['past_rank']
        
        # 合并回原数据
        result = df.merge(sector_momentum[['l2_name', 'trade_date', 'rotation_strength']], 
                         on=['l2_name', 'trade_date'], how='left')
        result['factor'] = self.name
        result['value'] = result['rotation_strength']
        
        # 只保留有效数据
        result = result.dropna(subset=['value', 'l2_name'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class SWIndustryRank20(FactorBase):
    """
    SWIndustryRank20：申万行业排名因子。
    计算股票在其所属申万行业中的相对表现排名（百分比）。
    公式：申万行业排名 = ts_rank(股票收益率, 申万行业内所有股票, pct=True)
    方向（direction=1）：申万行业排名越高，未来收益可能越高。
    值域：[0, 1]，0.15表示申万行业前15%。
    """
    name = "SWIndustryRank20"
    direction = 1  # 申万行业排名高，未来收益可能高
    description = (
        "SWIndustryRank20：申万行业排名因子。\n"
        "计算股票在其所属申万行业中的相对表现排名（百分比）。\n"
        "公式：申万行业排名 = ts_rank(股票收益率, 申万行业内所有股票, pct=True)\n"
        "方向（direction=1）：申万行业排名越高，未来收益可能越高。\n"
        "值域：[0, 1]，0.15表示申万行业前15%。"
    )
    data_requirements = {
        'daily': {'window': 20},  # 需要20天数据计算收益率
        'industry_member': {'window': 1}  # 申万行业分类数据
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        industry_member_df = data['industry_member'].copy()
        
        # 使用已有的pct_chg字段计算20日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并申万行业数据
        df = daily_df.merge(industry_member_df[['stock_code', 'l1_name']], 
                           on=['stock_code'], how='left')
        
        # 按日期和申万行业分组计算排名
        df['value'] = df.groupby(['trade_date', 'l1_name'])['returns_20d'].rank(method='average', pct=True)
        df['factor'] = self.name
        
        # 只保留有效数据
        result = df.dropna(subset=['value', 'l1_name'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class CityStrength20(FactorBase):
    """
    CityStrength20：城市强度因子。
    计算股票所属城市的整体表现强度。
    公式：城市强度 = mean(城市内股票20日收益率)
    方向（direction=1）：城市强度越高，未来收益可能越高。
    """
    name = "CityStrength20"
    direction = 1  # 城市强度高，未来收益可能高
    description = (
        "CityStrength20：城市强度因子。\n"
        "计算股票所属城市的整体表现强度。\n"
        "公式：城市强度 = mean(城市内股票20日收益率)\n"
        "方向（direction=1）：城市强度越高，未来收益可能越高。"
    )
    data_requirements = {
        'daily': {'window': 20},
        'company_info': {'window': 1}  # 使用city作为城市分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        company_info_df = data['company_info'].copy()
        
        # 使用已有的pct_chg字段计算20日累计收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['pct_chg'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并城市数据
        df = daily_df.merge(company_info_df[['stock_code', 'city']], 
                           on=['stock_code'], how='left')
        
        # 计算城市强度
        city_strength = df.groupby(['city', 'trade_date'])['returns_20d'].mean().reset_index()
        city_strength = city_strength.rename(columns={'returns_20d': 'city_strength'})
        
        # 合并回原数据
        result = df.merge(city_strength, on=['city', 'trade_date'], how='left')
        result['factor'] = self.name
        result['value'] = result['city_strength']
        
        # 只保留有效数据
        result = result.dropna(subset=['value', 'city'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True) 