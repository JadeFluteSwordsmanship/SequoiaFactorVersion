import pandas as pd
import numpy as np
from .factor_base import FactorBase
from .numba_utils import ts_rank_numba, rolling_corr_numba

class IndustryRank(FactorBase):
    """
    IndustryRank：行业排名因子。
    计算股票在其所属行业中的相对表现排名。
    公式：行业排名 = ts_rank(股票收益率, 行业内所有股票)
    方向（direction=1）：行业排名越高，未来收益可能越高。
    """
    name = "IndustryRank"
    direction = 1  # 行业排名高，未来收益可能高
    description = (
        "IndustryRank：行业排名因子。\n"
        "计算股票在其所属行业中的相对表现排名。\n"
        "公式：行业排名 = ts_rank(股票收益率, 行业内所有股票)\n"
        "方向（direction=1）：行业排名越高，未来收益可能越高。"
    )
    data_requirements = {
        'daily': {'window': 20},  # 需要20天数据计算收益率
        'stock_basic': {'window': 1}  # 行业分类数据
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        stock_basic_df = data['stock_basic'].copy()
        
        # 计算股票收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns'] = daily_df.groupby('stock_code')['close'].pct_change()
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['returns'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并行业数据
        df = daily_df.merge(stock_basic_df[['stock_code', 'industry']], 
                           on=['stock_code'], how='left')
        
        # 按日期和行业分组计算排名
        df['value'] = df.groupby(['trade_date', 'industry'])['returns_20d'].rank(method='average')
        df['factor'] = self.name
        
        # 只保留有效数据
        result = df.dropna(subset=['value', 'industry'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class SectorMomentum(FactorBase):
    """
    SectorMomentum：板块动量因子。
    计算股票所属板块的整体动量表现。
    公式：板块动量 = mean(板块内所有股票20日收益率)
    方向（direction=1）：板块动量越强，未来收益可能越高。
    """
    name = "SectorMomentum"
    direction = 1  # 板块动量强，未来收益可能高
    description = (
        "SectorMomentum：板块动量因子。\n"
        "计算股票所属板块的整体动量表现。\n"
        "公式：板块动量 = mean(板块内所有股票20日收益率)\n"
        "方向（direction=1）：板块动量越强，未来收益可能越高。"
    )
    data_requirements = {
        'daily': {'window': 20},
        'stock_basic': {'window': 1}  # 使用market作为板块分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        stock_basic_df = data['stock_basic'].copy()
        
        # 计算股票收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns'] = daily_df.groupby('stock_code')['close'].pct_change()
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['returns'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并板块数据（使用market作为板块分类）
        df = daily_df.merge(stock_basic_df[['stock_code', 'market']], 
                           on=['stock_code'], how='left')
        
        # 计算板块动量
        sector_momentum = df.groupby(['market', 'trade_date'])['returns_20d'].mean().reset_index()
        sector_momentum = sector_momentum.rename(columns={'returns_20d': 'sector_momentum'})
        
        # 合并回原数据
        result = df.merge(sector_momentum, on=['market', 'trade_date'], how='left')
        result['factor'] = self.name
        result['value'] = result['sector_momentum']
        
        # 只保留有效数据
        result = result.dropna(subset=['value', 'market'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class RegionalStrength(FactorBase):
    """
    RegionalStrength：地域强度因子。
    计算股票所属地域的整体表现强度。
    公式：地域强度 = mean(地域内股票20日收益率)
    方向（direction=1）：地域强度越高，未来收益可能越高。
    """
    name = "RegionalStrength"
    direction = 1  # 地域强度高，未来收益可能高
    description = (
        "RegionalStrength：地域强度因子。\n"
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
        
        # 计算股票收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns'] = daily_df.groupby('stock_code')['close'].pct_change()
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['returns'].rolling(20).sum().reset_index(0, drop=True)
        
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

class IndustryCorrelation(FactorBase):
    """
    IndustryCorrelation：行业相关性因子。
    计算股票与其所属行业平均收益的相关性。
    公式：行业相关性 = corr(股票收益率, 行业平均收益率, 20)
    方向（direction=1）：相关性越高，跟随行业趋势越强。
    """
    name = "IndustryCorrelation"
    direction = 1  # 相关性高，跟随行业趋势强
    description = (
        "IndustryCorrelation：行业相关性因子。\n"
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
        
        # 计算股票收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns'] = daily_df.groupby('stock_code')['close'].pct_change()
        
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

class SectorRotation(FactorBase):
    """
    SectorRotation：板块轮动因子。
    计算板块相对强度变化，识别板块轮动机会。
    公式：板块轮动 = ts_rank(板块动量, 所有板块) - ts_rank(板块动量, 所有板块, 20)
    方向（direction=1）：轮动强度越高，板块可能处于上升期。
    """
    name = "SectorRotation"
    direction = 1  # 轮动强度高，板块可能处于上升期
    description = (
        "SectorRotation：板块轮动因子。\n"
        "计算板块相对强度变化，识别板块轮动机会。\n"
        "公式：板块轮动 = ts_rank(板块动量, 所有板块) - ts_rank(板块动量, 所有板块, 20)\n"
        "方向（direction=1）：轮动强度越高，板块可能处于上升期。"
    )
    data_requirements = {
        'daily': {'window': 40},
        'stock_basic': {'window': 1}  # 使用market作为板块分类
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        stock_basic_df = data['stock_basic'].copy()
        
        # 计算股票收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns'] = daily_df.groupby('stock_code')['close'].pct_change()
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['returns'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并板块数据（使用market作为板块分类）
        df = daily_df.merge(stock_basic_df[['stock_code', 'market']], 
                           on=['stock_code'], how='left')
        
        # 计算板块动量
        sector_momentum = df.groupby(['market', 'trade_date'])['returns_20d'].mean().reset_index()
        sector_momentum = sector_momentum.rename(columns={'returns_20d': 'sector_momentum'})
        
        # 计算板块轮动
        sector_momentum = sector_momentum.sort_values(['market', 'trade_date'])
        
        # 计算当前排名和20天前排名
        sector_momentum['current_rank'] = sector_momentum.groupby('trade_date')['sector_momentum'].rank(method='average')
        sector_momentum['past_rank'] = sector_momentum.groupby('market')['current_rank'].shift(20)
        
        # 计算轮动强度
        sector_momentum['rotation_strength'] = sector_momentum['current_rank'] - sector_momentum['past_rank']
        
        # 合并回原数据
        result = df.merge(sector_momentum[['market', 'trade_date', 'rotation_strength']], 
                         on=['market', 'trade_date'], how='left')
        result['factor'] = self.name
        result['value'] = result['rotation_strength']
        
        # 只保留有效数据
        result = result.dropna(subset=['value', 'market'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class SWIndustryRank(FactorBase):
    """
    SWIndustryRank：申万行业排名因子。
    计算股票在其所属申万行业中的相对表现排名。
    公式：申万行业排名 = ts_rank(股票收益率, 申万行业内所有股票)
    方向（direction=1）：申万行业排名越高，未来收益可能越高。
    """
    name = "SWIndustryRank"
    direction = 1  # 申万行业排名高，未来收益可能高
    description = (
        "SWIndustryRank：申万行业排名因子。\n"
        "计算股票在其所属申万行业中的相对表现排名。\n"
        "公式：申万行业排名 = ts_rank(股票收益率, 申万行业内所有股票)\n"
        "方向（direction=1）：申万行业排名越高，未来收益可能越高。"
    )
    data_requirements = {
        'daily': {'window': 20},  # 需要20天数据计算收益率
        'industry_member': {'window': 1}  # 申万行业分类数据
    }

    def _compute_impl(self, data):
        daily_df = data['daily'].copy()
        industry_member_df = data['industry_member'].copy()
        
        # 计算股票收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns'] = daily_df.groupby('stock_code')['close'].pct_change()
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['returns'].rolling(20).sum().reset_index(0, drop=True)
        
        # 合并申万行业数据
        df = daily_df.merge(industry_member_df[['stock_code', 'l1_name']], 
                           on=['stock_code'], how='left')
        
        # 按日期和申万行业分组计算排名
        df['value'] = df.groupby(['trade_date', 'l1_name'])['returns_20d'].rank(method='average')
        df['factor'] = self.name
        
        # 只保留有效数据
        result = df.dropna(subset=['value', 'l1_name'])
        
        return result[['stock_code', 'trade_date', 'factor', 'value']].rename(
            columns={'stock_code': 'code', 'trade_date': 'date'}
        ).reset_index(drop=True)

class CityStrength(FactorBase):
    """
    CityStrength：城市强度因子。
    计算股票所属城市的整体表现强度。
    公式：城市强度 = mean(城市内股票20日收益率)
    方向（direction=1）：城市强度越高，未来收益可能越高。
    """
    name = "CityStrength"
    direction = 1  # 城市强度高，未来收益可能高
    description = (
        "CityStrength：城市强度因子。\n"
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
        
        # 计算股票收益率
        daily_df = daily_df.sort_values(['stock_code', 'trade_date'])
        daily_df['returns'] = daily_df.groupby('stock_code')['close'].pct_change()
        daily_df['returns_20d'] = daily_df.groupby('stock_code')['returns'].rolling(20).sum().reset_index(0, drop=True)
        
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