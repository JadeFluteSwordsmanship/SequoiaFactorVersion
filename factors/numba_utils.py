import numpy as np
from numba import njit

@njit(cache=True)
def ts_rank_numba(arr, window):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        s = i - window + 1
        last = arr[i]
        r = 1
        for j in range(s, i):
            if not np.isnan(arr[j]) and arr[j] <= last:
                r += 1
        out[i] = r
    return out

@njit(cache=True)
def rolling_corr_numba(x, y, win):
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(win - 1, n):
        s = i - win + 1
        xw = x[s:i+1]
        yw = y[s:i+1]
        mask = ~(np.isnan(xw) | np.isnan(yw))
        xw = xw[mask]
        yw = yw[mask]
        cnt = len(xw)
        if cnt < win or cnt < 2:
            continue
        mx = np.mean(xw)
        my = np.mean(yw)
        cov = np.sum((xw - mx) * (yw - my))
        vx = np.sum((xw - mx) ** 2)
        vy = np.sum((yw - my) ** 2)
        den = np.sqrt(vx * vy)
        if den > 1e-14:
            out[i] = cov / den
        else:
            out[i] = np.nan
    return out

@njit(cache=True)
def rolling_max_numba(arr, win, min_periods=None):
    n = len(arr)
    out = np.full(n, np.nan)
    if min_periods is None:
        min_periods = win
    for i in range(n):
        s = max(0, i - win + 1)
        count = 0
        mv = -1e100
        for j in range(s, i + 1):
            v = arr[j]
            if not np.isnan(v):
                mv = max(mv, v)
                count += 1
        if count >= min_periods:
            out[i] = mv
        else:
            out[i] = np.nan
    return out

@njit(cache=True)
def decay_linear_numba(arr, period):
    """
    线性衰减加权移动平均
    权重为 [1, 2, 3, ..., period]，越新的数据权重越大
    """
    n = len(arr)
    out = np.full(n, np.nan)
    
    # 预计算权重和权重总和
    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = np.sum(weights)
    
    for i in range(period - 1, n):
        s = i - period + 1
        weighted_sum = 0.0
        valid_count = 0
        
        for j in range(period):
            val = arr[s + j]
            if not np.isnan(val):
                weighted_sum += val * weights[j]
                valid_count += 1
        
        if valid_count == period:  # 所有值都有效
            out[i] = weighted_sum / weight_sum
        else:
            out[i] = np.nan
    
    return out

@njit(cache=True)
def consecutive_moneyflow_strength_numba(ratios, window=5, min_consecutive=3):
    """
    计算连续资金流入强度
    ratios: 大单净流入占比数组
    window: 回看窗口大小
    min_consecutive: 最少连续流入天数
    """
    n = len(ratios)
    out = np.full(n, np.nan)
    
    # 动态生成权重衰减：从1.0开始，线性衰减到0.2
    weights = np.linspace(1.0, 0.2, window, dtype=np.float64)
    
    for i in range(window - 1, n):
        # 获取最近window天的数据
        start_idx = i - window + 1
        window_ratios = ratios[start_idx:i+1]
        
        # 计算连续流入强度
        consecutive_strength = 0.0
        consecutive_days = 0
        
        for j in range(window):
            ratio = window_ratios[j]
            weight = weights[j]
            
            if not np.isnan(ratio) and ratio > 0:  # 只考虑净流入
                consecutive_strength += ratio * weight
                consecutive_days += 1
            else:
                # 遇到流出或NaN就重置
                consecutive_strength = 0.0
                consecutive_days = 0
                break
        
        # 如果连续流入天数少于要求，则设为0
        if consecutive_days >= min_consecutive:
            out[i] = consecutive_strength
        else:
            out[i] = 0.0
    
    return out
