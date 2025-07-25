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
