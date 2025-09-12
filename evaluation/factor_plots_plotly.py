"""
factor_plots_plotly.py
Plotly figures for factor evaluation — consumes DataFrames from FactorEvaluator.
Auto-saves HTML (+PNG if kaleido available) under: <save_dir>/<factor_name>/...

All x-axes default to trading-day category (no weekend gaps) with "YYYY-MM-DD" labels.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Union
import os, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from settings import config  # 全局配置（settings.init() 已自动执行）

# ================= helpers =================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _get_base_save_dir() -> str:
    try:
        return config.get("save_dir", "E:/data/Backtest")
    except Exception:
        return "E:/data/Backtest"

def _safe_name(s: Optional[str]) -> str:
    if not s:
        return "Factor"
    return re.sub(r'[\\/:*?"<>|]+', "_", str(s)).strip() or "Factor"

def _factor_dir(factor_name: Optional[str]) -> str:
    base = _get_base_save_dir()
    fname = _safe_name(factor_name)
    out = os.path.join(base, fname)
    _ensure_dir(out)
    return out

def _to_x_labels(idx: pd.Index, date_format: str = "%Y-%m-%d") -> List[str]:
    """把索引转成用于分类轴的字符串标签（交易日连续、无周末空隙）。"""
    if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        return [pd.Timestamp(x).strftime(date_format) for x in idx]
    return [str(x) for x in idx]

def _apply_smart_xticks(fig: go.Figure, axis_id: str, labels: List[str],
                        max_xticks: int = 12, tickangle: int = 0) -> None:
    """
    根据标签数量自动抽样横轴刻度（分类轴）。
    axis_id 例如 'xaxis', 'xaxis2' ...
    """
    n = len(labels)
    if n == 0:
        return
    step = max(1, int(np.ceil(n / float(max_xticks))))
    tickvals = list(range(0, n, step))  # category 轴的 tickvals 用的是“位置索引”
    ticktext = [labels[i] for i in tickvals]
    fig.layout[axis_id].update(
        type="category",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=tickangle
    )

def _save_figure(fig: go.Figure,
                 factor_name: Optional[str],
                 base_filename: str,
                 save_html: bool = True,
                 save_png: bool = True,
                 include_plotlyjs_mode: str = "inline"  # 'inline' | 'cdn' | 'directory'
                 ) -> Dict[str, str]:
    saved = {}
    out_dir = _factor_dir(factor_name)

    if save_html:
        html_path = os.path.join(out_dir, base_filename + ".html")
        if include_plotlyjs_mode == "directory":
            fig.write_html(html_path, include_plotlyjs="directory", full_html=True)
        elif include_plotlyjs_mode == "cdn":
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        else:
            fig.write_html(html_path, include_plotlyjs="inline", full_html=True)
        saved["html"] = html_path
        print(f"[plotly] HTML saved to: {html_path}")

    if save_png:
        try:
            png_path = os.path.join(out_dir, base_filename + ".png")
            fig.write_image(png_path, scale=2)  # 需要 kaleido
            saved["png"] = png_path
            print(f"[plotly] PNG  saved to: {png_path}")
        except Exception as e:
            print(f"[plotly] PNG save skipped ({e})")

    return saved

def _hist_cum_df(series: pd.Series, bins: int = 20) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        return pd.DataFrame(columns=["mid", "prob", "cumprob", "edges"])
    counts, edges = np.histogram(s.values, bins=bins, density=True)
    mids = 0.5 * (edges[1:] + edges[:-1])
    total = counts.sum()
    prob = counts / total if total != 0 else counts
    cumprob = prob.cumsum()
    out = pd.DataFrame({"mid": mids, "prob": prob, "cumprob": cumprob})
    out["edges"] = [f"{edges[i]:.3f}" for i in range(len(mids))]
    return out

def _apply_common_layout(fig: go.Figure):
    fig.update_layout(font=dict(family="Microsoft YaHei, SimHei, Arial"),
                      template="plotly_white")

# ================= IR / IC panels =================

def fig_ir_distribution_panel(
    ir_df: pd.DataFrame,
    value_col: Optional[str] = None,         # 默认自动识别
    *,
    factor_name: Optional[str] = None,
    title_top: Optional[str] = None,
    title_bottom: Optional[str] = None,
    save: bool = True,
    filename: Optional[str] = None,
    bins: int = 20,
    save_html: bool = True,
    save_png: bool = True,
    include_plotlyjs_mode: str = "inline",
    as_category_axis: bool = True,
    date_format: str = "%Y-%m-%d",
    max_xticks: int = 12,
    tickangle: int = 0,
) -> go.Figure:
    """
    ir_df:
      - 默认优先找 'IR_from_IC'，否则回退到 'IR'/'IR_LS'/'tstat_topN'
      - 如无 'IR_cum'，会自动基于 value_col 生成：fillna(0).cumsum()
      - index 建议为 DatetimeIndex（也支持任意索引）
    """
    if ir_df is None or ir_df.empty:
        raise ValueError("ir_df 不能为空")

    candidates = ["IR_from_IC", "IR", "IR_LS", "tstat_topN"]
    if value_col is None:
        for c in candidates:
            if c in ir_df.columns:
                value_col = c
                break
    if value_col is None or value_col not in ir_df.columns:
        raise ValueError(f"value_col 未指定或不存在，ir_df 可用列里找不到：{candidates}")

    df = ir_df.copy()
    if "IR_cum" not in df.columns:
        df["IR_cum"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).cumsum()

    histdf = _hist_cum_df(df[value_col], bins=bins)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15,
        row_heights=[0.5, 0.5], specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )

    # 上面板：直方图+累计概率
    fig.add_trace(go.Bar(x=histdf["mid"], y=histdf["prob"], name="概率", opacity=0.8),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=histdf["mid"], y=histdf["cumprob"],
                             name="累计概率", mode="lines+markers"),
                  row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="概率", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="累计概率", row=1, col=1, secondary_y=True, rangemode="tozero")

    # 下面板：逐日柱子+累计线（分类轴、中文日期）
    x_labels = _to_x_labels(df.index, date_format=date_format)
    fig.add_trace(go.Bar(x=x_labels, y=df[value_col], name=value_col, opacity=0.8),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_labels, y=df["IR_cum"], name="IR累计值", mode="lines+markers"),
                  row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="IR累计值", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text=value_col, row=2, col=1, secondary_y=False)
    if as_category_axis:
        # 主图在第二行 -> xaxis2
        _apply_smart_xticks(fig, "xaxis2", x_labels, max_xticks=max_xticks, tickangle=tickangle)

    _top = title_top or f"{value_col} 分布图" + (f" — {_safe_name(factor_name)}" if factor_name else "")
    _btm = title_bottom or f"每期 {value_col} 图" + (f" — {_safe_name(factor_name)}" if factor_name else "")
    fig.update_layout(title_text=f"{_top}<br><sub>{_btm}</sub>",
                      barmode="overlay", legend_orientation="h",
                      legend_y=1.1, margin=dict(l=40, r=40, t=80, b=40))
    _apply_common_layout(fig)

    if save:
        base = filename or f"{_safe_name(factor_name)}_{value_col}_panel"
        _save_figure(fig, factor_name, base,
                     save_html=save_html, save_png=save_png,
                     include_plotlyjs_mode=include_plotlyjs_mode)
    return fig


def fig_ic_ir_panel(
    df: pd.DataFrame,
    bar_col: str,
    cum_col: str,
    *,
    factor_name: Optional[str] = None,
    title: Optional[str] = None,
    save: bool = True,
    filename: Optional[str] = None,
    save_html: bool = True,
    save_png: bool = True,
    include_plotlyjs_mode: str = "inline",
    as_category_axis: bool = True,
    date_format: str = "%Y-%m-%d",
    max_xticks: int = 12,
    tickangle: int = 0,
) -> go.Figure:
    """通用：左轴柱子(bar_col) + 右轴累计线(cum_col)"""
    if df.empty or bar_col not in df or cum_col not in df:
        raise ValueError("df 为空或缺列")

    x_labels = _to_x_labels(df.index, date_format=date_format)
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x_labels, y=df[bar_col], name=bar_col, opacity=0.8),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x_labels, y=df[cum_col], name=cum_col, mode="lines+markers"),
                  row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text=bar_col, secondary_y=False)
    fig.update_yaxes(title_text=cum_col, secondary_y=True)
    if as_category_axis:
        _apply_smart_xticks(fig, "xaxis", x_labels, max_xticks=max_xticks, tickangle=tickangle)

    _title = title or f"{bar_col} 与 {cum_col}" + (f" — {_safe_name(factor_name)}" if factor_name else "")
    fig.update_layout(title_text=_title, margin=dict(l=40, r=40, t=60, b=40), xaxis_title="日期")
    _apply_common_layout(fig)

    if save:
        base = filename or f"{_safe_name(factor_name)}_{bar_col}_panel"
        _save_figure(fig, factor_name, base,
                     save_html=save_html, save_png=save_png,
                     include_plotlyjs_mode=include_plotlyjs_mode)
    return fig

# ================= 分组净值 / 多空 =================

def fig_group_navs(
    group_daily_returns: pd.DataFrame,
    *,
    factor_name: Optional[str] = None,
    title: Optional[str] = None,
    save: bool = True,
    filename: Optional[str] = None,
    save_html: bool = True,
    save_png: bool = True,
    include_plotlyjs_mode: str = "inline",
    as_category_axis: bool = True,
    date_format: str = "%Y-%m-%d",
    max_xticks: int = 12,
    tickangle: int = 0,
) -> go.Figure:
    """
    输入：各组“日收益”DataFrame（列名如 'Q1(最高)' … 'Qn(最低)'）
    本函数内部转净值并保存到 <save_dir>/<factor_name>/...
    """
    if group_daily_returns.empty:
        raise ValueError("group_daily_returns 为空")

    daily = group_daily_returns.sort_index().fillna(0.0)
    nav = (1 + daily).cumprod()

    x_labels = _to_x_labels(nav.index, date_format=date_format)
    fig = go.Figure()
    for col in nav.columns:
        fig.add_trace(go.Scatter(x=x_labels, y=nav[col], mode="lines+markers", name=str(col)))
    _title = title or ("分组净值（重叠组合日频）" + (f" — {_safe_name(factor_name)}" if factor_name else ""))
    fig.update_layout(title_text=_title, legend_orientation="h", legend_y=1.05,
                      margin=dict(l=40, r=40, t=60, b=40),
                      yaxis_title="净值", xaxis_title="日期")
    if as_category_axis:
        _apply_smart_xticks(fig, "xaxis", x_labels, max_xticks=max_xticks, tickangle=tickangle)
    _apply_common_layout(fig)

    if save:
        base = filename or f"{_safe_name(factor_name)}_GroupNAV"
        _save_figure(fig, factor_name, base,
                     save_html=save_html, save_png=save_png,
                     include_plotlyjs_mode=include_plotlyjs_mode)
    return fig


def fig_longshort_bars(
    ls_df: pd.DataFrame,
    ls_col: str = "LS_daily",
    cum_line: str = "IR_cum",
    *,
    factor_name: Optional[str] = None,
    title: Optional[str] = None,
    save: bool = True,
    filename: Optional[str] = None,
    save_html: bool = True,
    save_png: bool = True,
    include_plotlyjs_mode: str = "inline",
    as_category_axis: bool = True,
    date_format: str = "%Y-%m-%d",
    max_xticks: int = 12,
    tickangle: int = 0,
) -> go.Figure:
    """
    柱子：多空“日收益”；折线：累计指标（如 IR_cum）。
    """
    if ls_df.empty or ls_col not in ls_df:
        raise ValueError("ls_df 为空或缺少需要的列")

    x_labels = _to_x_labels(ls_df.index, date_format=date_format)
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x_labels, y=ls_df[ls_col], name=ls_col, opacity=0.8),
                  row=1, col=1, secondary_y=False)
    if cum_line in ls_df.columns:
        fig.add_trace(go.Scatter(x=x_labels, y=ls_df[cum_line], name=cum_line, mode="lines+markers"),
                      row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text=cum_line, secondary_y=True)
    fig.update_yaxes(title_text=ls_col, secondary_y=False)
    if as_category_axis:
        _apply_smart_xticks(fig, "xaxis", x_labels, max_xticks=max_xticks, tickangle=tickangle)

    _title = title or ("多空收益与累计指标" + (f" — {_safe_name(factor_name)}" if factor_name else ""))
    fig.update_layout(title_text=_title, margin=dict(l=40, r=40, t=60, b=40), xaxis_title="日期")
    _apply_common_layout(fig)

    if save:
        base = filename or f"{_safe_name(factor_name)}_LS_bars"
        _save_figure(fig, factor_name, base,
                     save_html=save_html, save_png=save_png,
                     include_plotlyjs_mode=include_plotlyjs_mode)
    return fig

# ================= Strategy vs Benchmark =================

def fig_topn_vs_index_nav(
    topn_daily_returns: pd.Series,
    index_daily_df: pd.DataFrame,
    index_ts_codes: Union[str, List[str]],
    *,
    factor_name: Optional[str] = None,
    title: Optional[str] = None,
    save: bool = True,
    filename: Optional[str] = None,
    save_html: bool = True,
    save_png: bool = True,
    include_plotlyjs_mode: str = "inline",
    as_category_axis: bool = True,
    date_format: str = "%Y-%m-%d",
    max_xticks: int = 12,
    tickangle: int = 0,
) -> go.Figure:
    """
    绘制 TopN 策略净值 vs 多个指数基准净值对比。
    - topn_daily_returns: 策略"日收益"（已按 k_hold 折算/叠加前的日收益）索引为日期
    - index_daily_df: 指数日线数据（含多个指数），要求列包含 ['ts_code','trade_date','close'] 或包含 'pct_chg'
    - index_ts_codes: 基准指数 ts_code，可以是单个字符串如 '000300.SH' 或列表如 ['000300.SH', '000016.SH', '399006.SZ']
    """
    if topn_daily_returns is None or len(topn_daily_returns) == 0:
        raise ValueError("topn_daily_returns 为空")
    if index_daily_df is None or index_daily_df.empty:
        raise ValueError("index_daily_df 为空")

    # 统一处理 index_ts_codes 为列表
    if isinstance(index_ts_codes, str):
        index_ts_codes = [index_ts_codes]
    
    if not index_ts_codes:
        raise ValueError("index_ts_codes 不能为空")

    # 对齐日期索引
    strat_daily = pd.to_numeric(topn_daily_returns, errors='coerce')
    strat_daily.index = pd.to_datetime(strat_daily.index)

    # 计算策略净值
    nav_strat = (1.0 + strat_daily.fillna(0.0)).cumprod().rename("Strategy_NAV")

    # 处理每个指数
    nav_data = {"Strategy_NAV": nav_strat}
    available_indices = []
    
    for ts_code in index_ts_codes:
        # 处理指数数据：过滤、按日期排序
        idx = index_daily_df[index_daily_df['ts_code'] == ts_code].copy()
        if idx.empty:
            print(f"警告：未找到指数 {ts_code}，跳过")
            continue
            
        if 'trade_date' not in idx.columns:
            print(f"警告：指数 {ts_code} 缺少 'trade_date' 列，跳过")
            continue
            
        idx['trade_date'] = pd.to_datetime(idx['trade_date'])
        idx = idx.sort_values('trade_date')

        # 计算指数日收益：优先 close.pct_change，其次 pct_chg/100
        if 'close' in idx.columns:
            idx_ret = idx['close'].pct_change()
        elif 'pct_chg' in idx.columns:
            idx_ret = pd.to_numeric(idx['pct_chg'], errors='coerce') / 100.0
        else:
            print(f"警告：指数 {ts_code} 需要包含 'close' 或 'pct_chg' 列，跳过")
            continue

        # 对齐到策略日期
        idx_ret = idx_ret.set_axis(idx['trade_date'])
        idx_ret = idx_ret.reindex(strat_daily.index).fillna(0.0)
        
        # 计算指数净值
        nav_index = (1.0 + idx_ret).cumprod().rename(f"{ts_code}_NAV")
        nav_data[f"{ts_code}_NAV"] = nav_index
        available_indices.append(ts_code)

    if not available_indices:
        raise ValueError("没有找到任何可用的指数数据")

    # 组装绘图
    x_labels = _to_x_labels(nav_strat.index, date_format=date_format)
    fig = go.Figure()
    
    # 添加策略净值线
    fig.add_trace(go.Scatter(x=x_labels, y=nav_strat.values, mode="lines+markers", 
                            name="TopN策略净值", line=dict(width=3, color='red')))
    
    # 添加各个指数净值线
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i, ts_code in enumerate(available_indices):
        nav_key = f"{ts_code}_NAV"
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(x=x_labels, y=nav_data[nav_key].values, 
                                mode="lines+markers", name=f"指数净值({ts_code})",
                                line=dict(color=color)))

    # 生成标题
    indices_str = "、".join(available_indices)
    _title = title or (f"TopN策略 vs 指数基准对比({indices_str})" + (f" — {_safe_name(factor_name)}" if factor_name else ""))
    
    fig.update_layout(title_text=_title,
                      legend_orientation="h", legend_y=1.05,
                      margin=dict(l=40, r=40, t=60, b=40),
                      yaxis_title="净值", xaxis_title="日期")
    if as_category_axis:
        _apply_smart_xticks(fig, "xaxis", x_labels, max_xticks=max_xticks, tickangle=tickangle)
    _apply_common_layout(fig)

    if save:
        indices_safe = "_".join([_safe_name(code) for code in available_indices])
        base = filename or f"{_safe_name(factor_name)}_TopN_vs_{indices_safe}"
        _save_figure(fig, factor_name, base,
                     save_html=save_html, save_png=save_png,
                     include_plotlyjs_mode=include_plotlyjs_mode)
    return fig
