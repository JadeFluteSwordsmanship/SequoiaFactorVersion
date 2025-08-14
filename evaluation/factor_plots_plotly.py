"""
factor_plots_plotly.py
Plotly figures for factor evaluation — consumes DataFrames from FactorEvaluator.
Auto-saves HTML (+PNG if kaleido available) under: <save_dir>/<factor_name>/...

Usage:
    fig_group_navs(daily_df, factor_name="MyFactor")  # 自动保存
"""

from __future__ import annotations
from typing import Optional, Dict
import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from settings import config  # 全局配置（settings.init() 已自动执行）


# ============= helpers =============

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
    # 去掉不适合做文件/目录的字符
    return re.sub(r'[\\/:*?"<>|]+', "_", str(s)).strip() or "Factor"

def _factor_dir(factor_name: Optional[str]) -> str:
    base = _get_base_save_dir()
    fname = _safe_name(factor_name)
    out = os.path.join(base, fname)
    _ensure_dir(out)
    return out

def _save_figure(fig: go.Figure,
                 factor_name: Optional[str],
                 base_filename: str,
                 save_html: bool = True,
                 save_png: bool = True,
                 include_plotlyjs_mode: str = "inline"  # 'inline' | 'cdn' | 'directory'
                 ) -> Dict[str, str]:
    """
    Save figure to <save_dir>/<factor_name>/<base_filename>.*
    include_plotlyjs_mode:
      - 'inline'   : HTML 内嵌 plotly.js（离线可打开，文件较大）【默认、推荐】
      - 'cdn'      : 走 CDN（需要外网）
      - 'directory': 将 plotly.js 保存在同目录（离线可用，多个图共用）
    """
    saved = {}
    out_dir = _factor_dir(factor_name)

    if save_html:
        html_path = os.path.join(out_dir, base_filename + ".html")
        # directory 模式需要一个相对目录，Plotly 会写入 plotly-*.js
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
            # 没装 kaleido 或环境问题时跳过 PNG，不报错中断
            print(f"[plotly] PNG save skipped ({e})")

    return saved

def _hist_cum_df(series: pd.Series, bins: int = 20) -> pd.DataFrame:
    s = series.dropna().astype(float)
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


# ============= IR / IC panels =============

def fig_ir_distribution_panel(
    ir_df: pd.DataFrame,
    value_col: str = "IR",
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
) -> go.Figure:
    """
    ir_df:
      - 必含 value_col（默认 'IR'）
      - 可含 'IR_cum'（副轴累计线）
      - index 为 DatetimeIndex
    """
    if ir_df.empty or value_col not in ir_df.columns:
        raise ValueError("ir_df 为空或缺少必需列")

    histdf = _hist_cum_df(ir_df[value_col], bins=bins)
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

    # 下面板：逐日柱子+累计线
    x = ir_df.index
    fig.add_trace(go.Bar(x=x, y=ir_df[value_col], name=value_col, opacity=0.8),
                  row=2, col=1, secondary_y=False)
    if "IR_cum" in ir_df.columns:
        fig.add_trace(go.Scatter(x=x, y=ir_df["IR_cum"], name="IR累计值", mode="lines+markers"),
                      row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="IR累计值", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text=value_col, row=2, col=1, secondary_y=False)

    _top = title_top or f"{value_col} 分布图" + (f" — {_safe_name(factor_name)}" if factor_name else "")
    _btm = title_bottom or f"每期 {value_col} 图" + (f" — {_safe_name(factor_name)}" if factor_name else "")
    fig.update_layout(title_text=f"{_top}<br><sub>{_btm}</sub>",
                      barmode="overlay", legend_orientation="h",
                      legend_y=1.1, margin=dict(l=40, r=40, t=80, b=40),
                      xaxis_title="")

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
) -> go.Figure:
    """通用：左轴柱子(bar_col) + 右轴累计线(cum_col)"""
    if df.empty or bar_col not in df or cum_col not in df:
        raise ValueError("df 为空或缺列")

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    x = df.index
    fig.add_trace(go.Bar(x=x, y=df[bar_col], name=bar_col, opacity=0.8),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df[cum_col], name=cum_col, mode="lines+markers"),
                  row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text=bar_col, secondary_y=False)
    fig.update_yaxes(title_text=cum_col, secondary_y=True)

    _title = title or f"{bar_col} 与 {cum_col}" + (f" — {_safe_name(factor_name)}" if factor_name else "")
    fig.update_layout(title_text=_title, margin=dict(l=40, r=40, t=60, b=40), xaxis_title="日期")

    if save:
        base = filename or f"{_safe_name(factor_name)}_{bar_col}_panel"
        _save_figure(fig, factor_name, base,
                     save_html=save_html, save_png=save_png,
                     include_plotlyjs_mode=include_plotlyjs_mode)
    return fig


# ============= 分组净值 / 多空 =============

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
) -> go.Figure:
    """
    输入：各组“日收益”DataFrame（列名如 'Q1(最高)' … 'Qn(最低)'）
    本函数内部转净值并保存到 <save_dir>/<factor_name>/...
    """
    if group_daily_returns.empty:
        raise ValueError("group_daily_returns 为空")

    daily = group_daily_returns.sort_index().fillna(0.0)
    nav = (1 + daily).cumprod()

    fig = go.Figure()
    for col in nav.columns:
        fig.add_trace(go.Scatter(x=nav.index, y=nav[col], mode="lines+markers", name=str(col)))
    _title = title or ("分组净值（重叠组合日频）" + (f" — {_safe_name(factor_name)}" if factor_name else ""))
    fig.update_layout(title_text=_title, legend_orientation="h", legend_y=1.05,
                      margin=dict(l=40, r=40, t=60, b=40),
                      yaxis_title="净值", xaxis_title="日期")

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
) -> go.Figure:
    """
    柱子：多空“日收益”；折线：累计指标（如 IR_cum）。
    """
    if ls_df.empty or ls_col not in ls_df:
        raise ValueError("ls_df 为空或缺少需要的列")

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=ls_df.index, y=ls_df[ls_col], name=ls_col, opacity=0.8),
                  row=1, col=1, secondary_y=False)
    if cum_line in ls_df.columns:
        fig.add_trace(go.Scatter(x=ls_df.index, y=ls_df[cum_line], name=cum_line, mode="lines+markers"),
                      row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text=cum_line, secondary_y=True)
    fig.update_yaxes(title_text=ls_col, secondary_y=False)

    _title = title or ("多空收益与累计指标" + (f" — {_safe_name(factor_name)}" if factor_name else ""))
    fig.update_layout(title_text=_title, margin=dict(l=40, r=40, t=60, b=40), xaxis_title="日期")

    if save:
        base = filename or f"{_safe_name(factor_name)}_LS_bars"
        _save_figure(fig, factor_name, base,
                     save_html=save_html, save_png=save_png,
                     include_plotlyjs_mode=include_plotlyjs_mode)
    return fig
