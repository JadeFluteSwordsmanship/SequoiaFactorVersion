
from __future__ import annotations
from typing import Optional, Sequence, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------- helpers -----------------

def _hist_cum_df(series: pd.Series, bins: int = 20) -> pd.DataFrame:
    s = series.dropna().astype(float)
    if s.empty:
        return pd.DataFrame(columns=['mid','prob','cumprob','edges'])
    counts, edges = np.histogram(s.values, bins=bins, density=True)
    mids = 0.5*(edges[1:] + edges[:-1])
    total = counts.sum()
    prob = counts / total if total != 0 else counts
    cumprob = prob.cumsum()
    out = pd.DataFrame({'mid':mids, 'prob':prob, 'cumprob':cumprob})
    out['edges'] = [f"{edges[i]:.3f}" for i in range(len(mids))]
    return out

# ----------------- IR / IC panels -----------------

def fig_ir_distribution_panel(ir_df: pd.DataFrame, value_col: str = 'IR',
                              title_top: str = "IR 分布图",
                              title_bottom: str = "每期 IR 图") -> go.Figure:
    """
    Expects ir_df columns:
      - value_col (default 'IR')
      - optional 'IR_cum' for the cumulative line in the bottom panel
      - index is DatetimeIndex (for bottom panel x-axis)
    """
    histdf = _hist_cum_df(ir_df[value_col])
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15,
        row_heights=[0.5, 0.5],
        specs=[[{'secondary_y': True}], [{'secondary_y': True}]]
    )

    # Top: histogram + cumulative
    fig.add_trace(go.Bar(x=histdf['mid'], y=histdf['prob'],
                         name='概率', opacity=0.8),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=histdf['mid'], y=histdf['cumprob'],
                             name='累计概率', mode='lines+markers'),
                  row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="概率", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="累计概率", row=1, col=1, secondary_y=True, rangemode="tozero")

    # Bottom: daily bars + cumulative
    x = ir_df.index
    fig.add_trace(go.Bar(x=x, y=ir_df[value_col], name=value_col, opacity=0.8),
                  row=2, col=1, secondary_y=False)
    if 'IR_cum' in ir_df.columns:
        fig.add_trace(go.Scatter(x=x, y=ir_df['IR_cum'], name='IR累计值',
                                 mode='lines+markers'),
                      row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="IR累计值", row=2, col=1, secondary_y=True)

    fig.update_yaxes(title_text=value_col, row=2, col=1, secondary_y=False)
    fig.update_layout(title_text=f"{title_top}<br><sub>{title_bottom}</sub>",
                      barmode='overlay', legend_orientation='h',
                      legend_y=1.1, margin=dict(l=40, r=40, t=80, b=40))
    return fig

def fig_ic_ir_panel(df: pd.DataFrame, bar_col: str, cum_col: str,
                    title: str) -> go.Figure:
    """Generic 'bar + cumulative line (secondary axis)' panel."""
    fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])
    x = df.index
    fig.add_trace(go.Bar(x=x, y=df[bar_col], name=bar_col, opacity=0.8),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=df[cum_col], name=cum_col,
                             mode='lines+markers'),
                  row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text=bar_col, secondary_y=False)
    fig.update_yaxes(title_text=cum_col, secondary_y=True)
    fig.update_layout(title_text=title, margin=dict(l=40, r=40, t=60, b=40))
    return fig

# ----------------- group returns -----------------

def fig_group_navs(group_daily_returns: pd.DataFrame,
                   title: str = "分组净值（重叠组合日频）") -> go.Figure:
    """Input daily returns per group, cumprod here to NAV (start=1)."""
    daily = group_daily_returns.sort_index().fillna(0.0)
    nav = (1 + daily).cumprod()
    fig = go.Figure()
    for col in nav.columns:
        fig.add_trace(go.Scatter(x=nav.index, y=nav[col],
                                 mode='lines+markers', name=str(col)))
    fig.update_layout(title_text=title, legend_orientation='h', legend_y=1.05,
                      margin=dict(l=40, r=40, t=60, b=40),
                      yaxis_title="净值")
    return fig

def fig_longshort_bars(ls_df: pd.DataFrame, ls_col: str = 'LS',
                       cum_line: str = 'IR_cum',
                       title: str = "多空收益与累计指标") -> go.Figure:
    fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Bar(x=ls_df.index, y=ls_df[ls_col], name=ls_col, opacity=0.8),
                  row=1, col=1, secondary_y=False)
    if cum_line in ls_df.columns:
        fig.add_trace(go.Scatter(x=ls_df.index, y=ls_df[cum_line],
                                 name=cum_line, mode='lines+markers'),
                      row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text=cum_line, secondary_y=True)
    fig.update_yaxes(title_text=ls_col, secondary_y=False)
    fig.update_layout(title_text=title, margin=dict(l=40, r=40, t=60, b=40))
    return fig
