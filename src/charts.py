"""Plotly chart builders used by the Streamlit app."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PRIMARY = "#4F46E5"
SECONDARY = "#0EA5E9"
MUTED = "#94A3B8"
FONT = dict(family="Inter, Segoe UI, sans-serif", size=13)


def _layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=16)), height=height, font=FONT,
                      margin=dict(l=10, r=10, t=50, b=10), template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def metric_comparison(rows: list[dict], metric: str, title: str) -> go.Figure:
    df = pd.DataFrame(rows)
    colors = [MUTED if "baseline" in m.lower() else PRIMARY for m in df["model"]]
    fig = go.Figure(go.Bar(x=df[metric], y=df["model"], orientation="h", marker_color=colors,
                           text=df[metric].round(2), textposition="auto"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height=90 + 45 * len(df))


def actual_vs_predicted(actual, predicted) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers", name="Test students",
                             marker=dict(color=PRIMARY, opacity=0.55, size=7)))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect prediction",
                             line=dict(color=MUTED, dash="dash")))
    fig.update_xaxes(title="Actual final exam score", range=[0, 100])
    fig.update_yaxes(title="Predicted score", range=[0, 100])
    fig = _layout(fig, "Actual vs predicted (test set)")
    fig.update_layout(legend=dict(orientation="v", x=0.02, y=0.98, xanchor="left", yanchor="top",
                                  bgcolor="rgba(255,255,255,0.8)"))
    return fig


def residuals(actual, predicted) -> go.Figure:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predicted, y=actual - predicted, mode="markers", name="Residual",
                             marker=dict(color=SECONDARY, opacity=0.55, size=7)))
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
    fig.update_xaxes(title="Predicted score")
    fig.update_yaxes(title="Actual − predicted")
    return _layout(fig, "Residuals (test set)")


def residual_histogram(actual, predicted) -> go.Figure:
    res = np.asarray(actual) - np.asarray(predicted)
    fig = go.Figure(go.Histogram(x=res, nbinsx=30, marker_color=SECONDARY))
    fig.update_xaxes(title="Actual − predicted (points)")
    fig.update_yaxes(title="Students")
    return _layout(fig, "Distribution of errors (test set)")


def feature_importance(values: list[dict]) -> go.Figure:
    df = pd.DataFrame(values).sort_values("importance_mean")
    fig = go.Figure(go.Bar(x=df["importance_mean"], y=df["label"], orientation="h",
                           error_x=dict(type="data", array=df["importance_std"], color=MUTED),
                           marker_color=PRIMARY))
    fig.update_xaxes(title="Increase in MAE when the feature is shuffled (points)")
    return _layout(fig, "Permutation feature importance (validation set)", height=400)


def target_histogram(counts: list[int], edges: list[float]) -> go.Figure:
    centers = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    fig = go.Figure(go.Bar(x=centers, y=counts, width=4.5, marker_color=PRIMARY))
    fig.update_xaxes(title="Final exam score", range=[0, 100])
    fig.update_yaxes(title="Records")
    return _layout(fig, "Final exam scores in the training dataset")


def score_gauge(score: float, support_threshold: float, strong_threshold: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(suffix=" / 100", font=dict(size=34)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color, thickness=0.3),
                   steps=[dict(range=[0, support_threshold], color="#FFEDD5"),
                          dict(range=[support_threshold, strong_threshold], color="#FEF3C7"),
                          dict(range=[strong_threshold, 100], color="#D1FAE5")])))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=10), font=FONT)
    return fig


def batch_distribution(scores, support_threshold: float, strong_threshold: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=scores, nbinsx=20, marker_color=PRIMARY))
    for x in (support_threshold, strong_threshold):
        fig.add_vline(x=x, line_dash="dash", line_color=MUTED)
    fig.update_xaxes(title="Predicted final exam score", range=[0, 100])
    fig.update_yaxes(title="Students")
    return _layout(fig, "Predicted scores in this upload", height=320)
