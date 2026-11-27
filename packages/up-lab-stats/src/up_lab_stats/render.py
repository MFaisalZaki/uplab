"""The `anytime` renderer: summed IPC quality score over time."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from up_lab.results_plugin import figure
from up_lab_stats.analyses import UPLabStatsError

COLUMNS = ["variant", "run", "cpu_time", "cost", "reference_cost"]


def anytime_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summed quality score of every variant as a step function of time.

    The score of a run at time `t` is the IPC satisficing score of the best
    plan found until `t`: reference cost / cost, at most 1, and 0 before the
    first plan. The scores of all runs of a variant in the table are summed.

    :param df: The result of the query `anytime-quality`: one row per valid
        plan with `variant`, `run`, `cpu_time`, `cost` and `reference_cost`.
    :return: `variant`, `time`, `score`: the score from each time on.
    """
    missing = [column for column in COLUMNS if column not in df.columns]
    if missing:
        raise UPLabStatsError(f"The anytime renderer needs the columns {missing}.")
    df = df.dropna(subset=COLUMNS).sort_values("cpu_time", kind="stable")
    best = df.groupby("run")["cost"].cummin()
    quality = (df["reference_cost"] / best.where(best > 0)).fillna(1.0).clip(upper=1.0)
    gain = quality - quality.groupby(df["run"]).shift(fill_value=0.0)
    steps = gain.groupby([df["variant"], df["cpu_time"]]).sum()
    score = steps.groupby(level="variant").cumsum()
    return score.rename("score").rename_axis(["variant", "time"]).reset_index()


def _draw(ax: Any, df: pd.DataFrame, options: Dict[str, Any]) -> None:
    steps = anytime_steps(df)
    end = options.get("time_limit") or steps["time"].max()
    for name, part in steps.groupby("variant"):
        times = [part["time"].iloc[0], *part["time"], end]
        scores = [0.0, *part["score"], part["score"].iloc[-1]]
        ax.step(times, scores, where="post", label=str(name))
    ax.set_xscale(options.get("xscale", "log"))
    ax.set_xlabel(options.get("xlabel", "CPU time (s)"))
    ax.set_ylabel(options.get("ylabel", "sum of quality scores"))
    if len(steps):
        ax.legend(title="variant", frameon=False)


def render_anytime(df: pd.DataFrame, stem: Path, options: Dict[str, Any]) -> List[Path]:
    """
    Plots, per variant, the summed quality score of :func:`anytime_steps`
    against time and saves it as PDF and PNG.

    :param df: The result of the query `anytime-quality`.
    :param stem: The output path without extension.
    :param options: `time_limit` (where the curves end; default: the last
        plan), `xscale` (default `log`), `xlabel`, `ylabel`, `title`,
        `figsize`.
    :return: The written files.
    """
    import matplotlib.pyplot as plt

    fig = figure(_draw, df, options)
    paths = [stem.with_name(f"{stem.name}.{ext}") for ext in ("pdf", "png")]
    for path in paths:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    return paths
