"""Metrics printing utilities for EnergyPlus RL2 training."""
import csv
import os
from typing import Dict, List, Optional

import numpy as np


# Per-month baseline: first time we see a month, record its mean return.
# Used to show relative improvement (delta) rather than absolute reward,
# since summer/winter months are inherently harder (lower absolute reward).
_month_baseline: Dict[int, float] = {}


def print_epoch_metrics(
    log_dir: str,
    epoch_idx: int,
    last_n_episodes: Optional[int] = None,
) -> None:
    """Read episode_metrics.csv and print per-epoch training progress.

    Shows absolute return AND delta-from-baseline per month, so you can
    see improvement even when summer/winter months have lower raw rewards.
    """
    csv_path = os.path.join(log_dir, "episode_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"[metrics][epoch {epoch_idx}] No episode_metrics.csv yet at {log_dir}")
        return

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    if last_n_episodes is not None and len(rows) > last_n_episodes:
        rows = rows[-last_n_episodes:]

    returns = [float(r["episode_return"]) for r in rows]
    comfort_ratios = [float(r["comfort_ratio"]) for r in rows]
    hvac_powers = [float(r["hvac_power_sum"]) for r in rows]
    months = [int(r["month"]) for r in rows]

    mean_return = np.mean(returns)
    mean_comfort = np.mean(comfort_ratios)
    mean_hvac = np.mean(hvac_powers)
    n = len(rows)

    print(
        f"[train][epoch {epoch_idx:3d}] "
        f"n_eps={n:4d}  "
        f"return={mean_return:+8.4f}  "
        f"comfort={mean_comfort:.3f}  "
        f"hvac_power={mean_hvac:.1f}"
    )

    # Per-month breakdown with delta-from-baseline
    month_stats: Dict[int, Dict] = {}
    for row in rows:
        m = int(row["month"])
        if m not in month_stats:
            month_stats[m] = {"returns": [], "comfort": [], "hvac": []}
        month_stats[m]["returns"].append(float(row["episode_return"]))
        month_stats[m]["comfort"].append(float(row["comfort_ratio"]))
        month_stats[m]["hvac"].append(float(row["hvac_power_sum"]))

    for m in sorted(month_stats):
        s = month_stats[m]
        m_return = float(np.mean(s["returns"]))
        m_comfort = float(np.mean(s["comfort"]))
        m_hvac = float(np.mean(s["hvac"]))

        # Record baseline on first encounter; compute delta afterwards
        if m not in _month_baseline:
            _month_baseline[m] = m_return
            delta_str = "  (baseline)"
        else:
            delta = m_return - _month_baseline[m]
            delta_str = f"  Δ={delta:+.4f}"

        print(
            f"  month={m:2d}  "
            f"return={m_return:+8.4f}{delta_str}  "
            f"comfort={m_comfort:.3f}  "
            f"hvac={m_hvac:.1f}"
        )


def print_validation_summary(
    epoch_idx: int,
    yearly_results: List[Dict],
    summary: Dict,
) -> None:
    """Print validation results after full-year evaluation."""
    print(
        f"[valid][epoch {epoch_idx:3d}] "
        f"year_return={summary['year_total_return']:+10.3f}  "
        f"comfort={summary.get('year_comfort_ratio', float('nan')):.3f}  "
        f"hvac={summary.get('year_hvac_power', float('nan')):.1f}"
    )
    for r in yearly_results:
        if "month" in r:
            print(
                f"  month={r['month']:2d}  "
                f"return={r.get('r', 0.0):+8.3f}  "
                f"comfort={r.get('comfort_ratio', float('nan')):.3f}  "
                f"hvac={r.get('hvac_power', float('nan')):.1f}"
            )
