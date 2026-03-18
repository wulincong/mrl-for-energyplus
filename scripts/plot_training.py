"""Visualize RL2TRPO training progress for EnergyPlus 5-zone control.

Usage:
    python scripts/plot_training.py [--log_dir eplog/garage-rl2-trpo] [--out training.png]
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
SEASON_COLOR = {  # rough season grouping for month coloring
    1:"#4e79a7", 2:"#4e79a7", 3:"#76b7b2",
    4:"#59a14f", 5:"#59a14f", 6:"#f28e2b",
    7:"#e15759", 8:"#e15759", 9:"#f28e2b",
    10:"#76b7b2", 11:"#4e79a7", 12:"#4e79a7",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_episode_metrics(log_dir: str) -> pd.DataFrame:
    path = os.path.join(log_dir, "episode_metrics.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Assign epoch: group consecutive rows with the same timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Each epoch has the same timestamp; assign epoch index
    ts_groups = df["timestamp"].ne(df["timestamp"].shift()).cumsum() - 1
    df["epoch"] = ts_groups
    # Per-step return for fair comparison across episodes of different lengths
    df["return_per_step"] = df["episode_return"] / df["episode_steps"].clip(lower=1)
    return df


def load_yearly_validation(log_dir: str):
    """Returns (epoch_summaries: list[dict], monthly_df: pd.DataFrame)."""
    path = os.path.join(log_dir, "yearly_validation.csv")
    if not os.path.exists(path):
        return [], pd.DataFrame()

    summaries = []
    data_lines = []
    header_line = None

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                summaries.append(json.loads(line[1:]))
            elif header_line is None:
                header_line = line  # CSV header
            else:
                data_lines.append(line)

    if not data_lines or header_line is None:
        return summaries, pd.DataFrame()

    from io import StringIO
    csv_text = header_line + "\n" + "\n".join(data_lines)
    df = pd.read_csv(StringIO(csv_text))

    # Each epoch = 12 consecutive rows (one per month)
    df["epoch"] = df.index // 12
    return summaries, df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def epoch_train_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-epoch mean return, comfort, hvac from episode_metrics."""
    return (
        df.groupby("epoch")
        .agg(
            mean_return=("episode_return", "mean"),
            mean_return_per_step=("return_per_step", "mean"),
            mean_comfort=("comfort_ratio", "mean"),
            mean_hvac=("hvac_power_sum", "mean"),
            month=("month", lambda x: x.mode()[0]),  # dominant month this epoch
        )
        .reset_index()
    )


def per_month_train_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per (epoch, month) mean return and comfort."""
    return (
        df.groupby(["epoch", "month"])
        .agg(
            mean_return=("episode_return", "mean"),
            mean_return_per_step=("return_per_step", "mean"),
            mean_comfort=("comfort_ratio", "mean"),
        )
        .reset_index()
    )


def smooth(values, window=5):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(log_dir: str, out_path: str):
    ep_df = load_episode_metrics(log_dir)
    summaries, val_df = load_yearly_validation(log_dir)

    has_train = not ep_df.empty
    has_val = not val_df.empty

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("EnergyPlus RL2TRPO Training Progress", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # -----------------------------------------------------------------------
    # Row 0: Return over epochs
    # -----------------------------------------------------------------------
    ax_ret_train = fig.add_subplot(gs[0, 0])
    ax_ret_val   = fig.add_subplot(gs[0, 1])

    if has_train:
        stats = epoch_train_stats(ep_df)
        ax_ret_train.scatter(stats["epoch"], stats["mean_return"],
                             c=[SEASON_COLOR.get(m, "#aaa") for m in stats["month"]],
                             s=20, alpha=0.6, zorder=3, label="epoch (colored by month)")
        ax_ret_train.plot(stats["epoch"], smooth(stats["mean_return"].values),
                          color="black", lw=1.5, label="smoothed")
        ax_ret_train.set_title("Training: Mean Episode Return per Epoch")
        ax_ret_train.set_xlabel("Epoch")
        ax_ret_train.set_ylabel("Mean Return")
        ax_ret_train.legend(fontsize=7)
        ax_ret_train.grid(True, alpha=0.3)

    if has_val:
        val_epoch_stats = val_df.groupby("epoch").agg(
            year_return=("r", "sum"),
            mean_comfort=("comfort_ratio", "mean"),
            mean_hvac=("hvac_power", "mean"),
        ).reset_index()
        ax_ret_val.plot(val_epoch_stats["epoch"], val_epoch_stats["year_return"],
                        color="#e15759", lw=2, marker="o", markersize=4)
        ax_ret_val.set_title("Validation: Year Total Return per Epoch")
        ax_ret_val.set_xlabel("Epoch")
        ax_ret_val.set_ylabel("Sum of Monthly Returns")
        ax_ret_val.grid(True, alpha=0.3)

    # -----------------------------------------------------------------------
    # Row 1: Comfort ratio over epochs
    # -----------------------------------------------------------------------
    ax_com_train = fig.add_subplot(gs[1, 0])
    ax_com_val   = fig.add_subplot(gs[1, 1])

    if has_train:
        ax_com_train.scatter(stats["epoch"], stats["mean_comfort"],
                             c=[SEASON_COLOR.get(m, "#aaa") for m in stats["month"]],
                             s=20, alpha=0.6, zorder=3)
        ax_com_train.plot(stats["epoch"], smooth(stats["mean_comfort"].values),
                          color="black", lw=1.5)
        ax_com_train.axhline(1.0, color="green", lw=0.8, ls="--", alpha=0.5, label="perfect")
        ax_com_train.set_title("Training: Mean Comfort Ratio per Epoch\n(fraction of steps with zone temp 22–25°C)")
        ax_com_train.set_xlabel("Epoch")
        ax_com_train.set_ylabel("Comfort Ratio")
        ax_com_train.set_ylim(0, 1.05)
        ax_com_train.legend(fontsize=7)
        ax_com_train.grid(True, alpha=0.3)

    if has_val:
        ax_com_val.plot(val_epoch_stats["epoch"], val_epoch_stats["mean_comfort"],
                        color="#59a14f", lw=2, marker="o", markersize=4)
        ax_com_val.axhline(1.0, color="green", lw=0.8, ls="--", alpha=0.5)
        ax_com_val.set_title("Validation: Mean Comfort Ratio per Epoch")
        ax_com_val.set_xlabel("Epoch")
        ax_com_val.set_ylabel("Comfort Ratio")
        ax_com_val.set_ylim(0, 1.05)
        ax_com_val.grid(True, alpha=0.3)

    # -----------------------------------------------------------------------
    # Row 2: Per-month return heatmap (validation) + HVAC power
    # -----------------------------------------------------------------------
    ax_heatmap = fig.add_subplot(gs[2, 0])
    ax_hvac    = fig.add_subplot(gs[2, 1])

    if has_val:
        # Heatmap: months × epochs, color = return
        pivot = val_df.pivot_table(index="month", columns="epoch", values="r", aggfunc="mean")
        im = ax_heatmap.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                               origin="lower", interpolation="nearest")
        ax_heatmap.set_yticks(range(len(pivot.index)))
        ax_heatmap.set_yticklabels([MONTH_NAMES[m-1] for m in pivot.index], fontsize=8)
        ax_heatmap.set_xlabel("Epoch")
        ax_heatmap.set_title("Validation Return by Month × Epoch\n(green = higher return)")
        plt.colorbar(im, ax=ax_heatmap, label="Return")

        ax_hvac.plot(val_epoch_stats["epoch"], val_epoch_stats["mean_hvac"] / 1000,
                     color="#f28e2b", lw=2, marker="o", markersize=4)
        ax_hvac.set_title("Validation: Mean HVAC Power per Epoch")
        ax_hvac.set_xlabel("Epoch")
        ax_hvac.set_ylabel("HVAC Power (kW·step)")
        ax_hvac.grid(True, alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Per-month improvement figure (separate)
# ---------------------------------------------------------------------------

def plot_per_month_improvement(log_dir: str, out_path: str):
    """Show each month's return trajectory over validation epochs."""
    _, val_df = load_yearly_validation(log_dir)
    if val_df.empty:
        print("No validation data found.")
        return

    months = sorted(val_df["month"].unique())
    n_epochs = val_df["epoch"].max() + 1

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=False)
    fig.suptitle("Validation Return per Month over Epochs\n"
                 "(shows improvement within each month independently)",
                 fontsize=13, fontweight="bold")

    for idx, month in enumerate(months):
        ax = axes[idx // 4][idx % 4]
        mdf = val_df[val_df["month"] == month].sort_values("epoch")
        color = SEASON_COLOR.get(month, "#aaa")

        ax.plot(mdf["epoch"], mdf["r"], color=color, lw=2, marker="o", markersize=3)
        # Shade comfort ratio on secondary axis
        ax2 = ax.twinx()
        ax2.fill_between(mdf["epoch"], mdf["comfort_ratio"],
                         alpha=0.15, color="green", label="comfort")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Comfort", fontsize=7, color="green")
        ax2.tick_params(axis="y", labelsize=6, labelcolor="green")

        # Baseline delta annotation
        if len(mdf) > 1:
            delta = mdf["r"].iloc[-1] - mdf["r"].iloc[0]
            sign = "+" if delta >= 0 else ""
            ax.set_title(f"{MONTH_NAMES[month-1]}  Δ={sign}{delta:.1f}", fontsize=9)
        else:
            ax.set_title(MONTH_NAMES[month-1], fontsize=9)

        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Return", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="eplog/garage-rl2-trpo")
    parser.add_argument("--out", default="training_progress.png")
    parser.add_argument("--out_monthly", default="monthly_improvement.png")
    args = parser.parse_args()

    plot(args.log_dir, args.out)
    plot_per_month_improvement(args.log_dir, args.out_monthly)
