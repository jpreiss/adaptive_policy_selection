import os
import multiprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are
import seaborn as sns

from discrete import MPCHorizonSelector
from lineartracking import LinearTracking
from label_lines import labelLines
from MPCLTI import MPCLTI


LIGHT_GREY = "#BBBBBB"
ARM = "MPC horizon"
HEIGHT = 2.8


def reverse_category(df, col):
    cat = df[col].astype("category")
    cat = cat.cat.set_categories(cat.cat.categories[::-1], ordered=True)
    df[col] = cat


def main():

    zip = np.load("data/discrete_vs_cts.npz")
    exp3_batch = zip["exp3_batch"]
    horizon_cost_histories = zip["horizon_cost_histories"]
    dis_cost_history = zip["dis_cost_history"]
    dis_arm_history = zip["dis_arm_history"]
    cts_param_history = zip["cts_param_history"]
    cts_opt_cost_history = zip["cts_opt_cost_history"]
    cts_cost_history = zip["cts_cost_history"]

    # Infer dimensions.
    n_horizons, T = horizon_cost_histories.shape
    horizons = np.arange(n_horizons)
    max_horizon = n_horizons - 1
    n_batches = len(dis_arm_history)

    # Reconstruct some intermediate data.
    step_losses = np.mean(horizon_cost_histories, axis=1)

    plt.rc("figure", autolayout=True)
    if os.getenv("FAST").lower() != "true":
        plt.rc("text", usetex=True)
        plt.rc("font", size=12)

    # Plot the mean per-step cost of each MPC horizon with full trust.
    batches = horizon_cost_histories[:, :n_batches*exp3_batch].reshape((n_horizons, n_batches, exp3_batch))
    batch_means = np.mean(batches, axis=-1)
    dfs = []
    for k, means in enumerate(batch_means):
        assert len(means.shape) == 1
        dfs.append(pd.DataFrame({"mean cost": means, ARM: k}))
    df_horizons = pd.concat(dfs, ignore_index=True)
    reverse_category(df_horizons, ARM)
    grid = sns.catplot(
        data=df_horizons,
        kind="violin",
        bw=0.2,
        linewidth=0.5,
        orient="h",
        x="mean cost",
        y=ARM,
        height=HEIGHT,
        aspect=1.1,
        cut=0,
        inner="quartiles",
        color=LIGHT_GREY,
    )
    for ax in grid.axes.flatten():
        ax.grid(True, axis="x")
        for c in ax.collections:
            c.set_edgecolor("black")
    grid.savefig("plots/batch_sum_hists.pdf")

    # Plot the behavior of EXP3.
    BATCH = "BAPS batch"
    df_exp3 = pd.DataFrame({
        BATCH: np.arange(len(dis_arm_history)),
        ARM: dis_arm_history,
    })
    reverse_category(df_exp3, ARM)
    grid = sns.catplot(
        kind="swarm",
        s=1.8,
        data=df_exp3,
        x=BATCH,
        y=ARM,
        hue=ARM,
        height=HEIGHT,
        aspect=1.4,
        legend=False,
    )
    lines = np.unique(dis_arm_history) + 0.5
    lines = [-0.5] + list(lines)
    xmin = df_exp3[BATCH].min()
    xmax = df_exp3[BATCH].max()
    pad = xmax / 60.0
    width = 0.5
    for ax in grid.axes.flat:
        ax.set_yticks(lines, minor=True)
        ax.yaxis.grid(which="minor", linewidth=width, color="black")
        ax.yaxis.set_tick_params(which="major", length=0)
        ax.yaxis.set_tick_params(which="minor", length=14, width=width)
        ax.spines[:].set_linewidth(width)
        sns.despine(ax=ax, left=True, top=False)
    grid.set(xlim=[xmin - pad, xmax + pad])
    grid.savefig("plots/exp3_scatter.pdf")

    # Subsample line plots for less "grittiness".
    skip = T // 1000
    time_skip = np.arange(T)[::skip]

    # Plot the evolution of the policy parameters.
    THETA_I = "$\\theta_i$"
    df_gaps = pd.DataFrame(cts_param_history[::skip])
    df_gaps["time"] = time_skip
    df_gaps = pd.melt(df_gaps, id_vars="time", var_name=THETA_I)
    fig_params = sns.relplot(
        kind="line",
        data=df_gaps,
        x="time",
        y="value",
        hue=THETA_I,
        palette="flare",
        legend=False,
        height=HEIGHT,
        aspect=1.2,
    )
    ax_params = fig_params.axes[0, 0]
    lines = ax_params.get_lines()
    for i, line in enumerate(lines):
        line.set_label(f"$\\theta_{i}$")
    xvals = [0.6 * np.amax(lines[0].get_xdata())] * len(lines)
    labelLines(lines, align=False, xvals=xvals)
    ax_params.set(xlim=[0.0, 1e6], ylim=[-0.05, 0.9])
    sns.despine(ax=ax_params)
    fig_params.savefig("plots/params_update.pdf")

    # Regret analysis.
    optimal_horizon = np.argmin(step_losses)
    print(f"{optimal_horizon = }")
    dis_opt_cost_history = horizon_cost_histories[optimal_horizon]
    #reg_vs = np.cumsum(dis_cost_history - cts_cost_history)
    reg_dis = np.cumsum(dis_cost_history - dis_opt_cost_history)
    reg_cts = np.cumsum(cts_cost_history - cts_opt_cost_history)
    df_reg = pd.DataFrame({
        "time": time_skip,
        "BAPS vs.\\ optimal": reg_dis[::skip],
        "GAPS vs.\\ final": reg_cts[::skip],
        #"BAPS vs. GAPS": reg_vs[::skip],
    })
    REGRET = "cumulative cost diff."
    df_reg = pd.melt(df_reg, id_vars="time", var_name="algorithm", value_name=REGRET)
    grid = sns.relplot(
        data=df_reg,
        kind="line",
        x="time",
        y=REGRET,
        col="algorithm",
        color="black",
        height=HEIGHT,
        aspect=0.75,
    )
    ax = grid.axes[0, 0]
    ax.ticklabel_format(scilimits=(0, 0))
    ax.yaxis.offsetText.set(ha="right")
    grid.set_titles(col_template="{col_name}")
    grid.savefig("plots/dis_vs_cts_regret.pdf")

    # Show the advantage of using trust values instead of horizon tuning.
    dis_total = np.sum(dis_opt_cost_history)
    cts_total = np.sum(cts_opt_cost_history)
    print(f"LQ cost: optimal discrete = {dis_total:.1f}, optimal continuous = {cts_total:.1f}")
    print(f"         (ratio = {cts_total/dis_total:.2f})")


if __name__ == '__main__':
    main()
