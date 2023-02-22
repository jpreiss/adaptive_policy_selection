import itertools as it
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REGRET = "cost difference, ours - LQR"


def main():

    if os.getenv("FAST").lower() != "true":
        plt.rc("text", usetex=True)
        plt.rc("font", size=12)

    noises = ["walk", "gaussian"]

    dfs = []
    for noise in noises:
        data = np.load(f"data/pendulum_{noise}.npz")
        dt = data["dt"]
        cost_log = data["cost_log"]
        time = dt * np.arange(len(cost_log))
        cost_cumulative = np.cumsum(cost_log, axis=0).T
        regret = (cost_cumulative[0] - cost_cumulative[1]).squeeze()
        dfs.append(pd.DataFrame({
            "time": time,
            REGRET: regret,
            "disturbance": noise,
        }))
    df = pd.concat(dfs, ignore_index=True)

    # Too much data makes lines "gritty".
    df = df[::100]

    sns.set_style("ticks", {"axes.grid" : True})
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5), constrained_layout=True)
    sns.lineplot(
        data=df,
        ax=ax,
        #kind="line",
        x="time",
        y=REGRET,
        size="disturbance",
        color="black",
        #height=2.7,
        #aspect=1.4,
    )
    sns.despine(ax=ax)
    # Need to increase a little to make sure grid lines aren't clipped.
    ax.set(xticks=200*np.arange(6))
    ax.set(xlim=[0, 1002], ylim=[-8000, 5010])
    fig.savefig("plots/pendulum_costs.pdf")


if __name__ == "__main__":
    main()
