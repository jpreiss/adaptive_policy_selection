import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from util import fastmode


REGRET = "``regret'' GAPS - LQR"


def main():

    if not fastmode():
        plt.rc("text", usetex=True)
        plt.rc("font", size=12)

    paths = sys.argv[1:]

    dfs = []
    for path in paths:
        _, noise, seed = path.split("_")
        seed, _ = seed.split(".")
        seed = int(seed)
        data = np.load(path)
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
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.25), constrained_layout=True)
    sns.lineplot(
        data=df,
        ax=ax,
        x="time",
        y=REGRET,
        size="disturbance",
        color="black",
        errorbar="sd",
    )
    sns.despine(ax=ax)
    # Need to increase a little to make sure grid lines aren't clipped.
    ax.set(xticks=200*np.arange(6))
    ax.set(xlim=[0, 1002], ylim=[-13000, 5050])
    ax.legend(title="disturbance", handlelength=1.5)
    formatter = EngFormatter(sep="")
    ax.yaxis.set_major_formatter(formatter)
    fig.savefig("plots/pendulum_costs.pdf")


if __name__ == "__main__":
    main()
