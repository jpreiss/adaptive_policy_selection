import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from label_lines import labelLines
from util import fastmode


LIGHT_GREY = "#BBBBBB"
HEIGHT = 2.8
COMPTIME = "cumulative compute time (sec)"


def main():

    plt.rc("figure", autolayout=True)
    if not fastmode():
        plt.rc("text", usetex=True)
        plt.rc("font", size=12)

    paths = sys.argv[1:]
    regret_dfs = []

    best_cost = np.inf
    best_cost_history = None
    print(paths)
    for path in paths:
        zip = np.load(path)
        opt_cost_history = zip["opt_cost_history"]
        total_cost = np.sum(opt_cost_history)
        name = path.split(".")[-2].split("_")[-1]
        if total_cost < best_cost:
            best_cost = total_cost
            best_cost_history = opt_cost_history
            print("updated best cost:", name)

    for path in paths:
        zip = np.load(path)
        name = path.split(".")[-2].split("_")[-1]
        param_history = zip["param_history"]
        cost_history = zip["cost_history"]
        times = zip["times"]

        # Infer dimensions.
        T = cost_history.size

        # Regret and computation time analysis.
        regret = np.cumsum(cost_history - best_cost_history[:len(cost_history)])
        # remove transient spike and make same length as others
        times = np.concatenate([[0, 0], times[1:]])
        cumtime = np.cumsum(times)
        regret_dfs.append(
            pd.DataFrame({
                "timestep": np.arange(T),
                "regret": regret,
                "gradient": name,
                COMPTIME: cumtime,
            })
        )
        print(f"{name}:")
        print(f"    {len(regret)} steps")
        print(f"    total time {cumtime[-1]}")
        print(f"    final regret {regret[-1]}")

    hue_order = ["GAPS", "OCO-M", "Ideal OGD"]

    regret_df = pd.concat(regret_dfs, ignore_index=True)
    grid = sns.relplot(
        data=regret_df,
        kind="line",
        x="timestep",
        y="regret",
        hue="gradient",
        hue_order=hue_order,
        height=HEIGHT,
        aspect=1.4,
        legend=False,
    )
    grid.savefig(f"plots/gaps_vs_ocom_regret.pdf")

    grid2 = sns.relplot(
        data=regret_df,
        kind="line",
        x="timestep",
        y=COMPTIME,
        hue="gradient",
        hue_order=hue_order,
        height=HEIGHT,
        aspect=1.4,
    )
    grid2.set(ylim=[-5, 100])
    grid2.savefig("plots/gaps_vs_ocom_comptime.pdf")


if __name__ == '__main__':
    main()
