import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from util import fastmode


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
        for controller in ["ours", "LQ"]:
            theta_log = data["theta_log_" + controller]
            time = dt * np.arange(len(theta_log))
            for value, paramname in zip(theta_log.T, ["$k_p$", "$k_d$"]):
                df = pd.DataFrame(dict(
                    time=time,
                    gain=value,
                    param=paramname,
                    controller=controller,
                    disturbance=noise,
                    seed=seed,
                ))
                dfs.append(df[::100])  # Resample.
    df = pd.concat(dfs, ignore_index=True)

    sns.set_style("ticks", {"axes.grid" : True})
    param_grid = sns.relplot(
        data=df,
        kind="line",
        col="disturbance",
        x="time",
        y="gain",
        style="controller",
        size="controller",
        sizes=[1.5, 1.0],
        hue="param",
        # use magenta/cyan for params, save orange/blue for ours-vs-theirs.
        palette=[(0.9, 0, 0.8), (0, 0.75, 0.85)],
        height=2.0,
        aspect=1.5,
        errorbar="sd",
        facet_kws=dict(
            gridspec_kws=dict(
                hspace=0.05,
            )
        )
    )
    # Python's sloppy scoping - `time` was defined in loading loop.
    param_grid.set(xticks=200 * np.arange(6))
    param_grid.set(xlim=[time[0], time[-1] + dt])
    param_grid.savefig("plots/pendulum_params.pdf")


if __name__ == "__main__":
    main()
