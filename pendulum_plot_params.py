import itertools as it
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():

    if os.getenv("FAST").lower() != "true":
        plt.rc("text", usetex=True)
        plt.rc("font", size=12)

    noises = ["gaussian", "walk"]

    dfs = []
    for noise in noises:
        data = np.load(f"data/pendulum_{noise}.npz")
        dt = data["dt"]
        for controller in ["ours", "LQ"]:
            theta_log = data["theta_log_" + controller]
            time = dt * np.arange(len(theta_log))
            for value, paramname in zip(theta_log.T, ["kp", "kd"]):
                dfs.append(pd.DataFrame(dict(
                    time=time,
                    gain=value,
                    param=paramname,
                    controller=controller,
                    disturbance=noise,
                )))
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
        height=2.5,
        aspect=1.2,
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
