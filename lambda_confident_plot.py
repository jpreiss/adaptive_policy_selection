import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from util import fastmode


PARAM = "confidence $\\lambda$"
TIME = "time"


def main():
    paths = sys.argv[1:]

    dfs = []
    for path in paths:
        seed = int(path.split("_")[-1].split(".")[0])
        data = np.load(path)
        dfs.append(pd.DataFrame({
            "seed": seed,
            "cost": data["gaps_cost"],
            PARAM: data["gaps_param"],
            "algorithm": "GAPS (ours)",
        }))
        dfs.append(pd.DataFrame({
            "seed": seed,
            "cost": data["lambda_cost"],
            PARAM: data["lambda_param"],
            "algorithm": "$\\lambda$-confident",
        }))
    df = pd.concat(dfs)

    if False:
        # Smoothing.
        gp = df.groupby(["algorithm", "seed"])
        roll = gp.rolling(5, center=True)
        m = roll.mean()
        df = m.reset_index()
        # TODO: There must be a less hacky way...
        df[TIME] = df["level_2"]
    else:
        df[TIME] = df.index

    df = df.melt(id_vars=["algorithm", TIME, "seed"])

    if not fastmode():
        plt.rc("text", usetex=True)
        plt.rc("font", size=11)

    order = sorted(df["algorithm"].unique())[::-1]

    grid = sns.relplot(
        kind="line",
        data=df,
        x=TIME,
        y="value",
        hue="algorithm",
        hue_order=order,
        style="algorithm",
        style_order=order,
        errorbar=("pi", 90),
        col="variable",
        col_order=[PARAM, "cost"],
        height=2.3,
        aspect=1.6,
        facet_kws=dict(
            sharey=False,
        ),
        zorder=3,
    )

    grid.refline(
        x=data["become_good"],
        label="$\\widehat w$ becomes accurate",
        color="black",
        linewidth=2.5,
        linestyle="solid",
    )
    # Redo legend to include refline.
    grid.legend.remove()
    grid.add_legend(title=None)

    grid.set_titles("")
    ax_lam, ax_err = grid.axes.flat
    ax_lam.set(ylabel=PARAM, ylim=[0.0, 1.05])
    ax_err.set(ylabel="cost", ylim=[0, 6])

    for ax in grid.axes.flat:
        ax.set(xlim=[-10, 400])
        ax.grid(True)

    grid.savefig("plots/lambda_confident.pdf")


if __name__ == '__main__':
    main()
