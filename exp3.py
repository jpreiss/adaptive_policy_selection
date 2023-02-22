import matplotlib.pyplot as plt
import numpy as np


def exp3(rng, arms, T=None, rate=None):
    """EXP3 algorithm for adversarial MAB with costs (not rewards).

    Is a generator. Use like:

        rng = np.random.default_rng()
        algo = exp3(...)
        loss = None
        for t in range(T):
            i = algo.send(loss)
            loss = <get loss from the world>

    Based on Haipeng Luo's lecture notes:
    https://haipeng-luo.net/courses/CSCI699_2019/lecture8.pdf
    with multiplicative-update optimization implemented.
    """

    # Choose optimal rate for known T.
    if rate is None:
        assert T is not None
        rate = np.sqrt(np.log(arms) / (T * arms))

    # Initialize weights with exp(0).
    y = np.ones(arms)

    while True:
        y /= np.sum(y)
        i = rng.choice(arms, p=y)
        loss = yield i
        y[i] *= np.exp(-rate * loss / y[i])


def main():

    # Small instance for testing.
    k = 10
    T = 1000

    # This will be a simple instance with constant rewards.
    # Not actually adversarial.
    # The sqrt increases the gap between best and second-best.
    costs = np.sqrt(np.linspace(0, 1, k))

    # Initialize the algorithm.
    rng = np.random.default_rng()
    algo = exp3(rng, k, T)

    # Run the algorithm.
    selection_log = np.zeros(T, dtype=int)
    loss = None
    for t in range(T):
        i = algo.send(loss)
        selection_log[t] = i
        loss = costs[i]

    # Compute regret accumulation.
    cost_history = costs[selection_log]
    opt_history = np.repeat(costs[0], T)
    regret = np.cumsum(cost_history - opt_history)

    # Plot.
    fig, (ax_choices, ax_regret) = plt.subplots(1, 2, figsize=(8, 3))
    ax_choices.plot(selection_log, linewidth=0, marker="o", markersize=2)
    ax_choices.set_ylabel("arm selection")

    ax_regret.plot(regret)
    ax_regret.set_ylabel("regret")

    for ax in (ax_choices, ax_regret):
        ax.set_xlabel("time")

    plt.savefig("exp3.pdf")


if __name__ == "__main__":
    main()
