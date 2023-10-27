# Online Adaptive Policy Selection

Numerical experiments for the manuscript:

*Online Adaptive Policy Selection in Time-Varying Systems: No-Regret via Contractive Perturbations*<br/>
Yiheng Lin, James A. Preiss, Emile Anand, Yingying Li, Yisong Yue, Adam Wierman<br/>
Caltech<br/>
NeurIPS 2023 (to appear).</br>
https://arxiv.org/abs/2210.12320

## To reproduce the plots in the manuscript:

1. Create the Anaconda environment (or get the required packages some other way):

       conda env create -f environment.yml
       conda activate gaps

2. Run the experiments in parallel. This will take a while - about 30 minutes on an old Mac laptop. The system may become unresponsive.

       make -j

3. Examine output in the `plots/` directory.

## Code structure

### Algorithms / Library
- `GAPS.py`: main algorithm of the manuscript
- `discrete.py`, `exp3.py`: bandit-based algorithm for MPC horizon selection
- `MPCLTI.py`, `lineartracking.py`: MPC algorithm and linear dynamical system "environment"
- `label_lines.py` (third-party), `util.py`: utilities

### Experiments
- `lambda_confident.py`: Experiment 1 - MPC scalar confidence tuning, GAPS vs. baseline
- `pendulum.py`: Experiment 2 - Linear controller in nonlinear system, GAPS vs. LQR
- `discrete_vs_cts.py`: Experiment 3 (appendix) - GAPS MPC multi-step confidence tuning vs. bandit MPC horizon tuning
- `gaps_vs_ocom.py`: Computation time experiment (appendix) - Computation time experiment vs. another gradient approximation [1]
- Any `*plot*.py`: Plots data written to file by one of the experiments above.

### Other
- `Makefile`: Establishes data-to-plot pipeline dependencies between files.
- `environment.yml`: Anaconda environment.

See file docstrings for more information.
