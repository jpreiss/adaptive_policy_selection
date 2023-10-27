all: plots/dis_vs_cts_regret.pdf plots/pendulum_params.pdf plots/pendulum_costs.pdf plots/lambda_confident.pdf plots/gaps_vs_ocom_regret.pdf

# ---------------------------------- SEEDS ---------------------------------- #
SEEDS32 = $(shell python -c "print(' '.join([str(i) for i in range(32)]))")
SEEDS100 = $(shell python -c "print(' '.join([str(i) for i in range(100)]))")

PENDULUMS := \
$(SEEDS32:%=data/pendulum_gaussian_%.npz) \
$(SEEDS32:%=data/pendulum_walk_%.npz)

LAMBDAS := \
$(SEEDS100:%=data/lambda_confident_%.npz)

# ---------------------------------- PLOTS ---------------------------------- #
plots/pendulum_params.pdf: pendulum_plot_params.py $(PENDULUMS)
	mkdir -p plots
	python $^

plots/pendulum_costs.pdf: pendulum_plot_costs.py $(PENDULUMS)
	mkdir -p plots
	python $^

plots/dis_vs_cts_regret.pdf: discrete_vs_cts_plot.py data/discrete_vs_cts.npz
	mkdir -p plots
	python $<

plots/gaps_vs_ocom_regret.pdf: gaps_vs_ocom_plot.py data/gaps_vs_ocom_GAPS.npz
	mkdir -p plots
	python $< data/gaps_vs_ocom*.npz

plots/lambda_confident.pdf: lambda_confident_plot.py $(LAMBDAS)
	mkdir -p plots
	python $^

# ---------------------------------- DATA ----------------------------------- #
data/lambda_confident_%.npz: lambda_confident.py
	mkdir -p data
	python $< $* $@

data/pendulum_gaussian_%.npz: pendulum.py
	mkdir -p data
	python $< $* $@

data/pendulum_walk_%.npz: pendulum.py
	mkdir -p data
	python $< --walk $* $@

data/discrete_vs_cts.npz: discrete_vs_cts.py discrete.py lineartracking.py MPCLTI.py
	mkdir -p data
	python $<

data/gaps_vs_ocom_GAPS.npz: gaps_vs_ocom.py lineartracking.py MPCLTI.py
	mkdir -p data
	python $<


clean:
	rm -f plots/*.pdf
	rm -f data/*.npz
