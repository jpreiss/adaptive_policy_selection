all: plots/dis_vs_cts_regret.pdf plots/pendulum_params.pdf plots/pendulum_costs.pdf plots/lambda_confident_comparison.pdf

plots/pendulum_params.pdf: pendulum_plot_params.py data/pendulum_gaussian.npz data/pendulum_walk.npz
	mkdir -p plots
	python $^

plots/pendulum_costs.pdf: pendulum_plot_costs.py data/pendulum_gaussian.npz data/pendulum_walk.npz
	mkdir -p plots
	python $^

plots/dis_vs_cts_regret.pdf: discrete_vs_cts_plot.py data/discrete_vs_cts.npz
	mkdir -p plots
	python $<

plots/lambda_confident_comparison.pdf: lambda_confident_comparison.py
	mkdir -p plots
	python $<

data/pendulum_gaussian.npz: pendulum.py
	mkdir -p data
	python $< $@

data/pendulum_walk.npz: pendulum.py
	mkdir -p data
	python $< --walk $@

data/discrete_vs_cts.npz: discrete_vs_cts.py
	mkdir -p data
	python $<

clean:
	rm -f plots/*.pdf
	rm -f data/*.npz
