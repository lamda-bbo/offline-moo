from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.indicators.hv import HV 
from pymoo.indicators.igd import IGD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

color_set = {
	'Amaranth': np.array([0.9, 0.17, 0.31]), 
	'Amber': np.array([1.0,0.49,0.0]),  
	'Bleu de France': np.array([0.19,0.55,0.91]),
	'Electric violet': np.array([0.56, 0.0, 1.0]),
	'Arsenic': np.array([0.23, 0.27, 0.29]),
	'Blush': np.array([0.87, 0.36, 0.51]),
	'Dark sea green': np.array([0.56,0.74,0.56]),
	'Dark electric blue': np.array([0.33,0.41,0.47]),
	'Dark gray': np.array([0.66, 0.66, 0.66]),
	'French beige': np.array([0.65, 0.48, 0.36]),
	'Grullo': np.array([0.66, 0.6, 0.53]),
	'Dark coral': np.array([0.8, 0.36, 0.27]),
	'Old lavender': np.array([0.47, 0.41, 0.47]),
	'Sandy brown': np.array([0.96, 0.64, 0.38]),
	'Dark cyan': np.array([0.0, 0.55, 0.55]),
	'Brick red': np.array([0.8, 0.25, 0.33]),
	'Dark pastel green': np.array([0.01, 0.75, 0.24])
}

env_name = 'zdt3'

ref_dirs = get_reference_directions('uniform', 2, n_partitions=15)

name_to_algo = {
    'nsga2': NSGA2(),
    'nsga3': NSGA3(ref_dirs=ref_dirs),
    'moead': MOEAD(ref_dirs=ref_dirs),
}
try:
    problem = get_problem(env_name)
except:
    import os, sys
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
    from off_moo_bench.problem import get_problem as m2bo_bench_get
    problem = m2bo_bench_get(env_name)
try:
    nadir_point = problem.nadir_point()
except:
    nadir_point = problem.get_nadir_point()
try:
    pf = problem.pareto_front()
except:
    pf = problem.get_pareto_front()
hv = HV(ref_point=nadir_point)
igd = IGD(pf=pf)

algo_to_hv = {}
algo_to_igd = {}

for algo_name, algo in name_to_algo.items():
    hv_all = []
    igd_all = []

    def callback(algorithm):
        x = algorithm.pop.get('X')
        y = algorithm.pop.get('F')
        hv_value = hv(y)
        igd_value = igd(y)

        hv_all.append(hv_value)
        igd_all.append(igd_value)
    
    algo.callback = callback
    res = minimize(
        algorithm=algo,
        problem=problem,
        termination=('n_gen', 800)
    )

    x = res.pop.get('X')
    y = res.pop.get('F')
    plt.figure(figsize=(8, 8))
    plt.scatter(pf[:, 0], pf[:, 1], color='red')
    plt.scatter(y[:, 0], y[:, 1], color='blue')
    plt.savefig(f'Performance of {algo_name} on {env_name}.png')

    algo_to_hv[algo_name] = hv_all
    algo_to_igd[algo_name] = igd_all

color_idx = 2
fig, axs = plt.subplots(2, 1, figsize=(8, 16))
ax0 = axs[0]
ax0.set_title(f'HV on {env_name}')
ax0.set_ylabel('Hypervolume')
ax0.set_xlabel('# generations')
for algo_name, hv_all in algo_to_hv.items():
    ax0.plot(range(len(hv_all)), hv_all, color=list(color_set.values())[color_idx], label=algo_name)
    color_idx = (color_idx + 1) % len(color_set)
ax0.legend()

color_idx = 2
ax1 = axs[1]
ax1.set_title(f'IGD on {env_name}')
ax1.set_ylabel('Inverse Generation Distance')
ax1.set_xlabel('# generations')
for algo_name, igd_all in algo_to_igd.items():
    ax1.plot(range(len(igd_all)), igd_all, color=list(color_set.values())[color_idx], label=algo_name)
    color_idx = (color_idx + 1) % len(color_set)
ax1.legend()

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(f'Indicators of {env_name}')