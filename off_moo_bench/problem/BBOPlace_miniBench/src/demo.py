import argparse

import cma
import numpy as np
from bboplace_utils.args_parser import parse_args
from evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--pop_size", type=int, default=20)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--placer", type=str, choices=["sp", "gg"], default="gg")
parser.add_argument("--benchmark", type=str, default="ispd2005/adaptec1")
parser.add_argument("--max_evals", type=int, default=1000)
args = parser.parse_args()
args = parse_args(args)

evaluator = Evaluator(args=args)

dim = evaluator.n_dim
xl = evaluator.xl.tolist()
xu = evaluator.xu.tolist()
assert len(xl) == len(xu) == dim

x0 = np.random.uniform(low=xl, high=xu, size=dim)

# Initialize CMA-ES
cmaes = cma.CMAEvolutionStrategy(
    x0, args.sigma, {"popsize": args.pop_size, "bounds": [xl, xu]}
)

# Run optimization
while cmaes.result.evaluations < args.max_evals:
    solutions = cmaes.ask()
    res, macro_pos_lst = evaluator.evaluate(solutions)
    fitness_values = res["hpwl"]
    cmaes.tell(solutions, fitness_values)
    print(f"Generation {cmaes.countiter}: Best fitness = {min(fitness_values):.6f}")

# Print results
print("\nOptimization finished")
print(f"Best solution found: {cmaes.result.xbest}")
print(f"Best fitness: {cmaes.result.fbest}")
print(f"Number of evaluations: {cmaes.result.evaluations}")
