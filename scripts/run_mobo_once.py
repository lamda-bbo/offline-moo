import os, sys 
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_path)

import numpy as np
import torch
import argparse
import pandas as pd

from off_moo_bench.problem import get_problem
from off_moo_bench.evaluation.metrics import hv, igd 
from utils import (
    normalize_x, normalize_y,
    denormalize_x, denormalize_y
)
from utils import (
    read_data, read_filter_data, get_N_nondominated_index,
    set_seed, get_config_path, load_config, get_quantile_solutions
)
from pymoo.algorithms.moo.nsga2 import NSGA2, NonDominatedSorting

from algorithm.mobo_once import MOBO_Once
from algorithm.mobo_permutation import MOBO_Permutation
from algorithm.mobo_sequence import MOBO_Sequence
from algorithm.mobo_jes import MOBO_JES_Once
from algorithm.mobo_parego import MOBO_ParEGO_Once
from algorithm.mobo_jes_perm import MOBO_JES_Permutation
from algorithm.mobo_parego_permutation import MOBO_ParEGO_Permutation
from algorithm.mobo_parego_sequence import MOBO_ParEGO_Sequence

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='re21')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num-solutions', type=int, default=256)
parser.add_argument('--normalize-x', action='store_true')
parser.add_argument('--normalize-y', action='store_true')
parser.add_argument('--retrain-model', action='store_true')
parser.add_argument('--train-mode', type=str, default='none')
parser.add_argument('--reweight-mode', type=str, default='none')
parser.add_argument('--model-name', type=str, default='mobo_once')
parser.add_argument('--filter-type', type=str, default='best')
parser.add_argument('--train-gp-data-size', type=int, default=256)
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--permutation', action='store_true')
parser.add_argument('--sequence', action='store_true')
parser.add_argument('--df-name', type=str, default=f'final-best_hv.csv')
run_args = parser.parse_args()

normalize_str = 'final-'

normalize_str += f'normalize-{run_args.filter_type}' if run_args.normalize_y \
                                    else f'no_normalize-{run_args.filter_type}'
normalize_str = f'{normalize_str}-{run_args.train_mode}' if run_args.train_mode != 'none'\
                        else normalize_str
normalize_str = f'{normalize_str}-{run_args.reweight_mode}' if run_args.reweight_mode != 'none'\
                        else normalize_str
results_dir = os.path.join(base_path, 'results', run_args.env_name, 'mobo_once',
                                normalize_str, f'{run_args.seed}')

if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    problem = get_problem(args.env_name)
    x_np, y_np, _ = read_data(env_name=args.env_name, filter_type=args.filter_type, return_rank=False)
    x_test_np, y_test_np, _ = read_filter_data(env_name=args.env_name, filter_type=args.filter_type, return_rank=False)


    if args.discrete:
        from utils import to_logits
        x_np = to_logits(x_np, args)
        x_test_np = to_logits(x_test_np, args)

    input_size = np.prod(x_np.shape[1:])
    input_shape = x_np.shape[1:]
    args.__dict__.update({'input_size': input_size, 'input_shape': input_shape})
    x_np = x_np.reshape(-1, input_size)
    x_test_np = x_test_np.reshape(-1, input_size)
    data_size = len(x_np)

    if args.normalize_x:
        x_np = normalize_x(args, x_np)
        x_test_np = normalize_x(args, x_test_np)
    if args.normalize_y:
        y_np = normalize_y(args, y_np)
        y_test_np = normalize_y(args, y_test_np)
    
    if args.permutation:
        if args.train_mode == 'jes':
            mobo_once_solver = MOBO_JES_Permutation(
                X_init = torch.Tensor(x_np).to(device=device, dtype=torch.double),
                Y_init = torch.Tensor(y_np).to(device=device, dtype=torch.double),
                train_gp_data_size = args.train_gp_data_size,
                output_size = args.num_solutions,
                env_name=args.env_name
            )
        elif args.train_mode == 'parego':
            mobo_once_solver = MOBO_ParEGO_Permutation(
                X_init = torch.Tensor(x_np).to(device=device, dtype=torch.double),
                Y_init = torch.Tensor(y_np).to(device=device, dtype=torch.double),
                train_gp_data_size = args.train_gp_data_size,
                output_size = args.num_solutions,
                env_name=args.env_name
            )
        else:
            mobo_once_solver = MOBO_Permutation(
                X_init = torch.Tensor(x_np).to(device=device, dtype=torch.double),
                Y_init = torch.Tensor(y_np).to(device=device, dtype=torch.double),
                train_gp_data_size = args.train_gp_data_size,
                output_size = args.num_solutions,
                env_name=args.env_name
            )
    elif args.sequence:
        if args.train_mode == 'parego':
            mobo_once_solver = MOBO_ParEGO_Sequence(
                X_init = torch.Tensor(x_np).to(device=device, dtype=torch.double),
                Y_init = torch.Tensor(y_np).to(device=device, dtype=torch.double),
                xl = problem.xl, 
                xu = problem.xu,
                train_gp_data_size = args.train_gp_data_size,
                output_size = args.num_solutions,
            )
        else:
            mobo_once_solver = MOBO_Sequence(
                X_init = torch.Tensor(x_np).to(device=device, dtype=torch.double),
                Y_init = torch.Tensor(y_np).to(device=device, dtype=torch.double),
                xl = problem.xl, 
                xu = problem.xu,
                train_gp_data_size = args.train_gp_data_size,
                output_size = args.num_solutions,
            )
    else:
        if args.train_mode == 'parego':
            MOBO_Solver = MOBO_ParEGO_Once
        elif args.train_mode == 'jes':
            MOBO_Solver = MOBO_JES_Once
        else:
            MOBO_Solver = MOBO_Once
            
        bounds_ = torch.zeros((2,input_size))
        bounds_[1] = 1.0

        mobo_once_solver = MOBO_Solver(
            X_init=torch.Tensor(x_np).to(device),
            Y_init=torch.Tensor(y_np).to(device),
            ref_point=torch.Tensor(1.1 * problem.get_nadir_point()).to(device),
            bounds=bounds_,
            train_gp_data_size=args.train_gp_data_size,
            output_size=args.num_solutions,
            negate=True
        )

    import botorch
    with botorch.settings.debug(True):
        res_x = mobo_once_solver.run()
        if isinstance(res_x, torch.Tensor):
            res_x = res_x.detach().cpu().numpy()

    if args.normalize_x and args.env_name not in ['dtlz1', 'dtlz2', 'dtlz3',
            'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7', 'vlmop2', "kursawe", "omnitest", 
             "sympart", "zdt1", "zdt2", "zdt3", "zdt4", "zdt6", 'mo_kp', 'mo_tsp', 'mo_cvrp',
            'vlmop3', 're21', 're23', 're33', 're34', 're37', 're42', 're61']:
        res_x = denormalize_x(args, res_x)
    if args.discrete:
        from utils import to_integers
        res_x = res_x.reshape((-1,) + input_shape)
        res_x = to_integers(res_x)

    print(res_x)
    # np.save(arr=res_x, file=f'mobo_once_{args.train_gp_data_size}_{args.env_name}_res_x.npy')
        
    res_y = problem.evaluate(res_x)[1] if args.env_name == 'qm9' else problem.evaluate(res_x)
    try:
        np.save(arr=res_y, file=os.path.join(results_dir, 'y_predict.npy'))
        # res_y = res_y[NonDominatedSorting().do(res_y)[0]]

        if args.env_name == 'molecule':
            index_to_remove = np.all(res_y == [1., 1.], axis=1)
            res_y = res_y[~index_to_remove]
        elif args.env_name.startswith(('c10mop', 'in1kmop')):
            visible_masks = np.ones(len(res_y))
            visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
            visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
            res_x = res_x[np.where(visible_masks == 1)[0]]
            res_y = res_y[np.where(visible_masks == 1)[0]]

        res_y_50_percent = get_quantile_solutions(res_y, 0.5)

        np.save(arr=res_y, file=os.path.join(results_dir, 'y_predict.npy'))
        # res_y = res_y[NonDominatedSorting().do(res_y)[0]]

        nadir_point = 2 * problem.get_nadir_point() if args.env_name in ['mo_hopper_v2', 'mo_swimmer_v2'] \
            else 1.1 * problem.get_nadir_point()
        # pareto_front = problem.get_pareto_front()
        if args.normalize_y:
            res_y = normalize_y(args, res_y)
            nadir_point = normalize_y(args, nadir_point)
            res_y_50_percent = normalize_y(args, res_y_50_percent)
            # res_y_50_percent = normalize_y(args, res_y_50_percent)
            # pareto_front = normalize_y(args.env_name, pareto_front)
    except:
        res_y = res_y.T
        np.save(arr=res_y, file=os.path.join(results_dir, 'y_predict.npy'))
        # res_y = res_y[NonDominatedSorting().do(res_y)[0]]

        if args.env_name == 'molecule':
            index_to_remove = np.all(res_y == [1., 1.], axis=1)
            res_y = res_y[~index_to_remove]
        elif args.env_name.startswith(('c10mop', 'in1kmop')):
            visible_masks = np.ones(len(res_y))
            visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
            visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
            res_x = res_x[np.where(visible_masks == 1)[0]]
            res_y = res_y[np.where(visible_masks == 1)[0]]

        res_y_50_percent = get_quantile_solutions(res_y, 0.5)

        np.save(arr=res_y, file=os.path.join(results_dir, 'y_predict.npy'))
        # res_y = res_y[NonDominatedSorting().do(res_y)[0]]

        nadir_point = 2 * problem.get_nadir_point() if args.env_name in ['mo_hopper_v2', 'mo_swimmer_v2'] \
            else 1.1 * problem.get_nadir_point()
        # pareto_front = problem.get_pareto_front()
        if args.normalize_y:
            res_y = normalize_y(args, res_y)
            nadir_point = normalize_y(args, nadir_point)
            res_y_50_percent = normalize_y(args, res_y_50_percent)
            # res_y_50_percent = normalize_y(args, res_y_50_percent)
            # pareto_front = normalize_y(args.env_name, pareto_front)
    
    
    indices_select = get_N_nondominated_index(y_np, args.num_solutions)
    d_best = y_np[indices_select]

    np.save(arr=denormalize_y(args, d_best), file=os.path.join(results_dir, 'pop_init.npy'))
    
    from plot import plot_y
    hv_value = hv(nadir_point, res_y)
    # igd_value = igd(pareto_front, res_y)
    print(hv_value)
    hv_50percent_value = hv(nadir_point, res_y_50_percent)
    print(hv_50percent_value)
    d_best_hv = hv(nadir_point, d_best)
    print(d_best_hv)
    # print(igd_value)
    
    plot_y(res_y, save_dir=results_dir, nadir_point=nadir_point, d_best=d_best)
    with open(os.path.join(results_dir, 'results.txt'), 'w+') as f:
        f.write(f"hv = {hv_value}" + '\n' + f"D(best) hv = {d_best_hv}")

    # args.df_name = '1-test-hv.csv'

    if not os.path.exists(os.path.join(base_path, args.df_name)):
        hv_df = pd.DataFrame()
    else:
        hv_df = pd.read_csv(args.df_name, header=0, index_col=0)
    
    hv_df.loc['D(best)', args.env_name] = d_best_hv
    entry_desc = 'MOBO'
    if args.reweight_mode == 'sigmoid':
        entry_desc += f'-sigmoid-{args.sigmoid_quantile}'
    if args.train_mode != 'none':
        entry_desc += f'-{args.train_mode}'
    if args.train_gp_data_size != 256:
        entry_desc += f'-{args.train_gp_data_size}'
    if args.num_solutions != 256:
        entry_desc += f'-{args.num_solutions}'
    else:
        pass
    hv_df.loc[entry_desc, args.env_name] = hv_value
    hv_df.loc[f'{entry_desc}-50percentile', args.env_name] = hv_50percent_value

    # if args.reweight_mode == 'sigmoid':
    #     hv_df.loc[f'{args.model_name}-sigmoid-{args.sigmoid_quantile}', args.env_name] = hv_value
    # elif args.train_mode != 'none':
    #     hv_df.loc[f'{args.model_name}-{args.train_mode}', args.env_name] = hv_value
    # elif args.mo_solver == 'mobo':
    #     hv_df.loc[f'{args.model_name}-mobo', args.env_name] = hv_value
    # elif args.train_data_mode != 'none':
    #     hv_df.loc[f'{args.model_name}-{args.train_data_mode}-20%', args.env_name] = hv_value
    # else:
    #     hv_df.loc[args.model_name, args.env_name] = hv_value
    hv_df.to_csv(args.df_name, index=True, mode='w')

if __name__ == "__main__":
    set_seed(run_args.seed)
    run_args.__dict__.update({'results_dir': results_dir})
    run(args=run_args)

