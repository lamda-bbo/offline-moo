import os, sys
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_path)

import numpy as np
import pandas as pd
import torch
import argparse
import time
import pickle
from datetime import datetime

from off_moo_bench.problem import get_problem
from off_moo_bench.evaluation.metrics import hv, igd
from algorithm.multi_head_nn import (
    MultiHeadModel, 
    train_model, 
    grad_norm_train_model, 
    pcgrad_train_model,
    MultiHeadSurrogateProblem
)
from algorithm.mo_solver.moea_solver import MOEASolver
from algorithm.mo_solver.mobo_solver import MOBOSolver
from algorithm.mo_solver.mobo_seq_perm_solver import MOBOSolver_seq_perm
from algorithm.mo_solver.callback import RecordCallback
from utils import normalize_x, normalize_y, denormalize_x, denormalize_y
from utils import read_data, set_seed, get_config_path, load_config, read_filter_data, get_N_nondominated_index, get_quantile_solutions
from pymoo.algorithms.moo.nsga2 import NSGA2, NonDominatedSorting
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='re21')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train-model-seed', type=int, default=1)
parser.add_argument('--num-solutions', type=int, default=256)
parser.add_argument('--normalize-x', action='store_true')
parser.add_argument('--normalize-y', action='store_true')
parser.add_argument('--retrain-model', action='store_true')
parser.add_argument('--model-name', type=str, default='multi_head_nn')
parser.add_argument('--filter-type', type=str, default='best')
parser.add_argument('--train-mode', type=str, default='none')
parser.add_argument('--train-data-mode', type=str, default='none')
parser.add_argument('--reweight-mode', type=str, default='none')
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--sigmoid-quantile', type=float, default=0.25)
parser.add_argument('--only-train-model', action='store_true')
parser.add_argument('--mo-solver', type=str, default='nsga2')
parser.add_argument('--df-name', type=str, default='1-final-best-hv.csv')
run_args = parser.parse_args()

normalize_str = 'final-'

normalize_str += f'normalize-{run_args.filter_type}' if run_args.normalize_y \
                                    else f'no_normalize-{run_args.filter_type}'
normalize_str = f'{normalize_str}-{run_args.train_data_mode}' if run_args.train_data_mode != 'none'\
                        else normalize_str
normalize_str = f'{normalize_str}-{run_args.train_mode}' if run_args.train_mode != 'none'\
                        else normalize_str
normalize_str = f'{normalize_str}-{run_args.reweight_mode}' if run_args.reweight_mode != 'none'\
                        else normalize_str

results_dir = os.path.join(base_path, 'results', run_args.env_name, 'multi_head',
                                normalize_str, f'{run_args.seed}')
print(results_dir)
os.makedirs(results_dir, exist_ok=True)

time_file = f'{run_args.env_name}-MultiHead-{run_args.train_mode}-{run_args.train_data_mode}.txt'
time_file = os.path.join(base_path, 'time_record', time_file)

def run(args):
    t1 = datetime.now()
    with open(time_file, 'a+') as f:
        f.write('Run begin!\nStart time: ' + str(t1))

    device = args.device
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

    model = MultiHeadModel(
        n_dim = problem.n_dim if not args.discrete else input_size,
        n_obj = problem.n_obj,
        args = args,
        hidden_size = [2048, 2048]
    )
    for obj in range(problem.n_obj):
        model.create_head(obj)
    model = model.to(device)
    
    from algorithm.multi_head_nn.trainer import load_model

    loaded_model = False

    def check_model_path(model):
        if not os.path.exists(model.feature_extractor.save_path):
            return False
        for head in model.obj2head.values():
            if not os.path.exists(head.save_path):
                return False
        return True
        
    if not args.retrain_model and check_model_path(model):
        model.feature_extractor = load_model(model.feature_extractor)
        model.feature_extractor = model.feature_extractor.to(model.device)
        
        for obj in model.obj2head.keys():
            model.obj2head[obj] = load_model(model.obj2head[obj])
            model.obj2head[obj] = model.obj2head[obj].to(model.device)
        
        loaded_model = True

    if not loaded_model:

        def get_pareto_efficient_data(ratio):
            res_size = int(data_size * ratio)
            indices_to_keep = get_N_nondominated_index(y_np, res_size, is_all_data=True)
            return x_np[indices_to_keep], \
                y_np[indices_to_keep]
        
        if args.train_data_mode.startswith('onlybest'):
            x_np, y_np = get_pareto_efficient_data(0.2)

        t2 = datetime.now()
        with open(time_file, 'a') as f:
            f.write('\n\nData preprocess Ended!\n' + f'Now time: {t2}' + '\n' + f'Time for data preprocessing: {t2 - t1}')

        t3 = datetime.now()
        with open(time_file, 'a') as f:
            f.write('\n\nBegin to train model!\n' + f'Now time: {t3}')
        
        if args.train_mode == 'grad_norm':
            grad_norm_train_model(model, x_np, y_np, x_test_np, y_test_np, device, 
                    retrain=args.retrain_model,
                    lr_model = args.lr,
                    layer = model.feature_extractor.layers[-1],
                    lr_decay = args.lr_decay,
                    n_epochs = args.n_epochs,
                    batch_size = args.batch_size,
                    split_ratio = args.split_ratio)
        elif args.train_mode == 'pcgrad':
            pcgrad_train_model(model, x_np, y_np, x_test_np, y_test_np, device, 
                        retrain=args.retrain_model,
                        lr = args.lr,
                        lr_decay = args.lr_decay,
                        n_epochs = args.n_epochs,
                        batch_size = args.batch_size,
                        split_ratio = args.split_ratio)
        else:
            train_model(model, x_np, y_np, x_test_np, y_test_np, device, 
                        retrain=args.retrain_model,
                        lr = args.lr,
                        lr_decay = args.lr_decay,
                        n_epochs = args.n_epochs,
                        batch_size = args.batch_size,
                        split_ratio = args.split_ratio)
            
        if args.only_train_model:
            return
        
        t4 = datetime.now()
        with open(time_file, 'a') as f:
            f.write('\n\nModel training Ended!\n' + f'Now time: {t4}' + '\n' + f'Time for Model Training: {t4 - t3}')

    set_seed(args.seed)

    surrogate_problem = MultiHeadSurrogateProblem(
        n_var = problem.n_dim if not args.discrete else input_size,
        n_obj = problem.n_obj,
        model = model,
        device = device,
    )

    if args.env_name in ['regex', 'rfp']:
        surrogate_problem.x_to_query_batches = problem.task_instance.x_to_query_batches
        surrogate_problem.query_batches_to_x = problem.task_instance.query_batches_to_x
        surrogate_problem.candidate_pool = problem.task_instance.candidate_pool
        surrogate_problem.op_types = problem.task_instance.op_types

    elif args.env_name.startswith(('c10mop', 'in1kmop')):
        surrogate_problem.xl = problem.xl
        surrogate_problem.xu = problem.xu
    
    callback = RecordCallback(
        real_problem=problem,
        surrogate_problem=surrogate_problem,
        args=args,
        iters_to_record=3
    )

    y_all = []
    y_su_all = []

    def y_callback(algorithm):
        x = algorithm.pop.get('X')
        y = problem.evaluate(x)
        y_all.append(y.reshape((1, -1, problem.n_obj)))
        y_su_all.append(algorithm.pop.get('F').reshape((1, -1, problem.n_obj)))
        with open(f'{args.env_name}-{args.train_data_mode}-y.pkl', 'wb+') as f:
            pickle.dump(y_all, f)
        with open(f'{args.env_name}-{args.train_data_mode}-y-srg.pkl', 'wb+') as f:
            pickle.dump(y_su_all, f)
    
    if run_args.mo_solver == 'nsga2':
        if run_args.env_name.startswith('motsp') or run_args.env_name.startswith('mocvrp') or run_args.env_name.startswith('mokp'):
            from pymoo.operators.sampling.rnd import PermutationRandomSampling
            from pymoo.operators.crossover.ox import OrderCrossover
            from pymoo.operators.mutation.inversion import InversionMutation
            from collecter import StartFromZeroRepair

            solver = MOEASolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, 
                            algo=NSGA2, pop_size=args.num_solutions,
                        callback=callback,
                        mutation=InversionMutation(),
                        crossover=OrderCrossover(),
                        repair=StartFromZeroRepair() if not run_args.env_name.startswith('mo_kp') else None,
                        eliminate_duplicates=True,)
            
        elif run_args.env_name in ['regex', 'rfp']:
            from off_moo_bench.problem.lambo.lambo.optimizers.sampler import CandidateSampler
            from pymoo.factory import get_crossover
            from off_moo_bench.problem.lambo.lambo.optimizers.mutation import LocalMutation
            from off_moo_bench.problem.lambo.lambo.utils import ResidueTokenizer
            solver = MOEASolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, 
                            algo=NSGA2, 
                        callback=callback if run_args.env_name == 'regex' else None,
                        crossover=get_crossover(name='int_sbx', prob=0., eta=16),
                        mutation=LocalMutation(prob=1., eta=16, safe_mut=False, tokenizer=ResidueTokenizer()),
                        eliminate_duplicates=True)
            
        elif run_args.env_name.startswith(('c10mop', 'in1kmop')):
            from off_moo_bench.problem.mo_nas import get_genetic_operator
            _, crossover, mutation = get_genetic_operator()
            solver = MOEASolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, 
                            algo=NSGA2, pop_size=args.num_solutions,
                        callback=callback,
                        crossover=crossover,
                        mutation=mutation,
                        eliminate_duplicates=True)

        else:
            solver = MOEASolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, algo=NSGA2, pop_size=args.num_solutions,
                        callback=callback if run_args.env_name != 'molecule' else None)
         
        t5 = datetime.now()
        with open(time_file, 'a') as f:
            f.write('\n\nBegin to search for the solutions!\n' + f'Now time: {t5}')

        res = solver.solve(problem=surrogate_problem, X=x_np, Y=y_np) 

        t6 = datetime.now()
        with open(time_file, 'a') as f:
            f.write('\n\nSolutions founded!\n' + f'Now time: {t6}' + '\n' + f'Time for Solutions searching: {t6 - t5}')

    elif run_args.mo_solver == 'nsga3':
        try:
            from pymoo.util.ref_dirs import get_reference_directions
        except:
            from pymoo.factory import get_reference_directions

        if run_args.env_name in ['mo_tsp', 'mo_kp', 'mo_cvrp']:
            from pymoo.operators.sampling.rnd import PermutationRandomSampling
            from pymoo.operators.crossover.ox import OrderCrossover
            from pymoo.operators.mutation.inversion import InversionMutation
            from collecter import StartFromZeroRepair

            solver = MOEASolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, 
                            algo=NSGA3, 
                        callback=callback,
                        mutation=InversionMutation(),
                        crossover=OrderCrossover(),
                        repair=StartFromZeroRepair() if run_args.env_name != 'mo_kp' else None,
                        eliminate_duplicates=True,
                        ref_dirs=get_reference_directions('uniform', problem.n_obj, n_partitions=args.num_solutions))
            
        elif run_args.env_name in ['regex', 'rfp']:
            from off_moo_bench.problem.lambo.lambo.optimizers.sampler import CandidateSampler
            from pymoo.factory import get_crossover
            from off_moo_bench.problem.lambo.lambo.optimizers.mutation import LocalMutation
            from off_moo_bench.problem.lambo.lambo.utils import ResidueTokenizer
            solver = MOEASolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, 
                            algo=NSGA3, 
                        callback=callback if run_args.env_name == 'regex' else None,
                        crossover=get_crossover(name='int_sbx', prob=0., eta=16),
                        mutation=LocalMutation(prob=1., eta=16, safe_mut=False, tokenizer=ResidueTokenizer()),
                        eliminate_duplicates=True,
                        ref_dirs=get_reference_directions('uniform', problem.n_obj, n_partitions=args.num_solutions))

        else:
            solver = MOEASolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, algo=NSGA3, 
                        callback=callback if run_args.env_name != 'molecule' else None,
                        ref_dirs=get_reference_directions('uniform', problem.n_obj, n_partitions=args.num_solutions))
         
        res = solver.solve(problem=surrogate_problem, X=x_np, Y=y_np) 
    
    elif run_args.mo_solver == 'moead':
        try:
            from pymoo.util.ref_dirs import get_reference_directions
        except:
            from pymoo.factory import get_reference_directions

        if run_args.env_name in ['mo_tsp', 'mo_kp', 'mo_cvrp']:
            from pymoo.operators.sampling.rnd import PermutationRandomSampling
            from pymoo.operators.crossover.ox import OrderCrossover
            from pymoo.operators.mutation.inversion import InversionMutation
            from collecter import StartFromZeroRepair

            solver = MOEASolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, 
                            algo=MOEAD,
                        callback=callback,
                        mutation=InversionMutation(),
                        crossover=OrderCrossover(),
                        repair=StartFromZeroRepair() if run_args.env_name != 'mo_kp' else None,
                        eliminate_duplicates=True,
                        ref_dirs=get_reference_directions('uniform', problem.n_obj, n_partitions=args.num_solutions))
            
        elif run_args.env_name in ['regex', 'rfp']:
            from off_moo_bench.problem.lambo.lambo.optimizers.sampler import CandidateSampler
            from pymoo.factory import get_crossover
            from off_moo_bench.problem.lambo.lambo.optimizers.mutation import LocalMutation
            from off_moo_bench.problem.lambo.lambo.utils import ResidueTokenizer
            solver = MOEASolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, 
                            algo=MOEAD, 
                        callback=callback if run_args.env_name == 'regex' else None,
                        crossover=get_crossover(name='int_sbx', prob=0., eta=16),
                        mutation=LocalMutation(prob=1., eta=16, safe_mut=False, tokenizer=ResidueTokenizer()),
                        ref_dirs=get_reference_directions('uniform', problem.n_obj, n_partitions=args.num_solutions))

        else:
            solver = MOEASolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, algo=MOEAD,
                        callback=callback if run_args.env_name != 'molecule' else None, ref_dirs=get_reference_directions('uniform', problem.n_obj, n_partitions=args.num_solutions))
         
        res = solver.solve(problem=surrogate_problem, X=x_np, Y=y_np) 

    elif run_args.mo_solver == 'mobo':
        if run_args.env_name in ['mo_tsp', 'mo_kp', 'mo_cvrp']:
            solver = MOBOSolver_seq_perm(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, var_type='permutation')
        elif run_args.env_name in ['rfp', 'regex']:
            solver = MOBOSolver_seq_perm(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions, var_type='sequence', xl=problem.xl, xu=problem.xu)
        else:
            solver = MOBOSolver(n_gen=52, pop_init_method='nds', batch_size=args.num_solutions)
        res = solver.solve(problem=surrogate_problem, X=x_np, Y=y_np)

    else:
        raise NotImplementedError
    
    # y_all = np.concatenate(y_all, axis=0)
    # np.save(arr=y_all, file=f'{args.env_name}-{args.train_data_mode}-y.npy')
    # import pickle 
    # with open(f'{args.env_name}-{args.train_data_mode}-y.pkl', 'wb+') as f:
    #     pickle.dump(y_all, f)
    # with open(f'{args.env_name}-{args.train_data_mode}-y-srg.pkl', 'wb+') as f:
    #     pickle.dump(y_su_all, f)
    # assert 0, len(y_all)

    t7 = datetime.now()
    with open(time_file, 'a') as f:
        f.write('\n\nBegin to evaluate the solutions!\n' + f'Now time: {t7}')

    res_x = res['x']
    
    if args.normalize_x and args.env_name not in ['dtlz1', 'dtlz2', 'dtlz3',
            'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7', 'vlmop2', "kursawe", "omnitest", 
             "sympart", "zdt1", "zdt2", "zdt3", "zdt4", "zdt6", 
            'vlmop3', 're21', 're23', 're33', 're34', 're37', 're42', 're61']:
        res_x = denormalize_x(args, res_x)
    if args.discrete:
        from utils import to_integers
        res_x = res_x.reshape((-1,) + input_shape)
        res_x = to_integers(res_x)

    res_y = problem.evaluate(res_x)[1] if args.env_name == 'qm9' else problem.evaluate(res_x)
    
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

    t8 = datetime.now()
    with open(time_file, 'a') as f:
        f.write('\n\nSolutions evaluated!\n' + f'Now time: {t8}' + '\n' + f'Time for Solutions evaluation: {t8 - t7}')

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
        # pareto_front = normalize_y(args.env_name, pareto_front)
    
    
    indices_select = get_N_nondominated_index(y_np, args.num_solutions, is_all_data=True)
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
    entry_desc = 'MultiHead-Model'
    if args.reweight_mode == 'sigmoid':
        entry_desc += f'-sigmoid-{args.sigmoid_quantile}'
    elif args.train_mode != 'none':
        entry_desc += f'-{args.train_mode}'
    elif args.mo_solver == 'mobo':
        entry_desc += f'-mobo'
    elif args.train_data_mode != 'none':
        entry_desc += f'-{args.train_data_mode}-20%'
    elif args.num_solutions != 256:
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
    set_seed(run_args.train_model_seed)
    config_path = get_config_path(run_args.model_name)
    config_args = load_config(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_args['device'] = device
    config_args['results_dir'] = results_dir
    run_args.__dict__.update(config_args)
    run(args=run_args)