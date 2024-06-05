import os, sys
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_path)

import torch
import numpy as np
import argparse
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from collecter import StartFromZeroRepair
from pymoo.optimize import minimize
from off_moo_bench.problem import get_problem
from collecter import AmateurRankAndCrowdSurvival, MoleculeEvaluator, molecule_callback

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='dtlz2')
parser.add_argument('--data-size', type=int, default=10000)
parser.add_argument('--collect-type', type=str, default='amateur', 
                    help="Data collect methods with choice in [amateur, expert, random]")
parser.add_argument('--perturb-proba', type=float, default=0.35)
parser.add_argument('--filter-ratio', type=float, default=0.4)
parser.add_argument('--filter-type', type=str, default='best',
                    help="Data filter methods with choice in [best, random_filter, layer_and_random, best_and_random]")
collect_args = parser.parse_args()

env_names = [ "vlmop1", "vlmop2", "vlmop3", "dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", 
             "dtlz6", "dtlz7",  "omnitest", "kursawe",
              "zdt1", "zdt2", "zdt3", "zdt4", "zdt6", "re21", "re22", 
             "re23", "re24", "re25", "re31", "re32", "re33",
               "re34", "re35", "re36" "re37", "re41", "re42", "re61", "re91", "nb201_test",  "mo_tsp_3obj",
             "mo_swimmer_v2", "mo_hopper_v2", "mo_tsp", "mo_kp", "mo_cvrp", "molecule", "regex", "rfp", "zinc",
             "mo_tsp_500", "mo_tsp_100", "mo_tsp_50", "mo_tsp_20", "mo_kp_200", "mo_kp_100", "mo_kp_50",
               "mo_cvrp_100", "mo_cvrp_50", "mo_cvrp_20", "motsp3obj_500", "mo_tsp_3obj_100", 
               "mo_tsp_3obj_50", "mo_tsp_3obj_20", "c10mop1", "c10mop2", "c10mop3", "c10mop4", "c10mop5", "c10mop6", "c10mop7", 
             "c10mop8", "c10mop9", "in1kmop1", "in1kmop2", "in1kmop3", "in1kmop4", "in1kmop5",
             "in1kmop6", "in1kmop7", "in1kmop8", "in1kmop9", "nb201_test"]

synthetic_envs = ["vlmop1", "vlmop2", "vlmop3", "dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", 
             "dtlz6", "dtlz7",  "omnitest", "kursawe",
              "zdt1", "zdt2", "zdt3", "zdt4", "zdt6", "re21", "re22", 
             "re23", "re24", "re25", "re31", "re32", "re33",
               "re34", "re35", "re36" "re37", "re41", "re42", "re61", "re91", "molecule", "regex", "rfp", "zinc"]

moco_envs = ["mo_tsp_500", "mo_tsp_100", "mo_tsp_50", "mo_tsp_20", "mo_kp_200", "mo_kp_100", "mo_kp_50", 
             "mo_cvrp_100", "mo_cvrp_50", "mo_cvrp_20", "motsp3obj_500", "mo_tsp_3obj_100", "mo_tsp_3obj_50",
               "mo_tsp_3obj_20"]

monas_env = ["c10mop1", "c10mop2", "c10mop3", "c10mop4", "c10mop5", "c10mop6", "c10mop7", 
             "c10mop8", "c10mop9", "in1kmop1", "in1kmop2", "in1kmop3", "in1kmop4", "in1kmop5",
             "in1kmop6", "in1kmop7", "in1kmop8", "in1kmop9", "nb201_test"]

env_name_n_gens = {
    "vlmop1": 30,
    "vlmop2": 30, 
    "vlmop3": 30, 
    "dtlz1": 30,
    "dtlz2":  30, 
    "dtlz3": 30,
    "dtlz4":  30, 
    "dtlz5": 30,
    "dtlz6":  30, 
    "dtlz7": 30,
    "kursawe": 30,
    "omnitest": 30, 
    "sympart": 30,
    "zdt1": 30, 
    "zdt2": 40, 
    "zdt3": 30, 
    "zdt4": 600, 
    "zdt6": 600, 
    "re21": 30,
    "re22": 30,
    "re23": 30,
    "re24": 30,
    "re25": 30,
    "re31": 30,
    "re32": 30,
    "re33": 30, 
    "re34": 100,
    "re35": 30,
    "re36": 30, 
    "re41": 30,
    "re37": 30, 
    "re42": 30, 
    "re61": 30, 
    're91': 30,
    "mo_tsp_500": 80000,
    "mo_tsp_100": 16000,
    "mo_tsp_50": 8000,
    "mo_tsp_20": 3200,
    "motsp3obj_500": 40000,
    "mo_tsp_3obj_100": 16000,
    "mo_tsp_3obj_50": 8000,
    "mo_tsp_3obj_20": 3200,
    "mo_cvrp_100": 8000,
    "mo_cvrp_50": 4000,
    "mo_cvrp_20": 1600,
    "mo_kp_200": 800,
    "mo_kp_100": 400,
    "mo_kp_50": 200,
    "molecule": 20,
    'regex': 100,
    'rfp': 100,
    'zinc': 100,
}

env_name_perturb_proba = {
    "vlmop1": 0.45,
    "vlmop2": 0.45, 
    "vlmop3": 0.45, 
    "dtlz1": 0.45,
    "dtlz2":  0.45, 
    "dtlz3": 0.45,
    "dtlz4":  0.45, 
    "dtlz5": 0.45,
    "dtlz6":  0.45, 
    "dtlz7": 0.45,
    "kursawe": 0.45,
    "omnitest": 0.45, 
    "sympart": 0.45,
    "zdt1": 0.45, 
    "zdt2": 0.45, 
    "zdt3": 0.45, 
    "zdt4": 0.25, 
    "zdt6": 0.2, 
    "re21": 0.45,
    "re22": 0.45,
    "re23": 0.45,
    "re24": 0.45,
    "re25": 0.45,
    "re31": 0.45,
    "re32": 0.45, 
    "re33": 0.45, 
    "re34": 0.3,
    "re35": 0.45,
    "re36": 0.45,
    "re37": 0.45,
    "re41": 0.45, 
    "re42": 0.45, 
    "re61": 0.45, 
    're91': 0.45,
    "mo_tsp_500": 0,
    "mo_tsp_100": 0,
    "mo_tsp_50": 0,
    "mo_tsp_20": 0,
    "motsp3obj_500": 0,
    "mo_tsp_3obj_100": 0,
    "mo_tsp_3obj_50": 0,
    "mo_tsp_3obj_20": 0,
    "mo_kp_200": 0.15,
    "mo_kp_100": 0.15,
    "mo_kp_50": 0.15,
    "mo_cvrp_100": 0,
    "mo_cvrp_50": 0,
    "mo_cvrp_20": 0,
    "molecule": 0.45,
    'regex': 0.2,
    'rfp': 0.1,
    'zinc': 0.2
}

for nas_env in monas_env:
    env_name_n_gens[nas_env] = 30
    env_name_perturb_proba[nas_env] = 0.45

def set_seed(seed):
    # print(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

class CollectHelper:
    def __init__(self, env_name, collect_type='amateur', filter_type='best', *args, **kwargs):
        self.env_name = env_name
        self.problem = get_problem(env_name, *args, **kwargs)
        self.data_size = collect_args.data_size
        self.raw_x = None
        self.raw_y = None
        self.save_path = os.path.join(base_path, 'data', env_name, filter_type)
        self.collect_per_iter = 50 if self.env_name.startswith('motsp') else 1
        self.collect_per_iter = 10 if self.env_name.startswith('mokp') else self.collect_per_iter
        self.collect_per_iter = 50 if self.env_name.startswith('mocvrp') else self.collect_per_iter
        os.makedirs(self.save_path, exist_ok=True)
        self.filter_type = filter_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'

        if env_name in synthetic_envs or env_name in moco_envs or env_name in monas_env:
            if collect_type == 'amateur':

                if env_name == 're34':
                    try:
                        from pymoo.util.ref_dirs import get_reference_directions
                        self.collect_algo = NSGA3(ref_dirs=get_reference_directions('uniform', 3, n_partitions=100))
                    except:
                         self.collect_algo = NSGA2(survival=AmateurRankAndCrowdSurvival(p=env_name_perturb_proba[env_name]))

                # elif env_name == 're91':
                #     try:
                #         from pymoo.util.ref_dirs import get_reference_directions
                #         self.collect_algo = MOEAD(ref_dirs=get_reference_directions('uniform', 3, n_partitions=50))
                #     except:
                #          self.collect_algo = NSGA2(survival=AmateurRankAndCrowdSurvival(p=env_name_perturb_proba[env_name]))

                elif env_name.startswith('motsp') or env_name.startswith('mocvrp'):

                    self.collect_algo = NSGA2(
                        pop_size=500,
                        sampling=PermutationRandomSampling(),
                        mutation=InversionMutation(),
                        crossover=OrderCrossover(),
                        repair=StartFromZeroRepair(),
                        survival=AmateurRankAndCrowdSurvival(p=env_name_perturb_proba[env_name]),
                        eliminate_duplicates=True,)
                    print(self.collect_algo)
                
                elif env_name.startswith('mokp'):
                    self.collect_algo = NSGA2(
                        pop_size=500,
                        sampling=PermutationRandomSampling(),
                        mutation=InversionMutation(),
                        crossover=OrderCrossover(),
                        survival=AmateurRankAndCrowdSurvival(p=env_name_perturb_proba[env_name]),
                        eliminate_duplicates=True,)
                    
                elif env_name == 'molecule':
                    self.collect_algo = NSGA2(pop_size=500, evaluator=MoleculeEvaluator(), callback=molecule_callback, 
                                            survival=AmateurRankAndCrowdSurvival(p=env_name_perturb_proba[env_name]),
                                            eliminate_duplicates=True)
                elif env_name in ['regex', 'rfp', 'zinc']:
                    from off_moo_bench.problem.lambo.lambo.optimizers.sampler import CandidateSampler
                    from pymoo.factory import get_crossover
                    from off_moo_bench.problem.lambo.lambo.optimizers.mutation import LocalMutation
                    from off_moo_bench.problem.lambo.lambo.utils import ResidueTokenizer
                    self.collect_algo = NSGA2(pop_size=16, sampling=CandidateSampler(tokenizer=ResidueTokenizer()),
                                              crossover=get_crossover(name='int_sbx', prob=0., eta=16),
                                              mutation=LocalMutation(prob=1., eta=16, safe_mut=False, tokenizer=ResidueTokenizer()),
                                              survival=AmateurRankAndCrowdSurvival(p=env_name_perturb_proba[env_name]),
                                              eliminate_duplicates=True)
                elif env_name in monas_env:
                    from off_moo_bench.problem.mo_nas import get_genetic_operator
                    sampling, crossover, mutation = get_genetic_operator()
                    self.collect_algo = NSGA2(pop_size=50, 
                                              sampling=sampling,
                                              crossover=crossover,
                                              mutation=mutation,
                                              survival=AmateurRankAndCrowdSurvival(p=env_name_perturb_proba[env_name]),
                                              eliminate_duplicates=True)

                else:
                    self.collect_algo = NSGA2(survival=AmateurRankAndCrowdSurvival(p=env_name_perturb_proba[env_name]),
                                              eliminate_duplicates=True)
            elif collect_type == 'expert':
                self.collect_algo = NSGA2(survival=RankAndCrowdingSurvival())
            elif collect_type == 'random':
                raise NotImplementedError
        
        
    def _run_and_collect(self):
        x, y = [], []
        seed = int(np.random.randint(low=0, high=10000, size=1))
        self.init_seed = seed

        def collect_callback(algorithm):
            try:
                print(algorithm.n_iter)
                if algorithm.n_iter % self.collect_per_iter == 0:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    x.append(X)
                    y.append(F)
            except:
                print(algorithm.n_gen)
                if algorithm.n_gen % self.collect_per_iter == 0:
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    x.append(X)
                    y.append(F)

        self.collect_algo.callback = collect_callback

        save_epoch = 5000
        accumulate_x = 0
        iiii = 0

        while True:
            seed = int(np.random.randint(low=0, high=100))
            set_seed(seed)
            try:
                x_all = np.concatenate(x, axis=0)
                accumulate_x += len(x_all)
                # x_all, indices = np.unique(x_all, axis=0, return_index=True)
            except:
                pass 

            if len(x) == 0 or x_all.shape[0] < self.data_size:
                if len(x) != 0:
                    print(x_all.shape[0], accumulate_x)
                    # if x_all.shape[0] >= save_epoch:
                    #     np.save(arr=x_all, file=os.path.join(self.save_path, f'{iiii}-x.npy'))
                    #     np.save(arr=np.concatenate(y, axis=0), file=os.path.join(self.save_path, f'{iiii}-y.npy'))
                    #     x = []
                    #     y = []
                    #     del x_all 
                    #     iiii += 1
                res = minimize(
                    algorithm=self.collect_algo,
                    problem=self.problem,
                    termination=('n_gen', env_name_n_gens[self.env_name]),
                    verbose=False
                )
            else:
                # np.save(arr=x_all, file=os.path.join(self.save_path, f'{iiii}-x.npy'))
                # np.save(arr=np.concatenate(y, axis=0), file=os.path.join(self.save_path, f'{iiii}-y.npy'))
                # sys.exit(0)
                y_all = np.concatenate(y, axis=0)
                # y_all = y_all[indices]
                indices_to_keep = np.random.choice(range(0, len(x_all)), size=self.data_size, replace=False)
                # from utils import get_N_nondominated_index
                # indices_to_keep = get_N_nondominated_index(y_all, self.data_size)
                raw_x_np = x_all[indices_to_keep]
                raw_y = y_all[indices_to_keep]
                raw_x = torch.from_numpy(raw_x_np).to(self.device)
                
                if self.env_name in synthetic_envs and \
                    not (self.env_name.lower().startswith('dtlz') or self.env_name in ['molecule', 'regex', 'rfp', 'zinc']):
                    raw_x = raw_x * (self.problem.ubound - self.problem.lbound) + self.problem.lbound

                self.raw_x = raw_x.cpu().numpy()
                self.raw_y = raw_y

                assert len(self.raw_y[0]) == self.problem.n_obj
                return

    def _save(self, preprocess=True):
        visible_masks = np.ones(len(self.raw_x))
        visible_masks[np.where(np.logical_or(np.isinf(self.raw_y), np.isnan(self.raw_y)))[0]] = 0
        visible_masks[np.where(np.logical_or(np.isinf(self.raw_x), np.isnan(self.raw_x)))[0]] = 0

        self.raw_x = self.raw_x[np.where(visible_masks == 1)[0]] 
        self.raw_y = self.raw_y[np.where(visible_masks == 1)[0]] 
        self.data_size = len(self.raw_x)

        raw_x_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-raw-x-0.npy")
        raw_y_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-raw-y-0.npy")
        np.save(file=raw_x_file, arr=self.raw_x)
        np.save(file=raw_y_file, arr=self.raw_y)
        print(f"Successfully save data to {raw_x_file} and {raw_y_file}.")

        nds = NonDominatedSorting()
        self.fronts, _ = nds.do(self.raw_y, return_rank=True)
        self.ranks = np.zeros((self.data_size,), dtype=int)
        cumulative_rank = 0
        for front in self.fronts:
            self.ranks[front] = int(cumulative_rank)
            cumulative_rank += len(front)
        rank_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-raw-rank-0.npy")
        np.save(file=rank_file, arr=self.ranks)
        print(f"Successfully save data to {rank_file}.")

        if preprocess:
            self._preprocess()

    def _preprocess(self):
        filter_ratio = collect_args.filter_ratio
        assert 0 <= filter_ratio <= 1, "Illegal filter ratio."
        n_filter = int(filter_ratio * self.data_size)
        
        # indices_to_keep = list(range(self.data_size))
        indices_to_keep = np.zeros(shape=(self.data_size, ))
        num_filtered = 0

        if self.filter_type == 'random_filter':
            indices_to_keep[np.random.choice(self.data_size, self.data_size-n_filter, replace=False)] = 1

        elif self.filter_type == 'best':
            indices_to_filter = np.zeros(shape=(self.data_size, ))
            indices_best_filter = np.zeros(shape=(self.data_size, ))
            for front in self.fronts:
                if num_filtered + len(front) > n_filter:
                    now_front = np.random.choice(front, n_filter-num_filtered, replace=False)
                    indices_to_filter[now_front] = 1
                    break
                else:
                    indices_to_filter[front] = 1
                    num_filtered += len(front)
            indices_to_keep[np.where(indices_to_filter == 0)] = 1
            indices_best_filter[np.where(indices_to_filter == 1)] = 1

        # elif self.filter_type == 'layer_and_random':
        #     indices_to_filter = np.zeros(shape=(self.data_size, ))
        #     for front in self.fronts[:3]:
        #         if num_filtered + len(front) > n_filter:
        #             now_front = np.random.choice(front, n_filter-num_filtered, replace=False)
        #             indices_to_filter[now_front] = 1
        #             num_filtered += (n_filter - num_filtered)
        #             break
        #         else:
        #             indices_to_filter[front] = 1
        #             num_filtered += len(front)
                    
        #     indices_to_keep[np.where(indices_to_filter == 0)] = 1

        #     if num_filtered < n_filter:
        #         idx_not_selected = np.where(indices_to_keep == 1)[0]
        #         indices_to_keep[
        #             np.random.choice(idx_not_selected,
        #                 (n_filter - num_filtered),
        #                 replace=False)] = 0
        #         indices_to_filter[np.where(indices_to_keep == 0)] = 1

        
        # elif self.filter_type == 'best_and_random':
        #     indices_to_filter = np.zeros(shape=(self.data_size, ))
        #     indices_best_filter = np.zeros(shape=(self.data_size, ))

        #     n_best_filter = int(0.05 * self.data_size)
        #     n_random_filter = int(0.35 * self.data_size)
        #     for front in self.fronts:
        #         if num_filtered + len(front) > n_best_filter:
        #             now_front = np.random.choice(front, n_best_filter-num_filtered, replace=False)
        #             indices_to_filter[now_front] = 1
        #             break
        #         else:
        #             indices_to_filter[front] = 1
        #             num_filtered += len(front)
            
        #     indices_to_keep[np.where(indices_to_filter == 0)] = 1
        #     indices_best_filter[np.where(indices_to_filter == 1)] = 1

        #     idx_not_selected = np.where(indices_to_keep == 1)[0]
        #     indices_to_keep[
        #             np.random.choice(idx_not_selected,
        #                 n_random_filter,
        #                 replace=False)] = 0
        #     indices_to_filter[np.where(indices_to_keep == 0)] = 1

        # elif self.filter_type == 'best_20':
        #     indices_to_filter = np.zeros(shape=(self.data_size, ))
        #     indices_best_filter = np.zeros(shape=(self.data_size, ))

        #     n_best_filter = int(0.2 * self.data_size)
        #     n_random_filter = int(0.2 * self.data_size)
        #     for front in self.fronts:
        #         if num_filtered + len(front) > n_best_filter:
        #             now_front = np.random.choice(front, n_best_filter-num_filtered, replace=False)
        #             indices_to_filter[now_front] = 1
        #             break
        #         else:
        #             indices_to_filter[front] = 1
        #             num_filtered += len(front)
            
        #     indices_to_keep[np.where(indices_to_filter == 0)] = 1
        #     indices_best_filter[np.where(indices_to_filter == 1)] = 1

        #     idx_not_selected = np.where(indices_to_keep == 1)[0]
        #     indices_to_keep[
        #             np.random.choice(idx_not_selected,
        #                 n_random_filter,
        #                 replace=False)] = 0
        #     indices_to_filter[np.where(indices_to_keep == 0)] = 1


        # elif self.filter_type == 'final':
        #     indices_to_filter = np.zeros(shape=(self.data_size, ))
        #     indices_best_filter = np.zeros(shape=(self.data_size, ))

        #     n_best_filter = int(0.1 * self.data_size)
        #     n_random_filter = int(0.3 * self.data_size)
        #     for front in self.fronts:
        #         if num_filtered + len(front) > n_best_filter:
        #             now_front = np.random.choice(front, n_best_filter-num_filtered, replace=False)
        #             indices_to_filter[now_front] = 1
        #             break
        #         else:
        #             indices_to_filter[front] = 1
        #             num_filtered += len(front)
            
        #     indices_to_keep[np.where(indices_to_filter == 0)] = 1
        #     indices_best_filter[np.where(indices_to_filter == 1)] = 1

        #     idx_not_selected = np.where(indices_to_keep == 1)[0]
        #     indices_to_keep[
        #             np.random.choice(idx_not_selected,
        #                 n_random_filter,
        #                 replace=False)] = 0
        #     indices_to_filter[np.where(indices_to_keep == 0)] = 1
            
        else:
            raise ValueError(f'Unknown filter type {self.filter_type}.')

        # indices_to_keep = torch.tensor(indices_to_keep).to(self.device)
        # res_x = torch.from_numpy(self.raw_x).to(self.device)
        # res_y = torch.from_numpy(self.raw_y).to(self.device)
        
        x_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-x-0.npy")
        y_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-y-0.npy")

        filter_x_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-filter-x-0.npy")
        filter_y_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-filter-y-0.npy")

        # print(np.where(indices_to_keep == 1))

        self.processed_x = self.raw_x[np.where(indices_to_keep == 1)]
        self.filtered_x = self.raw_x[np.where(indices_to_keep == 0)]
        
        np.save(file=x_file, arr=self.processed_x)
        print(f"Successfully save data to {x_file}.")

        np.save(file=filter_x_file, arr=self.filtered_x)
        print(f"Successfully save data to {filter_x_file}.")

        self.processed_y = self.raw_y[np.where(indices_to_keep == 1)]
        self.filtered_y = self.raw_y[np.where(indices_to_keep == 0)]
        assert self.processed_x.shape[0] == self.processed_y.shape[0]

        np.save(file=y_file, arr=self.processed_y)
        print(f"Successfully save data to {y_file}.")

        np.save(file=filter_y_file, arr=self.filtered_y)
        print(f"Successfully save data to {filter_y_file}.")

        if self.filter_type.startswith('best'):
            test_x_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-test-x-0.npy")
            test_y_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-test-y-0.npy")

            self.test_x = self.raw_x[np.where(indices_best_filter == 1)]
            self.test_y = self.raw_y[np.where(indices_best_filter == 1)]

            np.save(file=test_x_file, arr=self.test_x)
            print(f"Successfully save data to {test_x_file}.")

            np.save(file=test_y_file, arr=self.test_y)
            print(f"Successfully save data to {test_y_file}.")

        print(f"Data for problem {self.problem.name} has been preprocessed.")

        
    def collect_data(self):
        print("\n" + f"Begin to collect data for problem {self.problem.name}.")
        if self.env_name in monas_env:
            from evoxbench.benchmarks.nb201 import NASBench201SearchSpace
            from evoxbench.benchmarks.nats import NATSBenchSearchSpace
            problem = get_problem(self.env_name)
            search_space = problem.benchmark.search_space
            if isinstance(search_space, (NASBench201SearchSpace, NATSBenchSearchSpace)):
                def generate_feature_permutations(bounds):
                    feature_values = [np.arange(start, stop + 1) for start, stop in bounds]
                    grids = np.meshgrid(*feature_values)
                    feature_permutations = np.stack([grid.ravel() for grid in grids], axis=-1) 
                    return feature_permutations
                perm_matrix = generate_feature_permutations(np.stack([problem.xl, problem.xu], axis=-1))
                np.random.shuffle(perm_matrix)
                self.raw_x = perm_matrix.astype(np.int32)
                self.raw_y = problem.evaluate(self.raw_x)
            else:
                self._run_and_collect()

        elif (self.env_name in synthetic_envs or self.env_name in moco_envs)  \
            and self.env_name not in ['molecule', 'regex', 'rfp']:
            self._run_and_collect()
        else:
            raw_x_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-raw-x-0.npy")
            raw_y_file = os.path.join(self.save_path, f"{self.problem.name.lower()}-raw-y-0.npy")
            self.raw_x = np.load(raw_x_file)
            self.raw_y = np.load(raw_y_file)
            np.save(arr=self.raw_y, file=raw_y_file)
        self._save()

if __name__ == "__main__":
    env_name = collect_args.env_name
    assert env_name in env_names
    import time
    start_time = time.time()
    collect_helper = CollectHelper(env_name, collect_type=collect_args.collect_type, 
                                   filter_type=collect_args.filter_type)
    collect_helper.collect_data()
    print(time.time() - start_time)

