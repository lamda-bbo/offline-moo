import os
import numpy as np
import torch
import sys 
import yaml 
from typing import List, Optional
from types import SimpleNamespace

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

tkwargs = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'dtype': torch.float32
}

now_fronts = None
now_seed = None

def calc_crowding_distance(F) -> np.ndarray:

    if isinstance(F, list) or isinstance(F, np.ndarray):
        F = torch.tensor(F).to(**tkwargs)

    n_points, n_obj = F.shape

    # sort each column and get index
    I = torch.argsort(F, dim=0, descending=False)

    # sort the objective space values for the whole matrix
    F_sorted = torch.gather(F, 0, I)

    # calculate the distance from each point to the last and next
    inf_tensor = torch.full((1, n_obj), float('inf'), device=F.device, dtype=F.dtype)
    neg_inf_tensor = torch.full((1, n_obj), float('-inf'), device=F.device, dtype=F.dtype)
    dist = torch.cat([F_sorted, inf_tensor], dim=0) - torch.cat([neg_inf_tensor, F_sorted], dim=0)

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = torch.max(F_sorted, dim=0).values - torch.min(F_sorted, dim=0).values
    norm[norm == 0] = float('nan')

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist[:-1], dist[1:]
    dist_to_last, dist_to_next = dist_to_last / norm, dist_to_next / norm

    # if we divide by zero because all values in one column are equal replace by none
    dist_to_last[torch.isnan(dist_to_last)] = 0.0
    dist_to_next[torch.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = torch.argsort(I, dim=0, descending=False)
    crowding_dist = torch.sum(
        torch.gather(dist_to_last, 0, J) + torch.gather(dist_to_next, 0, J),
        dim=1
    ) / n_obj

    return crowding_dist.detach().cpu().numpy()

def _get_fronts(Y_all):
    global now_fronts
    if now_fronts is not None:
        return now_fronts
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    fronts = NonDominatedSorting().do(Y_all, return_rank=True)[0]
    now_fronts = fronts 
    return fronts

def get_N_nondominated_index(Y_all, num_ret, is_all_data=False) -> List[int]:
    if is_all_data:
        fronts = _get_fronts(Y_all)
    else:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        fronts = NonDominatedSorting().do(Y_all, return_rank=True, n_stop_if_ranked=num_ret)[0]
    indices_cnt = 0
    indices_select = []
    for front in fronts:
        if indices_cnt + len(front) < num_ret:
            indices_cnt += len(front)
            indices_select += [int(i) for i in front]
        else:
            n_keep = num_ret - indices_cnt
            F = Y_all[front]

            from pymoo.util.misc import find_duplicates
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]

            _F = F[is_unique]
            _d = calc_crowding_distance(_F)

            d = np.zeros(len(front))
            d[is_unique] = _d 
            I = np.argsort(d)[-n_keep:]
            indices_select += [int(i) for i in I]
            break
        
    return indices_select

def get_quantile_solutions(Y_all: np.ndarray, quantile) -> np.ndarray:
    assert 0 < quantile < 1
    n = len(Y_all)
    n_remove = int(n * (1-quantile))
    indices_to_remove = get_N_nondominated_index(Y_all, n_remove)
    indices_to_keep = np.ones(n)
    indices_to_keep[indices_to_remove] = 0
    return Y_all[np.where(indices_to_keep == 1)[0]]


def get_data_path(env_name):
    return os.path.join(base_path, 'data', env_name)

def read_data(env_name, filter_type='best', return_x=True, return_y=True, return_rank=True):
    try:
        assert return_x or return_y or return_rank, "Illegal params."
        env_name = env_name.lower()
        data_path = os.path.join(base_path, "data", env_name)
        x_file = os.path.join(data_path, f"{env_name}-x-0.npy") if return_x else None
        y_file = os.path.join(data_path, f"{env_name}-y-0.npy") if return_y else None
        rank_file = os.path.join(data_path, f"{env_name}-rank-0.npy") if return_rank else None
        
        x = np.load(x_file) if return_x else None
        y = np.load(y_file) if return_y else None
        rank = np.load(rank_file) if return_rank else None

        return (x if return_x else None,
                y if return_y else None, 
                rank if return_rank else None)
    except:
        assert return_x or return_y or return_rank, "Illegal params."
        env_name = env_name.lower()
        data_path = os.path.join(base_path, "data", env_name, filter_type)
        x_file = os.path.join(data_path, f"{env_name}-x-0.npy") if return_x else None
        y_file = os.path.join(data_path, f"{env_name}-y-0.npy") if return_y else None
        rank_file = os.path.join(data_path, f"{env_name}-rank-0.npy") if return_rank else None
        
        x = np.load(x_file) if return_x else None
        y = np.load(y_file) if return_y else None
        rank = np.load(rank_file) if return_rank else None

        return (x if return_x else None,
                y if return_y else None, 
                rank if return_rank else None)

def read_raw_data(env_name, filter_type='best', return_x=True, return_y=True, return_rank=True):
    try:
        assert return_x or return_y or return_rank, "Illegal params."
        env_name = env_name.lower()
        data_path = os.path.join(base_path, "data", env_name)
        x_file = os.path.join(data_path, f"{env_name}-x-0.npy") if return_x else None
        y_file = os.path.join(data_path, f"{env_name}-y-0.npy") if return_y else None
        rank_file = os.path.join(data_path, f"{env_name}-rank.npy") if return_rank else None
        
        test_x_file = os.path.join(data_path, f"{env_name}-test-x-0.npy") if return_x else None
        test_y_file = os.path.join(data_path, f"{env_name}-test-y-0.npy") if return_y else None

        x = np.load(x_file) if return_x else None
        y = np.load(y_file) if return_y else None
        
        
        test_x = np.load(test_x_file) if return_x else None
        test_y = np.load(test_y_file) if return_y else None
        
        x = np.concatenate((x, test_x), axis=0) if return_x else None 
        y = np.concatenate((y, test_y), axis=0) if return_y else None 
        
        rank = np.load(rank_file) if return_rank else None
        

        return (x if return_x else None,
                y if return_y else None, 
                rank if return_rank else None)

    except:
        assert return_x or return_y or return_rank, "Illegal params."
        env_name = env_name.lower()
        data_path = os.path.join(base_path, "data", env_name, filter_type)
        x_file = os.path.join(data_path, f"{env_name}-raw-x-0.npy") if return_x else None
        y_file = os.path.join(data_path, f"{env_name}-raw-y-0.npy") if return_y else None
        rank_file = os.path.join(data_path, f"{env_name}-rank.npy") if return_rank else None

        x = np.load(x_file) if return_x else None
        y = np.load(y_file) if return_y else None
        rank = np.load(rank_file) if return_rank else None

        return (x if return_x else None,
                y if return_y else None, 
                rank if return_rank else None)

def read_filter_data(env_name, filter_type='best', return_x=True, return_y=True, return_rank=True):
    try:
        assert return_x or return_y or return_rank, "Illegal params."
        env_name = env_name.lower()
        data_path = os.path.join(base_path, "data", env_name)
        x_file = os.path.join(data_path, f"{env_name}-test-x-0.npy") if return_x else None
        y_file = os.path.join(data_path, f"{env_name}-test-y-0.npy") if return_y else None
        rank_file = os.path.join(data_path, f"{env_name}-rank.npy") if return_rank else None

        x = np.load(x_file) if return_x else None
        y = np.load(y_file) if return_y else None
        rank = np.load(rank_file) if return_rank else None

        return (x if return_x else None,
                y if return_y else None, 
                rank if return_rank else None)
    except:
        assert return_x or return_y or return_rank, "Illegal params."
        env_name = env_name.lower()
        data_path = os.path.join(base_path, "data", env_name, filter_type)
        x_file = os.path.join(data_path, f"{env_name}-filter-x-0.npy") if return_x else None
        y_file = os.path.join(data_path, f"{env_name}-filter-y-0.npy") if return_y else None
        rank_file = os.path.join(data_path, f"{env_name}-rank.npy") if return_rank else None

        x = np.load(x_file) if return_x else None
        y = np.load(y_file) if return_y else None
        rank = np.load(rank_file) if return_rank else None

        return (x if return_x else None,
                y if return_y else None, 
                rank if return_rank else None)


def get_model_path(args, model_type, name):
    normalize_str = 'final-'

    normalize_str += f'normalize-{args.filter_type}' if args.normalize_y \
                                    else f'no_normalize-{args.filter_type}'
    normalize_str = f'{normalize_str}-{args.train_data_mode}' if args.train_data_mode != 'none'\
                            else normalize_str
    normalize_str = f'{normalize_str}-{args.train_mode}' if args.train_mode != 'none'\
                            else normalize_str
    normalize_str = f'{normalize_str}-{args.reweight_mode}' if args.reweight_mode != 'none'\
                            else normalize_str
    save_dir = os.path.join(base_path, 'model', args.env_name, model_type, normalize_str)
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, name + '.pth')

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True
    global now_seed
    now_seed = seed

def get_config_path(config_name):
    return os.path.join(base_path, 'configs', f'{config_name}.config')

def load_config(path, extra=None):
    if not os.path.exists(path):
        path = get_config_path(path)
        assert os.path.exists(path), f'Cannot find config file {path}.'
    
    support_types = ('int', 'float', 'str', 'bool', 'none')
    
    def convert_param(type_param_list):
        assert isinstance(type_param_list, list), \
            f'Illegal config: {type_param_list}'
        
        param_type, params = type_param_list[0], type_param_list[1]
        assert param_type in support_types, \
            f'Unsupported type {param_type}. It should be with in {support_types}.'
        
        is_list = isinstance(params, list)
        if not is_list:
            params = [params]
        
        res = []
        for x in params:
            if param_type == 'int':
                x = int(x)
            elif param_type == 'str':
                x = str(x)
            elif param_type == 'bool':
                x = bool(int(x))
            elif param_type == 'float':
                x = float(x)
            elif param_type == 'none':
                if x.lower() != 'none':
                    raise ValueError(f'For the none type, the value must be none instead of {x}')
                x = None
            else:
                raise TypeError(f'Does not know this type: {param_type}')
            res.append(x)
        if not is_list:
            res = res[0]
        return res

    import json
    with open(path, 'rb') as f:
        data = json.load(f)
    
    content = {k: convert_param(v) for k, v in data.items()}
    assert extra is None or isinstance(extra, dict), \
        f'Invalid type of extra: {extra}'
    
    if isinstance(extra, dict):
        content = {**content, **extra}
    
    return content
            
def one_hot(a, num_classes):
    out = np.zeros((a.size, num_classes), dtype=np.float32)
    out[np.arange(a.size), a.ravel()] = 1.0
    out.shape = a.shape + (num_classes, )
    return out

num_classes_map = {
    'nb201_test': 5,
    'qm9': 591
}

def to_logits(x, args, num_classes=2, soft_interpolation=0.6):

    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError('Cannot convert non-integers to logits')
    
    num_classes = num_classes_map[args.env_name]

    one_hot_x = one_hot(x, num_classes)
    uniform_prior = np.full_like(one_hot_x, 1 / float(num_classes))

    soft_x = soft_interpolation * one_hot_x + (
        1.0 - soft_interpolation) * uniform_prior
    
    x = np.log(soft_x)

    return (x[:, :, 1:] - x[:, :, :1]).astype(np.float32)

def to_integers(x):
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError('Cannot convert non-floating to integers')

    return np.argmax(np.pad(x, [[0, 0]] * (
            len(x.shape) - 1) + [[1, 0]]), axis=-1).astype(np.int32)

def normalize_x(args, x):
    raw_x, _, _ = read_raw_data(args.env_name, filter_type=args.filter_type, return_y=False, return_rank=False)
    if args.discrete:
        from utils import to_logits
        raw_x = to_logits(raw_x, args)
        raw_x = raw_x.reshape((-1, ) + x.shape[1:])
    x_max = np.max(raw_x, axis=0)
    x_min = np.min(raw_x, axis=0)
    x_ret = (x - x_min) / (x_max - x_min)
    return np.nan_to_num(x_ret)

def normalize_y(args, y):
    _, raw_y, _ = read_raw_data(args.env_name, filter_type=args.filter_type, return_x=False, return_rank=False)
    y_max = np.max(raw_y, axis=0)
    y_min = np.min(raw_y, axis=0)
    y_ret = (y - y_min) / (y_max - y_min)
    return np.nan_to_num(y_ret)

def denormalize_x(args, x):
    raw_x, _, _ = read_raw_data(args.env_name, filter_type=args.filter_type, return_y=False, return_rank=False)
    if args.discrete:
        from utils import to_logits
        raw_x = to_logits(raw_x, args)
        raw_x = raw_x.reshape((-1, ) + x.shape[1:])
    x_max = np.max(raw_x, axis=0)
    x_min = np.min(raw_x, axis=0)
    return x * (x_max - x_min) + x_min

def denormalize_y(args, y):
    _, raw_y, _ = read_raw_data(args.env_name, filter_type=args.filter_type, return_x=False, return_rank=False)
    y_max = np.max(raw_y, axis=0)
    y_min = np.min(raw_y, axis=0)
    return y * (y_max - y_min) + y_min

def plot_and_save_mse(model, mse, which_loss='val'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(mse)), mse)
    plt.xlabel('# epoch')
    plt.ylabel(f'{which_loss} loss')
    if model.args.train_mode != 'ict':
        plt.savefig(os.path.join(model.args.results_dir, f'model_{which_loss}_loss.png'))
        plt.savefig(os.path.join(model.args.results_dir, f'model_{which_loss}_loss.pdf'))       
        np.save(arr=np.array(mse), file=os.path.join(model.args.results_dir, f'model_{which_loss}_loss.npy'))
    else:
        plt.savefig(os.path.join(model.results_dir, f'model_{which_loss}_loss.png'))
        plt.savefig(os.path.join(model.results_dir, f'model_{which_loss}_loss.pdf'))       
        np.save(arr=np.array(mse), file=os.path.join(model.results_dir, f'model_{which_loss}_loss.npy'))


def process_args(return_dict=False):
    params = [arg.lstrip("--") for arg in sys.argv if arg.startswith("--")]
    cmd_config_dict = {} 
    for arg in params:
        key, value = arg.split('=')
        try:
            cmd_config_dict[key] = eval(value)
        except:
            cmd_config_dict[key] = value 
            
    # default config
    config_path = os.path.join(
        base_path,
        "configs",
        "default.yaml"
    )
    assert os.path.exists(config_path), f"Config {config_path} not found"
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in cmd_config_dict.items():
        config_dict[key] = value
        
    # model config
    model_config_path =  os.path.join(
        base_path,
        "configs",
        "algorithm",
        f"{config_dict['model']}-{config_dict['train_mode']}.yaml"
    )
    assert os.path.exists(model_config_path), \
        f"Model config {model_config_path} not found"
    with open(model_config_path, 'r') as f:
        try:
            config_dict.update(yaml.load(f, Loader=yaml.FullLoader))
        except:
            pass

    # task config
    task_config_path =  os.path.join(
        base_path,
        "configs",
        "task",
        f"{config_dict['task']}.yaml"
    )
    
    default_task_config_path =  os.path.join(
        base_path,
        "configs",
        "task",
        f"default.yaml"
    )
    assert os.path.exists(task_config_path) or \
        os.path.exists(default_task_config_path), \
        f"Problem config {task_config_path} or {default_task_config_path} not found"
    try:
        with open(task_config_path, 'r') as f:
            try:
                config_dict.update(yaml.load(f, Loader=yaml.FullLoader))
            except:
                pass
    except:
        with open(default_task_config_path, 'r') as f:
            try:
                config_dict.update(yaml.load(f, Loader=yaml.FullLoader))
            except:
                pass

    for key, value in cmd_config_dict.items():
        config_dict[key] = value 
    
    print("All config:", config_dict)
    
    return config_dict if return_dict else SimpleNamespace(**config_dict)
    

if __name__ == "__main__":
    _, y, _ = read_data(env_name='re21', filter_type='best', return_x=False, return_rank=False)
    y = y[:1000, :]
    i1 = get_N_nondominated_index(Y_all = y, num_ret=100)
    y1 = y[i1]
    y2 = get_quantile_solutions(y1, 0.5)
    print(y1.shape, y2.shape)
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.scatter(y1[:, 0], y1[:, 1], color='red')
    plt.scatter(y2[:, 0], y2[:, 1], color='blue')
    plt.savefig('test_quan.png')