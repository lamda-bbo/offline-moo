import torch
from pymoo.algorithms.moo.nsga2 import NonDominatedSorting

def custom_sigmoid(x, alpha=0.2):
    return 1 / (1 + torch.exp(-alpha * x))

def sigmoid_reweighting(y, quantile=0.25):

    ranks = torch.ones(y.shape[0])
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    fronts = NonDominatedSorting().do(y)

    pivot = -1
    pivot_loc = int(len(y) * quantile)
    cnt = 0

    for i, front in enumerate(fronts):
        ranks[front] = i
        if pivot == -1:
            cnt += len(front)
            if cnt > pivot_loc:
                pivot = i
        
    ranks = (-1) * ranks
    ranks += pivot
    weights = custom_sigmoid(ranks)
    weights = weights / torch.mean(weights, axis=0)
    return weights
