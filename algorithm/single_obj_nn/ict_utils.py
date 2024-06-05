import os
import re
import requests
import numpy as np
import functools

import random
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import scipy
import scipy.stats
import itertools

d = None


def load_d(task):
    global d
    d = np.load("npy/" + task + ".npy", allow_pickle=True)
    print("loading shape", d.shape)


weights = None


def load_weights(task_name, y, gamma):
    global weights
    # if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0','UTR-ResNet-v0']:
    #    index = np.argsort(y, axis=0).squeeze()
    #    anchor = y[index][-10]
    #    tmp = y>=anchor
    #    weights = tmp/np.sum(tmp)
    # elif task_name in ['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
    #        'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0']:
    #    tmp = np.exp(gamma*y)
    #    weights = tmp/np.sum(tmp)
    tmp = np.exp(gamma * y)
    weights = tmp / np.sum(tmp)
    print("weights", np.max(weights), np.min(weights))


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


y_min = None
y_max = None


def load_y(task_name):
    global y_min
    global y_max
    dic2y = np.load("npy/dic2y.npy", allow_pickle=True).item()
    y_min, y_max = dic2y[task_name]


def process_data(task, task_name, task_y0=None):
    if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0', 'UTR-ResNet-v0', 'TFBind10-Exact-v0', 'CIFARNAS-Exact-v0']:
        task_x = task.to_logits(task.x)
        if task_name == 'TFBind10-Exact-v0':
            interval = np.arange(0, 4161482, 83, dtype=int)[0: 50000]
            index = np.argsort(task_y0.squeeze())
            index = index[interval]
            task_x = task_x[index]
            task_y0 = task_y0[index]
    elif task_name in ['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
                       'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0']:
        task_x = copy.deepcopy(task.x)
    task_x = task.normalize_x(task_x)
    shape0 = task_x.shape
    task_x = task_x.reshape(task_x.shape[0], -1)
    if task_name in ['UTR-ResNet-v0']:
        mean_y = np.mean(task_y0)
        std_y = np.std(task_y0 - mean_y)
        task_y = (task_y0 - mean_y) / std_y
    else:
        task_y = task.normalize_y(task_y0)

    return task_x, task_y, shape0


def evaluate_sample(task, x_init, task_name, shape0):
    x_init = x_init.cpu().numpy()
    if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0', 'UTR-ResNet-v0', 'TFBind10-Exact-v0', 'CIFARNAS-Exact-v0']:
        X1 = x_init.reshape(-1, shape0[1], shape0[2])
    elif task_name in ['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
                       'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0']:
        X1 = x_init
    X1 = task.denormalize_x(X1)
    if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0', 'UTR-ResNet-v0', 'TFBind10-Exact-v0', 'CIFARNAS-Exact-v0']:
        X1 = task.to_integers(X1)
    # print(X1, X1.shape)
    Y1 = task.predict(X1)
    max_v = (np.max(Y1) - y_min) / (y_max - y_min)
    med_v = (np.median(Y1) - y_min) / (y_max - y_min)
    return max_v, med_v
    # return np.max(Y1), np.median(Y1)


def adjust_learning_rate(optimizer, lr0, epoch, T):
    lr = lr0 * (1 + np.cos((np.pi * epoch * 1.0) / (T * 1.0))) / 2.0
    print("epoch {} lr {}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_pcc(valid_preds, valid_labels):
    # vx = valid_preds.shape[0]  - torch.argsort(valid_preds)
    # vy = valid_labels.shape[0]  - torch.argsort(valid_labels)
    # vx = vx.float()
    # vy = vy.float()
    vx = valid_preds - torch.mean(valid_preds)
    vy = valid_labels - torch.mean(valid_labels)
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-12) * torch.sqrt(torch.sum(vy ** 2) + 1e-12))
    return pcc


def compute_invpair(r1, r2):
    tau, p_value = scipy.stats.kendalltau(r1, r2)
    N = r1.shape[0]
    return round(N * (N - 1) * (1 - tau) / 4)


def compute_meanrank(candidates):
    m = np.mean(candidates, axis=0)
    rank = np.arange(candidates.shape[1])
    rank[np.argsort(m)] = np.arange(candidates.shape[1])
    return rank


def compute_bestrank(candidates):
    best_ranking = None
    best_loss = 666666
    # generate all possible rankings
    ranking = np.arange(candidates.shape[1])
    rankings = list(itertools.permutations(ranking))
    for ranking in rankings:
        ranking = np.array(ranking)
        ranking_loss = 0
        for i in range(candidates.shape[0]):
            ranking_loss = ranking_loss + compute_invpair(ranking, candidates[i])
        print(ranking, 'loss', ranking_loss)
        if ranking_loss < best_loss:
            best_loss = ranking_loss
            best_ranking = ranking
    return best_ranking


def adjust_bpr(proxy, pairs):
    # pairs: 2xN
    # print('entering', proxy)
    opt = torch.optim.SGD(proxy.parameters(), lr=1e-3)
    pos_score = proxy(pairs[0, :])
    neg_score = proxy(pairs[1, :])
    loss = torch.mean(-torch.log(torch.sigmoid(pos_score - neg_score) + 1e-9))
    opt.zero_grad()
    # loss.backward(retain_graph=True)
    loss.backward()
    opt.step()


def compute_rank(pred):
    N = pred.shape[0]
    rank = torch.arange(N)
    rank[torch.argsort(-pred)] = torch.arange(N)
    return rank


def compute_inv_pairs(mrank, rank1):
    # 2*N, pos, neg
    N = rank1.shape[0]
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if (rank1[i] - rank1[j]) * (mrank[i] - mrank[j]) < 0:
                if rank1[i] > rank1[j]:
                    pair = [i, j]
                else:
                    pair = [j, i]
                pairs.append(pair)
    return torch.Tensor(np.array(pairs).T).long()


# reinforce pos > neg, all agree
def pair2vec(pairs, candidates):
    vec1 = candidates[pairs[0, :]].unsqueeze(0)
    vec2 = candidates[pairs[1, :]].unsqueeze(0)
    return torch.cat([vec1, vec2], dim=0)


def adjust_corank(proxy1, proxy2, candidate, N=10):
    candidate_neighbors = candidate.repeat(N, 1)
    candidate_neighbors = candidate_neighbors + 0.1 * \
                          torch.randn(candidate_neighbors.shape).to(candidate.device)
    pred1 = proxy1(candidate_neighbors).data.squeeze()
    pred2 = proxy2(candidate_neighbors).data.squeeze()
    rank1 = compute_rank(pred1)
    rank2 = compute_rank(pred2)
    mrank = compute_rank(-(rank1 + rank2) / 2.0)
    pairs1 = compute_inv_pairs(mrank, rank1)

    if pairs1.shape[0]:
        pairs1 = pair2vec(pairs1, candidate_neighbors)
        adjust_bpr(proxy1, pairs1.to(candidate.device))
    pairs2 = compute_inv_pairs(mrank, rank2)
    if pairs2.shape[0]:
        pairs2 = pair2vec(pairs2, candidate_neighbors)
        adjust_bpr(proxy2, pairs2.to(candidate.device))


def rank2bi(rank, N):
    rank[rank < int(N / 2)] = 0
    rank[rank >= int(N / 2)] = 1
    return rank


def compute_tri_inv_pairs(rank1, rank2, rank3):
    N = rank1.shape[0]
    pairs1 = []
    pairs2 = []
    pairs3 = []
    for i in range(N):
        for j in range(i + 1, N):
            rank_diff1 = rank1[i] - rank1[j]
            rank_diff2 = rank2[i] - rank2[j]
            rank_diff3 = rank3[i] - rank3[j]
            # identify violating pairs for model1
            if (rank_diff2 * rank_diff3 > 0) and (rank_diff1 * rank_diff2 < 0):
                if rank_diff1 > 0:
                    pairs1.append([i, j])
                else:
                    pairs1.append([j, i])
            # identify violating pairs for model2
            if (rank_diff1 * rank_diff3 > 0) and (rank_diff2 * rank_diff1 < 0):
                if rank_diff2 > 0:
                    pairs2.append([i, j])
                else:
                    pairs2.append([j, i])
            # identify violating pairs for model3
            if (rank_diff1 * rank_diff2 > 0) and (rank_diff3 * rank_diff1 < 0):
                if rank_diff3 > 0:
                    pairs3.append([i, j])
                else:
                    pairs3.append([j, i])
    pairs1 = torch.Tensor(np.array(pairs1).T).long()
    pairs2 = torch.Tensor(np.array(pairs2).T).long()
    pairs3 = torch.Tensor(np.array(pairs3).T).long()
    return pairs1, pairs2, pairs3


def adjust_proxy(proxy1, proxy2, proxy3, candidate, N=10):
    # compute neighbors
    candidate_neighbors = candidate.repeat(N, 1)
    candidate_neighbors = candidate_neighbors + 0.1 * \
                          torch.randn(candidate_neighbors.shape).to(candidate.device)
    # compute predictions & rankings
    pred1 = proxy1(candidate_neighbors).data.squeeze()
    pred2 = proxy2(candidate_neighbors).data.squeeze()
    pred3 = proxy3(candidate_neighbors).data.squeeze()
    rank1 = compute_rank(pred1)
    rank2 = compute_rank(pred2)
    rank3 = compute_rank(pred3)
    pairs1, pairs2, pairs3 = compute_tri_inv_pairs(rank1, rank2, rank3)
    # print('shape', pairs1.shape, pairs2.shape, pairs3.shape)
    if pairs1.shape[0]:
        pairs1 = pair2vec(pairs1, candidate_neighbors)
        adjust_bpr(proxy1, pairs1.to(candidate.device))
    if pairs2.shape[0]:
        pairs2 = pair2vec(pairs2, candidate_neighbors)
        adjust_bpr(proxy2, pairs2.to(candidate.device))
    if pairs3.shape[0]:
        pairs3 = pair2vec(pairs3, candidate_neighbors)
        adjust_bpr(proxy3, pairs3.to(candidate.device))


if __name__ == "__main__":
    r1 = np.arange(6)
    r2 = np.array([0, 3, 1, 2, 4, 5])  # r1[::-1]
    r3 = np.random.permutation(6)
    # print(compute_invpair(r1, r2), compute_invpair(r2, r1))
    # r = compute_bestrank(np.concatenate([r1.reshape(1, -1), r2.reshape(1, -1), r3.reshape(1, -1)], axis=0))
    r = compute_meanrank(np.concatenate([r1.reshape(1, -1), r2.reshape(1, -1), r3.reshape(1, -1)], axis=0))
    print("r1", r1, "r2", r2, "r3", r3)
    print('final', r)
