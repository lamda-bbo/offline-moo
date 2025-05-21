import numpy as np
import torch

import off_moo_bench as ob
from off_moo_bench.problem import RE21, RE23


class RE21_stand:
    def __init__(self, n_dim=4):
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma

        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor(
            [tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]
        ).float()
        self.ubound = torch.ones(n_dim).float() * 3 * tmp_val
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]

    def evaluate(self, x):
        F = 10.0
        E = 2.0 * 1e5
        L = 200.0

        if x.device.type == "cuda":
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        # x = x * (self.ubound - self.lbound) + self.lbound

        f1 = L * (
            (2 * x[:, 0]) + np.sqrt(2.0) * x[:, 1] + torch.sqrt(x[:, 2]) + x[:, 3]
        )
        f2 = ((F * L) / E) * (
            (2.0 / x[:, 0])
            + (2.0 * np.sqrt(2.0) / x[:, 1])
            - (2.0 * np.sqrt(2.0) / x[:, 2])
            + (2.0 / x[:, 3])
        )

        f1 = f1
        f2 = f2

        objs = torch.stack([f1, f2]).T

        return objs


class RE23_Stand:
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([1, 1, 10, 10]).float()
        self.ubound = torch.tensor([100, 100, 200, 240]).float()
        self.nadir_point = [5852.05896876, 1288669.78054]

    def evaluate(self, x):
        if x.device.type == "cuda":
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        # x = x * (self.ubound - self.lbound) + self.lbound

        x1 = 0.0625 * torch.round(x[:, 0])
        x2 = 0.0625 * torch.round(x[:, 1])
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = (
            (0.6224 * x1 * x3 * x4)
            + (1.7781 * x2 * x3 * x3)
            + (3.1661 * x1 * x1 * x4)
            + (19.84 * x1 * x1 * x3)
        )
        f1 = f1.float()

        # Original constraint functions
        g1 = x1 - (0.0193 * x3)
        g2 = x2 - (0.00954 * x3)
        g3 = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000

        g = torch.stack([g1, g2, g3])
        z = torch.zeros(g.shape).cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f2 = torch.sum(g, axis=0).to(torch.float64)

        objs = torch.stack([f1, f2]).T

        return objs


task_ob = ob.make("RE23-Exact-v0")
task = RE23()
task_stand = RE23_Stand()
# x = np.random.rand(10, 4) * (task.xu - task.xl) + task.xl
x = task_ob.x[:50]
print("standard y:", task_stand.evaluate(torch.from_numpy(x).cuda()))
print("y:", task.evaluate(x))
print("ob y:", task_ob.predict(x))
print("data y:", task_ob.y[:50])
