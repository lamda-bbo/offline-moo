import torch 
from gpytorch.kernels import Kernel

from off_moo_baselines.util.data_structure import FeatureCache
from off_moo_baselines.mobo.mobo_utils import tkwargs

feature_cache = FeatureCache() 

class OrderKernel(Kernel):
    has_lengthscale = True

    def forward(self, X, X2, **params):
        global feature_cache
        mat = torch.zeros((len(X), len(X2))).to(**tkwargs)
        x1 = []
        for i in range(len(X)):
            x1.append(feature_cache.push(X[i]))
        x2 = []
        for j in range(len(X2)):
            x2.append(feature_cache.push(X2[j]))
        #mat = self._count_discordant_pairs(x1, x2)
        x1 = torch.vstack(x1).to(**tkwargs)
        x2 = torch.vstack(x2).to(**tkwargs)
        x1 = torch.reshape(x1, (x1.shape[0], 1, -1)).to(**tkwargs)
        x2 = torch.reshape(x2, (1, x2.shape[0], -1)).to(**tkwargs)
        x1 = torch.tile(x1, (1, x2.shape[0], 1)).to(**tkwargs)
        x2 = torch.tile(x2, (x1.shape[0], 1, 1)).to(**tkwargs)
        mat = torch.sum((x1 - x2)**2, dim=-1).to(**tkwargs)
        mat = torch.exp(- self.lengthscale * mat)
        return mat

class CategoricalOverlap(Kernel):
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(CategoricalOverlap, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # First, convert one-hot to ordinal representation

        diff = x1[:, None] - x2[None, :]
        # nonzero location = different cat
        diff[torch.abs(diff) > 1e-5] = 1
        # invert, to now count same cats
        diff1 = torch.logical_not(diff).float()
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.sum(self.lengthscale * diff1, dim=-1) / torch.sum(self.lengthscale)
        else:
            # dividing by number of cat variables to keep this term in range [0,1]
            k_cat = torch.sum(diff1, dim=-1) / x1.shape[1]
        if diag:
            return torch.diag(k_cat).to(**tkwargs)
        return k_cat.to(**tkwargs)

class TransformedCategorical(CategoricalOverlap):
    """
    Second kind of transformed kernel of form:
    $$ k(x, x') = \exp(\frac{\lambda}{n}) \sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)
    or
    $$ k(x, x') = \exp(\frac{1}{n} \sum_{i=1}^n \lambda_i [x_i = x'_i]) $$ if ARD
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', **params):
        # expand x1 and x2 to calc hamming distance
        M1_expanded = x1.unsqueeze(2)
        M2_expanded = x2.unsqueeze(1)

        # calc hamming distance
        diff = (M1_expanded != M2_expanded)

        # (# batch, # batch)
        diff1 = diff
        # diff1 = torch.sum(diff, dim=2)
        # assert 0, (diff.shape, diff1.shape, x1.shape, x2.shape)
        def rbf(d, ard):
            if ard:
                return torch.exp(-torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale))
            else:
                return torch.exp(-self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1])

        def mat52(d, ard):
            raise NotImplementedError

        if exp == 'rbf':
            k_cat = rbf(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        elif exp == 'mat52':
            k_cat = mat52(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        else:
            raise ValueError('Exponentiation scheme %s is not recognised!' % exp)
        if diag:
            return torch.diag(k_cat).to(**tkwargs)
        return k_cat.to(**tkwargs)