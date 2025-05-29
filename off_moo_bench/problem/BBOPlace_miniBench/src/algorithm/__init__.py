REGISTRY = {}

from .bo.bo import BO
from .cma_es.cma_es import CMAES
from .ea.ea import EA
from .ea.nsgaii import NSGAII
from .ea.sa import SA
from .pso.pso import PSO

REGISTRY["ea"] = EA
REGISTRY["bo"] = BO
REGISTRY["sa"] = SA
REGISTRY["cma_es"] = CMAES
REGISTRY["pso"] = PSO
REGISTRY["nsgaii"] = NSGAII
