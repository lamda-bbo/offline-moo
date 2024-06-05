import os
import numpy as np

from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

def hv(nadir_point, y):
    nadir_point = nadir_point * 2
    return Hypervolume(ref_point=nadir_point).do(y)

def igd(pareto_front, y):
    return IGD(pareto_front).do(y)