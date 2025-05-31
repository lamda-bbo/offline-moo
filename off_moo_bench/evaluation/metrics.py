import numpy as np
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

from off_moo_bench.task_set import MORL


def hv(nadir_point, y, task_name):
    nadir_point = nadir_point * 2.2  # if task_name not in MORL \
    # else nadir_point * 4
    if task_name == "Molecule-Exact-v0":
        index_to_remove = np.all(y == [1.0, 1.0], axis=1)
        y = y[~index_to_remove]
    return Hypervolume(ref_point=nadir_point).do(y)


def igd(pareto_front, y):
    return IGD(pareto_front).do(y)


def get_SI(data, T, T_max, mode):
    """
    given a algorithm's intermediate result, get the SI index of the optimization curve
    Args:
        save_data: algorithm's intermediate result
        T: number of points on the optimization curve
        mode: how to get the points on the optimization curve, 'max' means get the maximum value, 'min' means get the minimum value, 'median' means get the median value
    """
    # data = get_data(save_data, T, mode)
    if len(data) < T:
        return None
    opt_step = T
    curve = data[:T]
    r = np.array([i for i in range(opt_step)])
    oi_a = curve[0]
    S_d = np.trapz(np.ones_like(curve) * oi_a, r)
    si_a = np.max(curve)
    S_O = np.trapz(np.ones_like(curve) * si_a, r)
    S = np.trapz(curve, r)
    SI = S / S_O

    return SI


def get_OI(data, T, T_max, mode):
    """
    given a algorithm's intermediate result, get the OI index of the optimization curve
    Args:
        save_data: algorithm's intermediate result
        T: number of points on the optimization curve
        mode: how to get the points on the optimization curve, 'max' means get the maximum value, 'min' means get the minimum value, 'median' means get the median value
    """
    # data = get_data(save_data, T, mode)
    if len(data) < T:
        return None
    opt_step = T
    curve = data[:T]
    # curve = [item[0] if isinstance(item, list) else item for item in curve]

    r = np.array([i for i in range(opt_step)])
    oi_a = curve[0]
    S_d = np.trapz(np.ones_like(curve) * oi_a, r)
    si_a = np.max(curve)
    S_O = np.trapz(np.ones_like(curve) * si_a, r)
    S = np.trapz(curve, r)
    OI = S / S_d
    return OI


def get_SO(data, T, T_max, mode):
    """
    given a algorithm's intermediate result, get the SO index of the optimization curve
    Args:
        save_data: algorithm's intermediate result
        T: number of points on the optimization curve
        mode: how to get the points on the optimization curve, 'max' means get the maximum value, 'min' means get the minimum value, 'median' means get the median value
    """
    # data = get_data(save_data, T, mode)
    if len(data) < T:
        return None
    opt_step = T
    curve = data[:T]
    r = np.array([i for i in range(opt_step)])
    oi_a = curve[0]
    S_d = np.trapz(np.ones_like(curve) * oi_a, r)
    si_a = np.max(curve)
    S_O = np.trapz(np.ones_like(curve) * si_a, r)
    S = np.trapz(curve, r)
    OI = S / S_d
    SI = S / S_O
    SO = SI * OI / (0.5 * SI + 0.5 * OI)
    return SO


def get_SOW(data, T, T_max, mode):
    """
    given a algorithm's intermediate result, get the SO index of the optimization curve
    Args:
        save_data: algorithm's intermediate result
        T: number of points on the optimization curve
        mode: how to get the points on the optimization curve, 'max' means get the maximum value, 'min' means get the minimum value, 'median' means get the median value
    """
    # data = get_data(save_data, T, mode)
    if len(data) < T:
        return None
    w = max(0, 1 - T / (T_max + 1))

    opt_step = T
    curve = data[:T]
    r = np.array([i for i in range(opt_step)])
    oi_a = curve[0]
    S_d = np.trapz(np.ones_like(curve) * oi_a, r)
    si_a = np.max(curve)
    S_O = np.trapz(np.ones_like(curve) * si_a, r)
    S = np.trapz(curve, r)
    OI = S / S_d
    SI = S / S_O
    SO = SI * OI / (w * SI + (1 - w) * OI)
    return SO
