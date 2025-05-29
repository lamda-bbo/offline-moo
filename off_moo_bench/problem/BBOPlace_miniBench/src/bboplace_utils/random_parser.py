import os
import random

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_state():
    state_dict = {
        "random": random.getstate(),
        "np_random": np.random.get_state(),
    }
    return state_dict


def set_state(state_dict):
    random.setstate(state_dict["random"])
    np.random.set_state(state_dict["np_random"])
