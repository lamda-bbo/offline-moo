import os
import numpy as np
import torch 
import abc

from problem import get_problem

class OracleBuilder(abc.ABC):
    def __init__(self,):
        pass 