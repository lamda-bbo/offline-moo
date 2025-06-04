import os
import pickle
import numpy as np
from off_moo_bench.problem.base import BaseProblem
from .lambo.tasks.regex import RegexTask as InnerRegexTask

class REGEX(BaseProblem):
    def __init__(self):
        regex_task_instance_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "experiments",
            "test",
            "regex_problem.pkl",
        )
        if not os.path.exists(regex_task_instance_file):
            raise FileNotFoundError(f"Cannot find regex task instance file: {regex_task_instance_file}")
            
        with open(regex_task_instance_file, "rb+") as f:
            self.task_instance = pickle.load(f)
            
        super().__init__(
            name=self.__class__.__name__,
            problem_type="discrete",
            n_obj=self.task_instance.n_obj,
            n_dim=self.task_instance.n_var,
            xl=self.task_instance.xl,
            xu=self.task_instance.xu,
        )
        print(type(self.task_instance))
        self.__dict__.update(self.task_instance.__dict__)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.task_instance.evaluate(X)

    def get_nadir_point(self):
        return np.array([0.64954899, 0.7886475, 0.73789501])

    def get_ideal_point(self):
        return np.array([-4.0, -3.79092841, -4.0])


class RFP(BaseProblem):
    def __init__(self):
        rfp_task_instance_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "experiments",
            "test",
            "proxy_rfp_problem.pkl",
        )
        if not os.path.exists(rfp_task_instance_file):
            raise FileNotFoundError(f"Cannot find RFP task instance file: {rfp_task_instance_file}")
            
        with open(rfp_task_instance_file, "rb+") as f:
            self.task_instance = pickle.load(f)
            
        super().__init__(
            name=self.__class__.__name__,
            problem_type="discrete",
            n_obj=self.task_instance.n_obj,
            n_dim=self.task_instance.n_var,
            xl=self.task_instance.xl,
            xu=self.task_instance.xu,
        )
        self.__dict__.update(self.task_instance.__dict__)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.task_instance.evaluate(X)

    def get_nadir_point(self):
        return np.array([4.0, 4.0])

    def get_ideal_point(self):
        return np.array([-4.0, -1.36930666])


class ZINC(BaseProblem):
    def __init__(self):
        zinc_task_instance_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "experiments",
            "test",
            "zinc_problem.pkl",
        )
        if not os.path.exists(zinc_task_instance_file):
            raise FileNotFoundError(f"Cannot find ZINC task instance file: {zinc_task_instance_file}")
            
        with open(zinc_task_instance_file, "rb+") as f:
            self.task_instance = pickle.load(f)
            
        super().__init__(
            name=self.__class__.__name__,
            problem_type="discrete",
            n_obj=self.task_instance.n_obj,
            n_dim=self.task_instance.n_var,
            xl=self.task_instance.xl,
            xu=self.task_instance.xu,
        )
        print(type(self.task_instance))
        self.__dict__.update(self.task_instance.__dict__)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.task_instance.evaluate(X)

    def get_nadir_point(self):
        return np.array([1.36227612, 2.25588286])

    def get_ideal_point(self):
        return np.array([-2.17846752, -2.77324161])
