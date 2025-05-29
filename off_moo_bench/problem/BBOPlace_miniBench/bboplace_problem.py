import os
import sys
from types import SimpleNamespace

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
from src.bboplace_utils.args_parser import parse_args
from src.evaluator import Evaluator

from off_moo_bench.problem.base import BaseProblem

_support_benchmarks = [
    "adaptec1",
    "adaptec2",
    "adaptec3",
    "adaptec4",
    "bigblue1",
    "bigblue3",
]


class PlacementProblem(BaseProblem):
    def __init__(
        self,
        benchmark_name: str,
        num_cpus: int = 12,
        problem_type: str = "continuous",
        nadir_point=None,
        ideal_point=None,
    ):
        global _support_benchmarks
        assert benchmark_name in _support_benchmarks

        args = SimpleNamespace(
            **{
                "n_cpu": num_cpus,
                "placer": "gg",  # GG in our paper
                "benchmark": f"ispd2005/{benchmark_name}",  # choose which placement benchmark,
            }
        )

        # Read config (i.e. benchmark, placer)
        args = parse_args(args)

        # Instantiate the evaluator
        self.evaluator = Evaluator(args)

        # Read problem metadata
        dim: int = self.evaluator.n_dim
        xl: np.ndarray = self.evaluator.xl
        xu: np.ndarray = self.evaluator.xu
        assert len(xl) == len(xu) == dim

        n_obj = 3
        super().__init__(
            benchmark_name,
            problem_type,
            n_obj=n_obj,
            n_dim=dim,
            nadir_point=nadir_point,
            ideal_point=ideal_point,
            xl=xl,
            xu=xu,
        )

    def evaluate(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        res, _ = self.evaluator.evaluate(x)
        return np.concatenate(
            [
                res["hpwl"].reshape(-1, 1),
                res["congestion"].reshape(-1, 1),
                res["regularity"].reshape(-1, 1),
            ],
            axis=1,
        )

    def get_nadir_point(self):
        return self.nadir_point

    def get_ideal_point(self):
        return self.ideal_point

class Adaptec1(PlacementProblem):
    def __init__(
        self,
    ):
        super(Adaptec1, self).__init__(
            benchmark_name="adaptec1",
            ideal_point=[630906.6249999935, 0.01742868684232235, 3853.1521031844222],
            nadir_point=[1237148.19642858, 0.04877175763249397, 5350.252147698511],
        )


class Adaptec2(PlacementProblem):
    def __init__(
        self,
    ):
        super(Adaptec2, self).__init__(
            benchmark_name="adaptec2",
            ideal_point=[7842726.669642873, 0.2266724407672882, 4901.764783360902],
            nadir_point=[26871833.669643052, 0.7403991222381592, 5712.647749349677],
        )


class Adaptec3(PlacementProblem):
    def __init__(
        self,
    ):
        super(Adaptec3, self).__init__(
            benchmark_name="adaptec3",
            ideal_point=[6204158.750000015, 0.059545550495386124, 9198.968775538116],
            nadir_point=[14316637.705357185, 0.11313248425722122, 11278.06327614315],
        )


class Adaptec4(PlacementProblem):
    def __init__(
        self,
    ):
        super(Adaptec4, self).__init__(
            benchmark_name="adaptec4",
            ideal_point=[6181336.455357182, 0.07117707282304764, 9762.035736836075],
            nadir_point=[10834817.34821432, 0.1190905049443245, 13169.126953862266],
        )


class Bigblue1(PlacementProblem):
    def __init__(
        self,
    ):
        super(Bigblue1, self).__init__(
            benchmark_name="bigblue1",
            ideal_point=[240383.30357142983, 0.005404765717685223, 4237.3058424812725],
            nadir_point=[404658.14285714313, 0.01245042122900486, 6420.407038966832],
        )


class Bigblue3(PlacementProblem):
    def __init__(
        self,
    ):
        super(Bigblue3, self).__init__(
            benchmark_name="bigblue3",
            ideal_point=[6795670.107142902, 0.06121605262160301, 8814.990651400503],
            nadir_point=[22757027.60714277, 0.198087677359581, 11462.956047993546],
        )
