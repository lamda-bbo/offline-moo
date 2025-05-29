import csv
import logging
import os
from abc import abstractmethod

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from bboplace_utils.compute_res import comp_res
from bboplace_utils.constant import get_n_power


class BasicPlacer:
    def __init__(self, args, placedb):
        self.args = args
        self.placedb = placedb

        self.canvas_width = placedb.canvas_width
        self.canvas_height = placedb.canvas_height

        self.fig_save_path = os.path.join(args.result_path, "figures")
        self.placement_save_path = os.path.join(args.result_path, "placements")
        os.makedirs(self.fig_save_path, exist_ok=True)
        os.makedirs(self.placement_save_path, exist_ok=True)

        self.metrics_file = os.path.join(args.result_path, "metrics.csv")
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            header = [
                "n_eval",
                "his_best_hpwl",
                "pop_best_hpwl",
                "pop_avg_hpwl",
                "pop_std_hpwl",
                "t_each_eval",
                "avg_t_each_eval",
            ]
            writer.writerow(header)

        self.placement_saving_lst = []
        self.figure_saving_lst = []
        self.n_max_saving_placement = args.n_max_saving_placement

    def evaluate(self, x):
        macro_pos = self._genotype2phenotype(x)
        hpwl, congestion, regularity = comp_res(macro_pos, self.placedb)
        res = {
            "hpwl": hpwl,
            "congestion": congestion,
            "regularity": regularity,
        }
        return res, macro_pos

    @abstractmethod
    def _genotype2phenotype(self, x):
        pass

    def save_placement(self, macro_pos, n_eval, hpwl):
        logging.info("Placer saving placement")
        scale_hpwl, n_power = get_n_power(hpwl)

        delete_file_name = None
        if len(self.placement_saving_lst) == self.n_max_saving_placement:
            delete_file_name = self.placement_saving_lst.pop(0)

        file_name = os.path.join(
            self.placement_save_path,
            f"{n_eval}_{scale_hpwl:.2f}e{n_power}.{self.args.file_format}",
        )
        type_map = {"pl": self.placedb.to_pl, "def": self.placedb.to_def}
        content = type_map[self.args.file_format](macro_pos)
        with open(file_name, "w") as f:
            f.write(content)

        if delete_file_name is not None:
            os.remove(delete_file_name)
        self.placement_saving_lst.append(file_name)
        assert len(self.placement_saving_lst) <= self.n_max_saving_placement

    def plot(self, macro_pos, n_eval, hpwl):
        logging.info("Placer ploting figure")
        scale_hpwl, n_power = get_n_power(hpwl)

        delete_file_name = None
        if len(self.figure_saving_lst) == self.n_max_saving_placement:
            delete_file_name = self.figure_saving_lst.pop(0)

        file_name = os.path.join(
            self.fig_save_path, f"{n_eval}_{scale_hpwl:.2f}e{n_power}.png"
        )
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="auto")
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        for macro in macro_pos:
            pos_x, pos_y = macro_pos[macro]
            size_x, size_y = (
                self.placedb.node_info[macro]["size_x"],
                self.placedb.node_info[macro]["size_y"],
            )

            pos_x = pos_x / self.placedb.canvas_ux
            pos_y = pos_y / self.placedb.canvas_uy
            size_x = size_x / self.placedb.canvas_ux
            size_y = size_y / self.placedb.canvas_uy
            ax.add_patch(
                patches.Rectangle(
                    (pos_x, pos_y),
                    size_x,
                    size_y,
                    linewidth=1,
                    edgecolor="k",
                    alpha=0.5,
                )
            )

        fig.savefig(file_name, dpi=90, bbox_inches="tight")
        plt.close()

        if delete_file_name is not None:
            os.remove(delete_file_name)
        self.figure_saving_lst.append(file_name)
        assert len(self.figure_saving_lst) <= self.n_max_saving_placement

    def save_metrics(
        self,
        n_eval,
        his_best_hpwl,
        pop_best_hpwl,
        pop_avg_hpwl,
        pop_std_hpwl,
        t_each_eval=0,
        avg_t_each_eval=0,
    ):
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            content = [
                n_eval,
                his_best_hpwl,
                pop_best_hpwl,
                pop_avg_hpwl,
                pop_std_hpwl,
                t_each_eval,
                avg_t_each_eval,
            ]
            writer.writerow(content)

    def _save_checkpoint(self, checkpoint_path):
        def save_and_delete(set_new, set_old, is_placement=True):
            if is_placement:
                suffix = "placements"
            else:
                suffix = "figures"
            for file_name in set_old - set_new:
                os.system(f"rm {os.path.join(checkpoint_path, suffix, file_name)}")

            for file_name in set_new - set_old:
                if is_placement:
                    os.system(
                        f"cp {os.path.join(self.placement_save_path, file_name)} "
                        + f"{os.path.join(checkpoint_path, suffix, file_name)}"
                    )
                else:
                    os.system(
                        f"cp {os.path.join(self.fig_save_path, file_name)} "
                        + f"{os.path.join(checkpoint_path, suffix, file_name)}"
                    )

        os.makedirs(os.path.join(checkpoint_path, "placements"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_path, "figures"), exist_ok=True)

        placement_set_old = set(os.listdir(os.path.join(checkpoint_path, "placements")))
        placement_set_new = set(
            [os.path.basename(file_name) for file_name in self.placement_saving_lst]
        )

        figure_set_old = set(os.listdir(os.path.join(checkpoint_path, "figures")))
        figure_set_new = set(
            [os.path.basename(file_name) for file_name in self.figure_saving_lst]
        )

        save_and_delete(placement_set_new, placement_set_old, is_placement=True)
        save_and_delete(figure_set_new, figure_set_old, is_placement=False)

    def _load_checkpoint(self, checkpoint_path):
        for file_name in os.listdir(os.path.join(checkpoint_path, "placements")):
            self.placement_saving_lst.append(
                os.path.join(self.placement_save_path, file_name)
            )
            os.system(
                f"cp {os.path.join(checkpoint_path, 'placements', file_name)} "
                + f"{os.path.join(self.placement_save_path, file_name)}"
            )

        for file_name in os.listdir(os.path.join(checkpoint_path, "figures")):
            self.figure_saving_lst.append(os.path.join(self.fig_save_path, file_name))
            os.system(
                f"cp {os.path.join(checkpoint_path, 'figures', file_name)} "
                + f"{os.path.join(self.fig_save_path, file_name)}"
            )

        self.placement_saving_lst = sorted(
            self.placement_saving_lst,
            key=lambda x: int(os.path.basename(x).split("_")[0]),
        )
        self.figure_saving_lst = sorted(
            self.figure_saving_lst, key=lambda x: int(os.path.basename(x).split("_")[0])
        )
