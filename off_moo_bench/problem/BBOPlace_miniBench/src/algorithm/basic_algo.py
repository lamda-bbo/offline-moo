import logging
import os
import pickle
from abc import abstractmethod

import numpy as np
from bboplace_utils.constant import INF
from bboplace_utils.debug import *
from bboplace_utils.random_parser import set_state


class BasicAlgo:
    def __init__(self, args, evaluator, logger) -> None:
        self.args = args
        self.evaluator = evaluator
        self.logger = logger

        self.n_eval = 0
        self.population = None
        self.best_hpwl = INF

        self.t_total = 0
        self.max_eval_time_second = args.max_eval_time * 60 * 60

        self.checkpoint_path = os.path.join(args.result_path, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)

    @abstractmethod
    def run(self):
        pass

    def _record_results(self, hpwl, macro_pos_all, t_each_eval=0, avg_t_each_eval=0):
        hpwl = hpwl.flatten()
        best_idx = np.argmin(hpwl)
        pop_best_hpwl = hpwl[best_idx]
        pop_avg_hpwl = np.mean(hpwl)
        pop_std_hpwl = np.std(hpwl)

        for h, m_pos in zip(hpwl, macro_pos_all):
            self.n_eval += 1
            if self.n_eval > self.args.max_evals:
                break
            if h < self.best_hpwl:
                self.best_hpwl = h
                logging.info(f"n_eval: {self.n_eval}\tbest_hpwl: {self.best_hpwl}")
                self.evaluator.placer.save_placement(
                    macro_pos=m_pos, n_eval=self.n_eval, hpwl=h
                )
                self.evaluator.placer.plot(macro_pos=m_pos, n_eval=self.n_eval, hpwl=h)

            self.logger.add("HPWL/his_best", self.best_hpwl)
            self.logger.add("HPWL/pop_best", pop_best_hpwl)
            self.logger.add("HPWL/pop_avg", pop_avg_hpwl)
            self.logger.add("HPWL/pop_std", pop_std_hpwl)
            self.logger.add("Time/each_eval", t_each_eval)
            self.logger.add("Time/avg_each_eval", avg_t_each_eval)
            self.logger.step()

            self.evaluator.placer.save_metrics(
                n_eval=self.n_eval,
                his_best_hpwl=self.best_hpwl,
                pop_best_hpwl=pop_best_hpwl,
                pop_avg_hpwl=pop_avg_hpwl,
                pop_std_hpwl=pop_std_hpwl,
                t_each_eval=t_each_eval,
                avg_t_each_eval=avg_t_each_eval,
            )

    def _save_checkpoint(self):
        logging.info("saving checkpoint")

        # logger checkpoint
        self.logger._save_checkpoint(path=self.checkpoint_path)

        # placement and corresponding figure checkpoint
        self.evaluator.placer._save_checkpoint(checkpoint_path=self.checkpoint_path)

        if self.t_total >= self.max_eval_time_second:
            logging.info(
                f"Reaching maximun running time ({self.t_total:.2f} >= {self.max_eval_time_second}), the program will exit"
            )
            exit(0)

    def _load_checkpoint(self):
        if hasattr(self.args, "checkpoint") and os.path.exists(self.args.checkpoint):
            logging.info(f"Loading checkpoint from {self.args.checkpoint}")
            log_file = os.path.join(self.args.checkpoint, "log.pkl")
            with open(log_file, "rb") as log_f:
                log_data = pickle.load(log_f)

            self.n_eval = len(log_data["HPWL/his_best"])
            assert self.n_eval == len(log_data["HPWL/pop_best"])
            assert self.n_eval == len(log_data["HPWL/pop_avg"])
            assert self.n_eval == len(log_data["HPWL/pop_std"])
            assert self.n_eval == len(log_data["Time/each_eval"])
            assert self.n_eval == len(log_data["Time/avg_each_eval"])

            set_state(log_data)

            for i_eval in range(0, self.n_eval):
                self.logger.add("HPWL/his_best", log_data["HPWL/his_best"][i_eval])
                self.logger.add("HPWL/pop_best", log_data["HPWL/pop_best"][i_eval])
                self.logger.add("HPWL/pop_avg", log_data["HPWL/pop_avg"][i_eval])
                self.logger.add("HPWL/pop_std", log_data["HPWL/pop_std"][i_eval])
                self.logger.add("Time/each_eval", log_data["Time/each_eval"][i_eval])
                self.logger.add(
                    "Time/avg_each_eval", log_data["Time/avg_each_eval"][i_eval]
                )
                self.logger.step()

                self.evaluator.placer.save_metrics(
                    n_eval=i_eval + 1,
                    his_best_hpwl=log_data["HPWL/his_best"][i_eval],
                    pop_best_hpwl=log_data["HPWL/pop_best"][i_eval],
                    pop_avg_hpwl=log_data["HPWL/pop_avg"][i_eval],
                    pop_std_hpwl=log_data["HPWL/pop_std"][i_eval],
                    t_each_eval=log_data["Time/each_eval"][i_eval],
                    avg_t_each_eval=log_data["Time/avg_each_eval"][i_eval],
                )
            self.best_hpwl = log_data["HPWL/his_best"][self.n_eval - 1]
            self.t_total = sum(log_data["Time/each_eval"])

            self.evaluator.placer._load_checkpoint(checkpoint_path=self.args.checkpoint)
