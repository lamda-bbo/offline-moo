import json
import logging
import os
import pickle

from bboplace_utils.random_parser import get_state


class Logger:
    def __init__(self, args) -> None:
        self.args = args

        # write args
        config_dict = vars(args).copy()
        for key in list(config_dict.keys()):
            if not (
                isinstance(config_dict[key], str)
                or isinstance(config_dict[key], int)
                or isinstance(config_dict[key], float)
            ):
                config_dict.pop(key)
        config_str = json.dumps(config_dict, indent=4)
        with open(os.path.join(args.result_path, "config.json"), "w") as config_file:
            config_file.write(config_str)

        self.log_info = {}
        self.log_checkpoint_info = {}

    def add(self, key, value):
        self.log_info[key] = value
        if key not in self.log_checkpoint_info:
            self.log_checkpoint_info[key] = [value]
        else:
            self.log_checkpoint_info[key].append(value)

    def step(self):
        self.log_info.clear()

    def _save_checkpoint(self, path: str):
        # log checkpoint
        random_state_dict = get_state()
        self.log_checkpoint_info.update(random_state_dict)
        with open(os.path.join(path, "log.pkl"), "wb") as f:
            pickle.dump(self.log_checkpoint_info, f)
