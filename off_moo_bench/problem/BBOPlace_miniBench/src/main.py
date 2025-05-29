import logging
import sys
from types import SimpleNamespace

from algorithm import REGISTRY as ALGO_REGISTRY
from bboplace_utils.args_parser import parse_args
from evaluator import Evaluator
from logger import Logger


def terminal_input():
    params = [arg.lstrip("--") for arg in sys.argv if arg.startswith("--")]

    config_dict = {}
    for arg in params:
        key, value = arg.split('=')
        try:
            config_dict[key] = eval(value)
        except:
            config_dict[key] = value

    args = SimpleNamespace(**config_dict)
    return args

def main(args):
    logger = Logger(args=args)
    evaluator = Evaluator(args=args)
    runner = ALGO_REGISTRY[args.algorithm.lower()](args=args, evaluator=evaluator, logger=logger)
    runner.run()
    logging.info("Exit single run")
    

if __name__ == "__main__":
    args = terminal_input()
    args = parse_args(args)
    main(args)