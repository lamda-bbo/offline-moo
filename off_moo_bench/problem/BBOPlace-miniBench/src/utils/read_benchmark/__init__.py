from .read_aux import read_benchmark as read_aux
from .read_def import read_benchmark as read_def

REGISTRY = {}

REGISTRY["ispd2005"] = read_aux
REGISTRY["iccad2015"] = read_def
