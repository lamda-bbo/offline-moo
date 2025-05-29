REGISTRY = {}

from .gg_placer import GGPlacer
from .sp_placer import SPPlacer

REGISTRY["gg"] = GGPlacer
REGISTRY["sp"] = SPPlacer
