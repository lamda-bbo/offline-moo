from .callback import HistoryCallback
from .crossover import GGUniformCrossover, SPOrderCrossover
from .mutation import GGShuffleMutation, SPInversionMutation
from .survival import AmateurRankAndCrowdSurvival

REGISTRY = {}

REGISTRY["mutation"] = {"gg": GGShuffleMutation, "sp": SPInversionMutation}

REGISTRY["crossover"] = {"gg": GGUniformCrossover, "sp": SPOrderCrossover}

REGISTRY["callback"] = {
    "gg": HistoryCallback,
}

REGISTRY["survival"] = {
    "gg": AmateurRankAndCrowdSurvival,
}
