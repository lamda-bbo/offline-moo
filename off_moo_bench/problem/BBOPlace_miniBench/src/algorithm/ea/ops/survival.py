import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival, calc_crowding_distance
from pymoo.util.randomized_argsort import randomized_argsort


class AmateurRankAndCrowdSurvival(RankAndCrowdingSurvival):
    def __init__(self, p=0.45, nds=None) -> None:
        super().__init__(nds)
        self.p = p

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if np.random.rand() < self.p:
            F = pop.get("F").astype(float, copy=False)
            fronts = self.nds.do(F)
            survivors = []

            for k in range(len(fronts) - 1, -1, -1):
                front = fronts[k]
                # calculate the crowding distance of the front
                crowding_of_front = calc_crowding_distance(F[front, :])

                # save rank and crowding in the individual class
                for j, i in enumerate(front):
                    pop[i].set("rank", k)
                    pop[i].set("crowding", crowding_of_front[j])

                # current front sorted by crowding distance if splitting
                if len(survivors) + len(front) > n_survive:
                    I = randomized_argsort(
                        crowding_of_front, order="descending", method="numpy"
                    )
                    I = I[: (n_survive - len(survivors))]

                # otherwise take the whole front unsorted
                else:
                    I = np.arange(len(front))

                # extend the survivors by all or selected individuals
                survivors.extend(front[I])
            return pop[survivors]

        else:
            return super()._do(problem, pop, *args, n_survive=n_survive, **kwargs)
