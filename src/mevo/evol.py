import typing as tp
from queue import PriorityQueue

from mevo.individual.base import Individual
from mevo.population.base import Population


def evolve(
        pop: Population,
        fitness_fn: tp.Callable[[Individual], float],
        drop_ratio: float = 0.4,
) -> tp.Tuple[Individual, tp.Dict[str, float], tp.Dict[str, float]]:
    # selection
    if drop_ratio >= 1. or drop_ratio <= 0:
        raise ValueError(f"drop_ratio must in range of (0, 1), but got {drop_ratio}")
    q = PriorityQueue(maxsize=len(pop.members))
    kept = {}
    for iid, ind in pop.members.items():
        fitness = fitness_fn(ind)
        q.put((fitness, iid))
        kept[iid] = fitness

    drop_num = max(1, int(len(pop.members) * drop_ratio))
    dropped = {}
    for _ in range(drop_num):
        fitness, iid = q.get()
        dropped[iid] = kept.pop(iid)
        pop.remove_by_id(iid)

    while q.qsize() > 1:
        q.get_nowait()
    iid = q.get()[1]
    top = pop.members[iid]

    # reproduce
    pop.reproduce(n_children=drop_num)
    return top, kept, dropped
