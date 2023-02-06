import gymnasium
import numpy as np

import mevo

pendulum_env = gymnasium.make(
        'Pendulum-v1',
    )
pendulum_env.reset(seed=1)


def pendulum_fitness_fn(ind: mevo.individuals.EvolutionStrategyDense, conf: dict) -> float:
    ep_r = 0
    seed = conf["seed"]
    index = conf["index"]

    pendulum_env.reset(seed=seed)
    c_ind = ind.clone_with_mutate(index, seed)

    for _ in range(2):
        s, _ = pendulum_env.reset()
        for _ in range(100):  # in one episode
            logits = c_ind.predict(s)
            a = np.tanh(logits) * 2
            s, r, _, _, _ = pendulum_env.step(a)
            ep_r += r
    return ep_r


def main():
    with mevo.EvolutionStrategyNet(
            max_size=15,
            layer_size=[3, 32, 1],
            mutate_strength=0.05,
            learning_rate=0.1,
            parallel=True,
            seed=2
    ) as pop:
        for ep in range(700):
            pop.evolve(fitness_fn=pendulum_fitness_fn)
            print(ep, pop.top.fitness)

    env = gymnasium.make(
        'Pendulum-v1',
        render_mode="human"
    )
    while True:
        s, _ = env.reset()
        for _ in range(200):  # in one episode
            logits = pop.top.predict(s)
            a = np.tanh(logits) * 2
            s, _, _, _, _ = env.step(a)


if __name__ == "__main__":
    main()
