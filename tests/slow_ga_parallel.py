import gymnasium
import numpy as np

import mevo

pendulum_env = gymnasium.make(
        'Pendulum-v1',
    )
pendulum_env.reset(seed=1)


def pendulum_fitness_fn(ind: mevo.individual.Individual) -> float:
    ep_r = 0
    for _ in range(2):
        s, _ = pendulum_env.reset()
        for _ in range(150):  # in one episode
            logits = ind.predict(s)
            a = np.tanh(logits) * 2
            s, r, _, _, _ = pendulum_env.step(a)
            ep_r += r
    return ep_r


def main():
    with mevo.GeneticAlgoNet(
            max_size=20,
            layer_size=[3, 32, 1],
            mutation_rate=0.4,
            parallel=True,
            seed=1
    ) as pop:
        for ep in range(400):
            top, kept, _ = pop.evolve(
                fitness_fn=pendulum_fitness_fn,
                drop_rate=0.7,
            )
            print(ep, kept[top.id])

    env = gymnasium.make(
        'Pendulum-v1',
        render_mode="human"
    )
    while True:
        s, _ = env.reset()
        for _ in range(200):  # in one episode
            logits = top.predict(s)
            a = np.tanh(logits) * 2
            s, _, _, _, _ = env.step(a)


if __name__ == "__main__":
    main()
