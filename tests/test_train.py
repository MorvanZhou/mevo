import unittest

import gymnasium
import numpy as np

import mevo

SHOW = False

pendulum_env = gymnasium.make(
        'Pendulum-v1',
    )
pendulum_env.reset(seed=1)


def pendulum_fitness_fn(ind: mevo.individual.Individual) -> float:
    ep_r = 0
    for _ in range(2):
        s, _ = pendulum_env.reset()
        for _ in range(100):  # in one episode
            logits = ind.predict(s)
            a = np.tanh(logits) * 2
            s, r, _, _, _ = pendulum_env.step(a)
            ep_r += r
    return ep_r


class GATest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(7)

    def test_ga_int(self):
        def wave_fitness_fn(ind: mevo.individual.Individual) -> float:
            binary = []
            for c in ind.chromosomes:
                binary.append(c.data)

            c = np.concatenate(binary, axis=0)
            a = 2 ** np.arange(len(c))[::-1]
            decimal = c.dot(a)

            x = decimal / float(2 ** len(c) - 1) * 5
            o = np.sin(10 * x) * x + np.cos(2 * x) * x
            return o

        max_size = 20
        pop = mevo.population.GeneticAlgoInt(
            max_size=max_size,
            chromo_size=10,
            chromo_initializer=mevo.chromosome.initializer.RandomInt(0, 2),
            mutation_rate=0.01,
        )

        for _ in range(5):
            _, kept, dropped = pop.evolve(fitness_fn=wave_fitness_fn, drop_rate=0.3)
            self.assertLessEqual(max(list(dropped.values())), min(list(kept.values())))
            self.assertEqual(max_size, len(kept) + len(dropped))

    def test_ga_order(self):
        positions = [np.random.rand(2) for _ in range(20)]

        def distance_fitness_fn(ind: mevo.individual.Individual) -> float:
            order = []
            for c in ind.chromosomes:
                order.append(c.data[0])

            cost = 0
            for i in range(len(order) - 1):
                p1 = positions[order[i]]
                p2 = positions[order[i + 1]]
                cost += np.square(p1 - p2).sum()
            fitness = -cost
            return fitness

        max_size = 50
        pop = mevo.population.GeneticAlgoOrder(
            max_size=max_size,
            chromo_size=len(positions),
            mutation_rate=0.01,
            seed=1
        )
        for _ in range(5):
            _, kept, dropped = pop.evolve(fitness_fn=distance_fitness_fn, drop_rate=0.3)
            # print(kept[top.id])
            self.assertLessEqual(max(list(dropped.values())), min(list(kept.values())))
            self.assertEqual(max_size, len(kept) + len(dropped))

    def test_ga_cartpole(self):
        env = gymnasium.make(
            'CartPole-v1',
            # render_mode="human"
        )
        env.reset(seed=1)

        def fitness_fn(ind: mevo.individual.Individual) -> float:
            s, _ = env.reset()
            r = 0
            for _ in range(200):  # in one episode
                logits = ind.predict(s)
                a = np.argmax(logits)
                s, _, done, _, _ = env.step(a)
                r += 1

                if done:
                    break
            return r

        max_size = 20
        pop = mevo.GeneticAlgoNet(
            max_size=max_size,
            layer_size=[4, 8, 2],
            mutation_rate=0.01,
            seed=1
        )
        for ep in range(5):
            top, kept, dropped = pop.evolve(fitness_fn=fitness_fn, drop_rate=0.5)
            print(ep, kept[top.id])
            self.assertLessEqual(max(list(dropped.values())), min(list(kept.values())))
            self.assertEqual(max_size, len(kept) + len(dropped))

        if SHOW:
            env = gymnasium.make(
                'CartPole-v1',
                render_mode="human"
            )
            while True:
                s, _ = env.reset()
                for _ in range(500):  # in one episode
                    logits = top.predict(s)
                    a = np.argmax(logits)
                    s, _, done, _, _ = env.step(a)
                    if done:
                        break

    def ga_pendulum(self, parallel, step):
        max_size = 40
        with mevo.GeneticAlgoNet(
            max_size=max_size,
            layer_size=[3, 32, 1],
            mutation_rate=0.5,
            parallel=parallel,
            seed=1
        ) as pop:
            for _ in range(step):
                top, kept, dropped = pop.evolve(
                    fitness_fn=pendulum_fitness_fn,
                    drop_rate=0.7,
                )
                # print(ep, kept[top.id])
                self.assertLessEqual(max(list(dropped.values())), min(list(kept.values())))
                self.assertEqual(max_size, len(kept) + len(dropped))

        if SHOW:
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

    def test_ga_pendulum(self):
        self.ga_pendulum(parallel=False, step=2)

    def test_ga_pendulum_parallel(self):
        pop = mevo.population.GeneticAlgoNet(
            max_size=5,
            layer_size=[3, 32, 1],
            mutation_rate=0.5,
            parallel=True,
        )
        with self.assertRaises(RuntimeError) as cm:
            pop.evolve(fitness_fn=pendulum_fitness_fn)
            self.assertIn(
                "RuntimeError: if run on parallel mode, please use with statement. For example",
                str(cm.exception))

        self.ga_pendulum(parallel=True, step=3)
