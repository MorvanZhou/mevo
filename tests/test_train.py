import unittest

import gymnasium
import numpy as np

import mevo

SHOW = False

pendulum_env = gymnasium.make(
        'Pendulum-v1',
    )
pendulum_env.reset(seed=1)


def pendulum_fitness_fn(ind: mevo.individuals.Individual, conf: dict) -> float:
    ep_r = 0
    pendulum_env.reset(seed=conf.get("seed"))
    for _ in range(2):
        s, _ = pendulum_env.reset()
        for _ in range(100):  # in one episode
            logits = ind.predict(s)
            a = np.tanh(logits) * 2
            s, r, _, _, _ = pendulum_env.step(a)
            ep_r += r
    return ep_r


class GATest(unittest.TestCase):

    def test_check_function(self):
        def f(ind):
            return 1

        with self.assertRaises(ValueError) as cm:
            pop = mevo.population.GeneticAlgoInt(
                max_size=2,
                chromo_size=10,
                drop_rate=0.3,
                chromo_initializer=mevo.chromosomes.initializers.RandomInt(0, 2),
                mutate_rate=0.01,
            )
            pop.evolve(f)
            self.assertEqual(
                    "cannot find 'ind' or 'conf' argument in the definition of fitness function", str(cm.exception))

    def test_ga_int(self):
        def wave_fitness_fn(ind: mevo.individuals.Individual, conf: dict) -> float:
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
            drop_rate=0.3,
            chromo_initializer=mevo.chromosomes.initializers.RandomInt(0, 2),
            mutate_rate=0.01,
        )

        for _ in range(5):
            drop_num = max(1, int(pop.max_size * pop.drop_rate))
            # selection
            pop.filter(
                fitness_fn=wave_fitness_fn,
                drop_num=drop_num,
            )
            self.assertEqual(max_size, len(pop.members) + drop_num)
            # reproduce
            pop.reproduce(n_children=drop_num)
            # print(ep, pop.top.fitness)

    def test_ga_order(self):
        positions = [np.random.rand(2) for _ in range(20)]

        def distance_fitness_fn(ind: mevo.individuals.Individual, conf: dict) -> float:
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
            drop_rate=0.3,
            mutate_rate=0.01,
            seed=1
        )
        for _ in range(5):
            drop_num = max(1, int(pop.max_size * pop.drop_rate))
            # selection
            pop.filter(
                fitness_fn=distance_fitness_fn,
                drop_num=drop_num,
            )
            self.assertEqual(max_size, len(pop.members) + drop_num)
            # reproduce
            pop.reproduce(n_children=drop_num)
            # print(ep, pop.top.fitness)

    def test_ga_cartpole(self):
        env = gymnasium.make(
            'CartPole-v1',
            # render_mode="human"
        )
        env.reset(seed=1)

        def fitness_fn(ind: mevo.individuals.Individual, conf: dict) -> float:
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
            drop_rate=0.5,
            seed=1
        )
        for _ in range(5):
            drop_num = max(1, int(pop.max_size * pop.drop_rate))
            # selection
            pop.filter(
                fitness_fn=fitness_fn,
                drop_num=drop_num,
            )
            self.assertEqual(max_size, len(pop.members) + drop_num)
            # reproduce
            pop.reproduce(n_children=drop_num)
            # print(ep, pop.top.fitness)

        if SHOW:
            env = gymnasium.make(
                'CartPole-v1',
                render_mode="human"
            )
            while True:
                s, _ = env.reset()
                for _ in range(500):  # in one episode
                    logits = pop.top.predict(s)
                    a = np.argmax(logits)
                    s, _, done, _, _ = env.step(a)
                    if done:
                        break

    def ga_pendulum(self, parallel, step):
        max_size = 40
        with mevo.GeneticAlgoNet(
            max_size=max_size,
            layer_size=[3, 32, 1],
            drop_rate=0.7,
            parallel=parallel,
            seed=1
        ) as pop:
            for _ in range(step):
                drop_num = max(1, int(pop.max_size * pop.drop_rate))
                # selection
                pop.filter(
                    fitness_fn=pendulum_fitness_fn,
                    drop_num=drop_num,
                )
                self.assertEqual(max_size, len(pop.members) + drop_num)
                # reproduce
                pop.reproduce(n_children=drop_num)
                # print(ep, pop.top.fitness)

        if SHOW:
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

    def test_ga_pendulum(self):
        self.ga_pendulum(parallel=False, step=3)

    def test_ga_pendulum_parallel(self):
        pop = mevo.population.GeneticAlgoNet(
            max_size=5,
            layer_size=[3, 32, 1],
            mutate_rate=0.5,
            parallel=True,
        )
        with self.assertRaises(RuntimeError) as cm:
            pop.evolve(fitness_fn=pendulum_fitness_fn)
            self.assertIn(
                "RuntimeError: if run on parallel mode, please use with statement. For example",
                str(cm.exception))

        self.ga_pendulum(parallel=True, step=3)


def pendulum_es_fitness_fn(ind: mevo.individuals.EvolutionStrategyDense, conf: dict) -> float:
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


class ESTest(unittest.TestCase):
    def test_es(self):
        max_size = 40
        with mevo.EvolutionStrategyNet(
                max_size=max_size,
                layer_size=[3, 32, 1],
                mutate_strength=0.7,
                parallel=True,
                seed=1
        ) as pop:
            for _ in range(3):
                # selection
                pop.evolve(
                    fitness_fn=pendulum_es_fitness_fn,
                )
                # print(ep, pop.top.fitness)

        if SHOW:
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
