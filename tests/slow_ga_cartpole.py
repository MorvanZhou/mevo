import gymnasium
import numpy as np

import mevo


# define a fitness function to get fitness for every individual
def fitness_fn(ind: mevo.individuals.Individual, conf: dict) -> float:
    ep_r = 0
    env = gymnasium.make('CartPole-v1')
    for _ in range(2):
        s, _ = env.reset()
        for _ in range(500):  # in one episode
            logits = ind.predict(s)
            a = np.argmax(logits)
            s, r, done, _, _ = env.step(a)
            ep_r += r
            if done:
                break
    return ep_r


if __name__ == "__main__":
    # training
    with mevo.GeneticAlgoNet(max_size=20, layer_size=[4, 8, 2], drop_rate=0.7, mutate_rate=0.5, n_worker=-1) as pop:
        for generation in range(40):
            pop.evolve(fitness_fn=fitness_fn)
            print(f"generation={generation}, top_fitness={pop.top.fitness:.2f}")

    # deploy the best individual
    env = gymnasium.make('CartPole-v1', render_mode="human")
    while True:
        s, _ = env.reset()
        while True:  # in one episode
            logits = pop.top.predict(s)
            a = np.argmax(logits)
            s, _, done = env.step(a)[:3]
            if done:
                break
