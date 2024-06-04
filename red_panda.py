from src.optimization_algorithm.meta import OptimizationAlgorithmMeta
from hashlib import sha1
import numpy as np
import random


def update_food_positions(population, fitness, x_best):
    pfs = np.array([x for k, x in enumerate(population) if
                    fitness[k] < fitness.min() or (x_best is not None and np.array_equal(x, x_best))])
    return pfs if len(pfs) > 0 else population


class RedPanda(OptimizationAlgorithmMeta):
    def __init__(
            self,
            population,
            function,
            lb,
            ub,
            max_iter,
            no_change_iteration=None,
            nll=3,
            nal=6):
        super().__init__(
            population, function, lb, ub, max_iter, no_change_iteration)

        self.nll = nll
        self.nal = nal
        self.head_leader = None
        self.local_leaders = None
        self.explorer_particles = None
        self.aimless_particles = None

        self.best_positions = {
            sha1(item).digest(): item
            for item in self._agents
        }

        self.ffs = {
            sha1(self._agents[i]).digest(): self._fitness[i]
            for i in range(self.n)
        }

        self.sort_agents()

    def sort_agents(self):
        order = self._sorted_fitness_args
        self._agents = self._agents[order]
        self._fitness = np.array(self._fitness)[order.tolist()]

    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.uniform(0, 1, self.dimension)

    def evaluate_population(self, population):
        return np.apply_along_axis(self._agents, 1, population)

    def red_panda_foraging_strategy(self, population, fitness, pfs, r, I):
        new_population = population.copy()
        for i in range(len(population)):
            selected_food = pfs[np.random.randint(0, len(pfs))]
            x_new = population[i] + r * (selected_food - I * population[i])
            f_new = self._agents(x_new)
            if f_new < fitness[i]:
                new_population[i] = x_new
        return new_population

    def rpo(self, dim, lb, ub, max_iter):
        population = self.initialize_population(self.n, dim, lb, ub)
        print(f"POP solution: {population}")
        fitness = self.evaluate_population(population)
        x_best = population[np.argmin(fitness)]

        r = np.random.uniform(0, 1, self.dimension)
        I = 0.5  # Example value for i

        for t in range(max_iter):
            pfs = update_food_positions(population, fitness, x_best)
            population = self.red_panda_foraging_strategy(population, fitness, pfs, x_best, r, I)
            fitness = self.evaluate_population(population)
            current_best = population[np.argmin(fitness)]
            if self._agents(current_best) < self._agents(x_best):
                x_best = current_best

        return x_best, self._agents(x_best)

    #best_solution, best_fitness = rpo(self, n, dim, lb, ub, max_iter)
    #print(f"Best solution: {best_solution}")
    #print(f"Best fitness: {best_fitness}")

    def _one_iter(self):
        self.initialize_population()
        self.evaluate_population()
        update_food_positions()
        self.red_panda_foraging_strategy()
        self.rpo()
        self.sort_agents()
