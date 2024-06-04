import numpy as np


def agents(x):
    # ТАм она self._agents
    return np.sum(x ** 2)


def initialize_population(n, dim, lb, ub):
    return lb + (ub - lb) * np.random.rand(n, dim)


def evaluate_population(population):
    return np.apply_along_axis(agents, 1, population)


def update_food_positions(population, fitness, x_best):
    PFS = np.array([x for k, x in enumerate(population) if
                    fitness[k] < fitness.min() or (x_best is not None and np.array_equal(x, x_best))])
    return PFS if len(PFS) > 0 else population


def red_panda_foraging_strategy(population, fitness, PFS, r, I):
    new_population = population.copy()
    for i in range(len(population)):
        selected_food = PFS[np.random.randint(0, len(PFS))]
        x_new = population[i] + r * (selected_food - I * population[i])
        F_new = agents(x_new)
        if F_new < fitness[i]:
            new_population[i] = x_new
    return new_population


def RPO(n, dim, lb, ub, max_iter):
    population = initialize_population(n, dim, lb, ub)
    print(f"POP solution: {population}")
    fitness = evaluate_population(population)
    x_best = population[np.argmin(fitness)]

    r = 0.5  # Example value for r
    I = 0.5  # Example value for I

    for t in range(max_iter):
        PFS = update_food_positions(population, fitness, x_best)
        population = red_panda_foraging_strategy(population, fitness, PFS, x_best, r, I)
        fitness = evaluate_population(population)
        current_best = population[np.argmin(fitness)]
        if agents(current_best) < agents(x_best):
            x_best = current_best

    return x_best, agents(x_best)


# Parameters
n = 50
dim = 10
lb = -10
ub = 10
max_iter = 100

# Run RPO
best_solution, best_fitness = RPO(n, dim, lb, ub, max_iter)
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")