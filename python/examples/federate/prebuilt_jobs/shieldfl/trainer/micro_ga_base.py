import numpy as np


class MicroGABase:
    def __init__(self, pop_size=15, generations=10, lambda_reg=0.1, seed=0):
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.lambda_reg = float(lambda_reg)
        self.gpu_accelerator = None
        self.rng = np.random.default_rng(int(seed))

    def _init_population(self, num_clients: int) -> np.ndarray:
        pop_list = [np.ones(num_clients) / num_clients]
        for _ in range(4):
            if len(pop_list) >= self.pop_size:
                break
            probe = np.zeros(num_clients)
            num_ones = (
                self.rng.integers(3, min(5, num_clients + 1))
                if num_clients >= 3
                else num_clients
            )
            indices = self.rng.choice(num_clients, num_ones, replace=False)
            probe[indices] = 1.0
            probe = probe / np.sum(probe)
            pop_list.append(probe)
        while len(pop_list) < self.pop_size:
            individual = self.rng.random(num_clients)
            pop_list.append(individual / np.sum(individual))
        return np.array(pop_list[: self.pop_size])

    def _tournament_selection(self, population, fitness_scores, k=2):
        selected = []
        for _ in range(len(population)):
            indices = self.rng.choice(len(population), k, replace=False)
            best_idx = indices[np.argmax(fitness_scores[indices])]
            selected.append(population[best_idx])
        return np.array(selected)

    def _crossover(self, p1, p2):
        beta = self.rng.random()
        child = beta * p1 + (1 - beta) * p2
        child = np.abs(child)
        total = np.sum(child)
        return child / total if total > 1e-9 else np.ones_like(child) / len(child)

    def _mutation(self, individual, sigma=0.05, prob=0.1):
        if self.rng.random() < prob:
            individual = individual + self.rng.normal(0, sigma, size=individual.shape)
        individual = np.abs(individual)
        total = np.sum(individual)
        return individual / total if total > 1e-9 else np.ones_like(individual) / len(individual)
