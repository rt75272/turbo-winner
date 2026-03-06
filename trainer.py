"""
Genetic algorithm trainer — implemented entirely from scratch.

Only Python's built-in ``random`` module is used; no external packages
or libraries are required.

Algorithm
---------
1. **Evaluate** every individual in the population by running a full
   Snake game episode and recording ``get_fitness()``.
2. **Select** parents using tournament selection (top-half pool).
3. **Crossover** two parents with uniform crossover (each gene comes
   from one of the two parents with 50 % probability each).
4. **Mutate** each offspring by adding Gaussian noise to a random
   subset of parameters.
5. **Elitism** — the top ``elite_count`` individuals are copied to the
   next generation unchanged so that good solutions are never lost.
"""

import random


class GeneticAlgorithm:
    """
    Genetic algorithm for evolving neural-network weights.

    Parameters
    ----------
    population_size:
        Number of individuals in the population.
    mutation_rate:
        Probability that any individual parameter is mutated (0.0–1.0).
    mutation_strength:
        Standard deviation of the Gaussian noise added during mutation.
    elite_fraction:
        Fraction of the best individuals carried over unchanged.
    """

    def __init__(
        self,
        population_size: int = 150,
        mutation_rate: float = 0.05,
        mutation_strength: float = 0.3,
        elite_fraction: float = 0.1,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_count = max(2, int(population_size * elite_fraction))

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def crossover(self, parent1, parent2):
        """
        Uniform crossover: each gene is taken from *parent1* or *parent2*
        with equal probability.

        Returns a new NeuralNetwork child.
        """
        p1 = parent1.flatten()
        p2 = parent2.flatten()
        child_params = [
            a if random.random() < 0.5 else b
            for a, b in zip(p1, p2)
        ]
        child = parent1.clone()
        child.unflatten(child_params)
        return child

    def mutate(self, network):
        """
        Apply Gaussian mutation to a random subset of parameters.

        Modifies *network* in-place and returns it.
        """
        params = network.flatten()
        for i in range(len(params)):
            if random.random() < self.mutation_rate:
                params[i] += random.gauss(0.0, self.mutation_strength)
        network.unflatten(params)
        return network

    # ------------------------------------------------------------------
    # Generation step
    # ------------------------------------------------------------------

    def next_generation(self, population: list, fitnesses: list[float]) -> list:
        """
        Build the next generation from the current population and their
        fitness scores.

        Args:
            population: List of NeuralNetwork individuals.
            fitnesses:  Corresponding fitness scores.

        Returns:
            New list of NeuralNetwork individuals.
        """
        # Sort by descending fitness
        paired = sorted(
            zip(fitnesses, population), key=lambda x: x[0], reverse=True
        )
        sorted_pop = [ind for _, ind in paired]

        # Elitism: copy best individuals unchanged
        new_population = [ind.clone() for ind in sorted_pop[: self.elite_count]]

        # Mating pool: top half of the sorted population
        pool_size = max(2, len(sorted_pop) // 2)
        pool = sorted_pop[:pool_size]

        # Fill the rest with crossover + mutation offspring
        while len(new_population) < self.population_size:
            p1, p2 = random.sample(pool, 2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_population.append(child)

        return new_population

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def statistics(fitnesses: list[float]) -> dict:
        """
        Compute basic statistics for the current generation's fitnesses.

        Returns a dict with keys ``'max'``, ``'mean'``, and ``'min'``.
        """
        if not fitnesses:
            return {"max": 0.0, "mean": 0.0, "min": 0.0}
        return {
            "max": max(fitnesses),
            "mean": sum(fitnesses) / len(fitnesses),
            "min": min(fitnesses),
        }
