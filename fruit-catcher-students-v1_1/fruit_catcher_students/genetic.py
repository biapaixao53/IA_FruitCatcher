import random

def create_individual(individual_size):
    return [random.uniform(-2, 2) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.1, mutation_rate=0.1):
    population = generate_population(individual_size, population_size)
    best_individual = None
    best_fitness = float('-inf')

    num_elite = int(population_size * elite_rate)

    for generation in range(generations):
        # Avaliar fitness de cada indivíduo
        fitnesses = [fitness_function(ind) for ind in population]

        # Guardar o melhor global
        for ind, fit in zip(population, fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_individual = ind

        print(f"Geração {generation + 1}/{generations} | Melhor: {best_fitness:.2f} | Média: {sum(fitnesses)/len(fitnesses):.2f}")

        # Verificar paragem antecipada
        if best_fitness >= target_fitness:
            break

        # Elitismo: selecionar os melhores
        elite = [ind for _, ind in sorted(zip(fitnesses, population), reverse=True)[:num_elite]]

        # Nova população: começa com os elites
        new_population = elite.copy()

        # Reproduzir até completar a população
        while len(new_population) < population_size:
            # Seleção por torneio
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Cruzamento (crossover)
            child = crossover(parent1, parent2)

            # Mutação
            child = mutate(child, mutation_rate)

            new_population.append(child)

        population = new_population

    return best_individual, best_fitness # This is expected to be a pair (individual, fitness)

def crossover(p1, p2):
    # Crossover médio
    return [(a + b) / 2 for a, b in zip(p1, p2)]

def mutate(ind, rate):
    strength = 0.2  
    return [
        gene + random.uniform(-strength, strength) if random.random() < rate else gene
        for gene in ind
    ]

def tournament_selection(population, fitnesses, k=2):
    # Seleção por torneio (2 indivíduos aleatórios)
    participants = random.sample(list(zip(population, fitnesses)), k)
    return max(participants, key=lambda x: x[1])[0]