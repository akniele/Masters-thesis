from tqdm import tqdm
import pickle

with open(f"train_data/train_big_100000_{i}.pkl", "rb") as f:
    probs1 = pickle.load(f)
probs1 = probs1.numpy()

with open(f"train_data/train_small_100000_{i}.pkl", "rb") as g:
    probs0 = pickle.load(f)
probs0 = probs0.numpy()


score = transform_and_score_probs(probs1, probs0, num_buckets, mean_entropy)



import random


# define the fitness function to optimize
def fitness_function(params):
    # TODO: replace with your own fitness function
    return sum(params)


# define the genetic algorithm function
def genetic_algorithm(population_size, num_generations, mutation_rate, params_min, params_max):
    # create an initial population of random parameter sets
    population = []
    for i in range(population_size):
        params = [random.uniform(params_min[j], params_max[j]) for j in range(len(params_min))]
        population.append(params)

    # loop over generations
    for generation in range(num_generations):
        # evaluate fitness for each member of the population
        fitness_scores = [fitness_function(params) for params in population]

        # select the fittest members of the population to mate
        mating_pool = []
        for i in range(population_size):
            parent_a = random.choices(population, weights=fitness_scores)[0]
            parent_b = random.choices(population, weights=fitness_scores)[0]
            mating_pool.append((parent_a, parent_b))

        # create a new generation of individuals by mating and mutating
        new_population = []
        for parents in mating_pool:
            child = []
            for i in range(len(parents[0])):
                if random.random() < mutation_rate:
                    child.append(random.uniform(params_min[i], params_max[i]))
                else:
                    if random.random() < 0.5:
                        child.append(parents[0][i])
                    else:
                        child.append(parents[1][i])
            new_population.append(child)

        # replace the old population with the new population
        population = new_population

    # return the fittest member of the final population
    best_params = max(population, key=fitness_function)
    return best_params


