import numpy as np
import random
import copy
from src.modeling.model import WildregressModel, evaluate_model
from src.modeling.blocks import Blocks
import tensorflow as tf

def initialize_population(pop_size, gene_template):
    population = []
    for _ in range(pop_size):
        bot_pop = copy.deepcopy(gene_template['bot_pop'])
        bot = copy.deepcopy(gene_template['bot'])
        setblockov = copy.deepcopy(gene_template['setblockov'])

        # случайные мутации
        bot[0][0] = random.choice([16, 32, 64])
        bot_pop[7] = random.randint(0, 3)  # выбор активации
        population.append((bot_pop, bot, setblockov))
    return population

def mutate(bot_pop, bot, mutation_rate=0.1):
    new_bot_pop = copy.deepcopy(bot_pop)
    new_bot = copy.deepcopy(bot)
    if random.random() < mutation_rate:
        new_bot[0][0] = random.choice([16, 32, 64, 128])
    if random.random() < mutation_rate:
        new_bot_pop[7] = random.randint(0, 3)
    return new_bot_pop, new_bot

def crossover(parent1, parent2):
    child_bot_pop = [(a if random.random() < 0.5 else b) for a, b in zip(parent1[0], parent2[0])]
    child_bot = [[(a[0] + b[0]) // 2] for a, b in zip(parent1[1], parent2[1])]
    setblockov = copy.deepcopy(parent1[2])
    return (child_bot_pop, child_bot, setblockov)

def evolve_population(X_train, y_train, X_val, y_val, scaler_y,
                      population_size=6, generations=3, input_shape=None, verbose=True):

    blocks = Blocks()
    gene_template = {
        'bot_pop': [0]*10,
        'bot': [[32]],
        'setblockov': [['Dense']]
    }

    population = initialize_population(population_size, gene_template)

    for gen in range(generations):
        results = []
        if verbose:
            print(f"🔁 Поколение {gen+1}/{generations}")

        for idx, (bot_pop, bot, setblockov) in enumerate(population):
            try:
                model_builder = WildregressModel(input_shape=input_shape)
                model = model_builder(bot_pop, bot, setblockov, blocks)
                val, train_time = evaluate_model(
                    model, scaler_y,
                    X_train, (X_val, y_val),
                    ep=10, verb=0,
                    optimizer=tf.keras.optimizers.Adam(),
                    loss='mse',
                    channels=[0], predict_lag=1,
                    XVAL=X_val, YVAL=y_val
                )
                results.append((val, (bot_pop, bot, setblockov)))
                if verbose:
                    print(f"  [{idx+1}] → val={val:.4f}, time={train_time:.2f}s")
            except Exception as e:
                print(f"  [{idx+1}] ❌ Ошибка: {e}")
                continue

        results.sort(key=lambda x: x[0])  # сортировка по val
        top = results[:2]  # отбор лучших

        # создаём новое поколение
        new_population = [top[0][1], top[1][1]]
        while len(new_population) < population_size:
            parent1 = random.choice(top)[1]
            parent2 = random.choice(top)[1]
            child = crossover(parent1, parent2)
            mutated_child = mutate(*child)
            new_population.append((*mutated_child, child[2]))  # добавляем setblockov
        population = new_population

    best_val, best_genome = results[0]
    return best_genome
