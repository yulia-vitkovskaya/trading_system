import random
import copy
import numpy as np
from src.modeling.model import WildregressModel, evaluate_model
from src.modeling.blocks import Blocks


def initialize_population(size, input_shape):
    population = []
    for _ in range(size):
        bot_pop = [random.randint(0, 1) for _ in range(10)]
        bot = [[random.randint(16, 64)]]
        setblockov = [['Dense']]
        individual = (bot_pop, bot, setblockov)
        population.append(individual)
    return population


def mutate(individual):
    bot_pop, bot, setblockov = copy.deepcopy(individual)
    index = random.randint(0, len(bot_pop) - 1)
    bot_pop[index] = 1 - bot_pop[index]
    if random.random() < 0.5:
        bot[0][0] = max(8, min(128, bot[0][0] + random.randint(-8, 8)))
    return (bot_pop, bot, setblockov)


def crossover(parent1, parent2):
    child_bot_pop = [(a if random.random() < 0.5 else b) for a, b in zip(parent1[0], parent2[0])]
    child_bot = [[(a[0] + b[0]) // 2] for a, b in zip(parent1[1], parent2[1])]
    setblockov = copy.deepcopy(parent1[2])
    return (child_bot_pop, child_bot, setblockov)


def evolve_population(X_train, y_train, X_val, y_val, scaler_y,
                      population_size, generations, input_shape, verbose=False):
    population = initialize_population(population_size, input_shape)
    blocks = Blocks()

    for generation in range(generations):
        results = []
        for individual in population:
            try:
                model = WildregressModel(input_shape=input_shape)(*individual, blocks)
                model.compile(optimizer='adam', loss='mse')
                val_loss, _ = evaluate_model(model, scaler_y,
                                             train_gen=(X_train, y_train),
                                             val_gen=(X_val, y_val),
                                             ep=3,
                                             verb=0,
                                             optimizer=tf.keras.optimizers.Adam(),
                                             loss='mse',
                                             channels=[0],
                                             predict_lag=1,
                                             XVAL=X_val,
                                             YVAL=y_val)
            except Exception:
                val_loss = np.inf
            results.append((val_loss, individual))

        top = sorted(results, key=lambda x: x[0])[:2]

        # fallback: если меньше двух
        while len(top) < 2:
            bot_pop = [random.randint(0, 1) for _ in range(10)]
            bot = [[random.randint(16, 64)]]
            setblockov = [['Dense']]
            top.append((np.inf, (bot_pop, bot, setblockov)))

        new_population = [top[0][1], top[1][1]]

        # Оставшиеся особи — мутации и кроссоверы
        while len(new_population) < population_size:
            if random.random() < 0.5:
                new_population.append(mutate(random.choice(top)[1]))
            else:
                new_population.append(crossover(*random.sample([t[1] for t in top], 2)))

        population = new_population

        if verbose:
            print(f"✅ Generation {generation + 1} best loss: {top[0][0]:.4f}")

    return top[0][1]  # (bot_pop, bot, setblockov)
