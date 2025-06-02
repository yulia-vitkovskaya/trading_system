import pytest
from src.modeling.evolution import initialize_population, mutate, crossover

# Шаблон для тестов
gene_template = {
    'bot_pop': [0]*10,
    'bot': [[32]],
    'setblockov': [['Dense']]
}

def test_initialize_population():
    pop = initialize_population(pop_size=5, gene_template=gene_template)
    assert len(pop) == 5
    for bot_pop, bot, setblockov in pop:
        assert isinstance(bot_pop, list)
        assert isinstance(bot, list)
        assert isinstance(setblockov, list)
        assert len(bot_pop) == 10
        assert isinstance(bot[0][0], int)

def test_mutate_changes_values():
    original_bot_pop = [0]*10
    original_bot = [[32]]
    mutated_bot_pop, mutated_bot = mutate(original_bot_pop, original_bot, mutation_rate=1.0)
    # 100% мутация должна изменить значения
    assert mutated_bot != original_bot or mutated_bot_pop != original_bot_pop

def test_crossover_combines_parents():
    parent1 = ([0]*10, [[32]], [['Dense']])
    parent2 = ([1]*10, [[64]], [['Dense']])
    child = crossover(parent1, parent2)
    assert len(child) == 3
    assert isinstance(child[0], list)
    assert isinstance(child[1], list)
    assert isinstance(child[2], list)
