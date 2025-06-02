import unittest
import tensorflow as tf
import numpy as np

from src.modeling.model import WildregressModel
from src.modeling.blocks import Blocks

class TestWildregressModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = (60, 5)
        self.blocks = Blocks()

        # Простейшая конфигурация — один Dense блок
        self.setblockov = [['Dense']]  # один скрытый слой
        self.bot = [[16]]              # параметры слоя (units=16)
        self.bot_pop = [0]*10
        self.bot_pop[7] = 0  # 'linear' активация
        self.bot_pop[9] = [] # признак того, что tiers ещё не сгенерированы

    def test_model_compiles(self):
        model_builder = WildregressModel(input_shape=self.input_shape)
        model = model_builder(bot_pop=self.bot_pop,
                              bot=self.bot,
                              setblockov=self.setblockov,
                              blocks=self.blocks)

        self.assertIsInstance(model, tf.keras.Model)
        model.compile(optimizer='adam', loss='mse')  # проверим, что модель действительно собирается
        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        self.assertGreater(len(summary_str), 0, "Model summary should not be empty")

if __name__ == '__main__':
    unittest.main()
