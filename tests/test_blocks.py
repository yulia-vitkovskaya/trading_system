# tests/test_blocks.py

import unittest
import tensorflow as tf
import numpy as np
from src.modeling.blocks import Blocks

class TestBlocks(unittest.TestCase):
    def setUp(self):
        self.blocks = Blocks()
        self.sample_input = tf.keras.Input(shape=(60, 5))  # (batch_size, time_steps, features)

    def test_buildblock_lstm(self):
        output = self.blocks.__buildblock__(self.sample_input, ['LSTM'], [32])
        self.assertTrue(hasattr(output, 'shape'))
        self.assertEqual(len(output.shape), 2)

    def test_buildblock_dense(self):
        output = self.blocks.__buildblock__(self.sample_input, ['Dense'], [16])
        self.assertTrue(hasattr(output, 'shape'))
        self.assertEqual(output.shape[-1], 16)

    def test_buildblock_conv1d(self):
        output = self.blocks.__buildblock__(self.sample_input, ['Conv1D'], [8, 3])
        self.assertTrue(hasattr(output, 'shape'))
        self.assertEqual(len(output.shape), 3)

    def test_buildblock_dropout(self):
        output = self.blocks.__buildblock__(self.sample_input, ['Dropout'], [0.5])
        self.assertTrue(hasattr(output, 'shape'))

    def test_flatconcat(self):
        t1 = tf.keras.layers.Dense(4)(self.sample_input)
        t2 = tf.keras.layers.Dense(4)(self.sample_input)
        merged = self.blocks.__flatconcat__([t1, t2])
        self.assertTrue(hasattr(merged, 'shape'))
        self.assertGreaterEqual(int(merged.shape[-1]), 8)

    def test_buildblockout(self):
        t = tf.keras.layers.GlobalAveragePooling1D()(self.sample_input)
        out = self.blocks.__buildblockout__(t, [0]*8)  # bot_pop[7] = 0 → 'linear'
        self.assertTrue(hasattr(out, 'shape'))
        self.assertEqual(out.shape[-1], self.blocks.neiro_out)

    def test_redim(self):
        shape = self.blocks.__redim__(input_dim=10, total_dims=3)
        self.assertEqual(shape, (10, 10, 10))

if __name__ == '__main__':
    unittest.main()
