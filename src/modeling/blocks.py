# src/modeling/blocks.py

import tensorflow as tf

class Blocks:
    def __init__(self):
        # Поддерживаемые типы слоёв
        self.net_lays = ['LSTM', 'Dense', 'Conv1D', 'Dropout']
        
        # Кол-во выходов модели
        self.neiro_out = 1

        # Список доступных активаций
        self.activ_out = ['linear', 'relu', 'sigmoid', 'tanh']

        # Служебный объект для redim
        self.set_net = self

    def __buildblock__(self, input_tensor, layer_config, params):
        """
        Построение слоя по конфигурации.
        layer_config: ['LSTM'] или ['Dense'] и т.п.
        params: параметры слоя, например [64] или [32, 0.2]
        """
        if not layer_config:
            return input_tensor
        
        layer_type = layer_config[0]

        if layer_type == 'LSTM':
            units = params[0] if len(params) > 0 else 50
            return tf.keras.layers.LSTM(units=units, return_sequences=False)(input_tensor)

        elif layer_type == 'Dense':
            units = params[0] if len(params) > 0 else 64
            activation = 'relu' if len(params) < 2 else params[1]
            return tf.keras.layers.Dense(units=units, activation=activation)(input_tensor)

        elif layer_type == 'Conv1D':
            filters = params[0] if len(params) > 0 else 32
            kernel_size = params[1] if len(params) > 1 else 3
            return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                          padding='same', activation='relu')(input_tensor)

        elif layer_type == 'Dropout':
            rate = params[0] if len(params) > 0 else 0.3
            return tf.keras.layers.Dropout(rate)(input_tensor)

        else:
            # Если неизвестный слой — возврат входа
            return input_tensor

    def __buildblockout__(self, tensor, bot_pop):
        """
        Финальный слой модели
        """
        activation_idx = bot_pop[7] if isinstance(bot_pop[7], int) and bot_pop[7] < len(self.activ_out) else 0
        return tf.keras.layers.Dense(units=self.neiro_out,
                                     activation=self.activ_out[activation_idx])(tensor)

    def __flatconcat__(self, layers):
        """
        Объединение нескольких выходов в один.
        """
        if len(layers) == 1:
            return layers[0]
        return tf.keras.layers.Concatenate()(layers)

    def __redim__(self, input_dim, total_dims, sort=0):
        """
        Простая функция для reshape — возвращает кортеж с одинаковыми значениями.
        """
        return (input_dim,) * total_dims
