import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.modeling.model import evaluate_model, WildregressModel
from src.modeling.blocks import Blocks

def test_evaluate_model_runs():
    # Генерация искусственных данных
    np.random.seed(42)
    X = np.random.rand(100, 60, 5)  # 100 примеров, окно 60, 5 признаков
    y = np.random.rand(100, 1)

    # Делим на обучение и валидацию
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    # Масштабирование
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # Сборка модели
    blocks = Blocks()
    input_shape = X.shape[1:]
    setblockov = [['Dense']]
    bot = [[32]]
    bot_pop = [0]*10
    bot_pop[7] = 0
    bot_pop[9] = []

    builder = WildregressModel(input_shape=input_shape)
    model = builder(bot_pop, bot, setblockov, blocks)

    val, train_time = evaluate_model(
        model, scaler_y,
        train_gen=(X_train, y_train_scaled),
        val_gen=(X_val, y_val_scaled),
        ep=2,
        verb=0,
        optimizer=tf.keras.optimizers.Adam(),
        loss='mse',
        channels=[0],
        predict_lag=1,
        XVAL=X_val,
        YVAL=y_val_scaled
    )

    val = float(val)
    assert isinstance(val, float)
    assert val >= 0
    assert train_time > 0
