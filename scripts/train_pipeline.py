import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from sklearn.preprocessing import MinMaxScaler
from src.data_processing.loader import get_sp500_tickers, choose_random_ticker, download_stock_data
from src.data_processing.preprocessor import DataPreprocessor
from src.modeling.model import WildregressModel
from src.modeling.blocks import Blocks
from src.modeling.evolution import evolve_population
import warnings
warnings.filterwarnings("ignore")


def create_sequences(X, y, window):
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i-window:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def parse_args():
    parser = argparse.ArgumentParser(description="📈 Обучение модели с нейроэволюцией")
    parser.add_argument('--ticker', type=str, default=None, help='Тикер из S&P 500 (например, AAPL). Если не задан, выбирается случайно.')
    parser.add_argument('--window', type=int, default=60, help='Размер окна для обучения модели')
    parser.add_argument('--generations', type=int, default=2, help='Количество поколений для эволюции')
    parser.add_argument('--population', type=int, default=4, help='Размер популяции в каждом поколении')
    parser.add_argument('--epochs', type=int, default=15, help='Эпохи обучения финальной модели')
    return parser.parse_args()


def main():
    args = parse_args()

    # Загрузка данных
    tickers = get_sp500_tickers()
    ticker = args.ticker if args.ticker in tickers else choose_random_ticker(tickers)
    df = download_stock_data(ticker, start='2018-01-01')

    # Предобработка
    df = DataPreprocessor.clean_dataset(df)
    df = DataPreprocessor.add_all_indicators(df, windows=[5, 10], indicators=['Close'])

    features = df.drop(columns=['Close'])
    target = df[['Close']]

    # Деление по времени
    split_index = int(len(df) * 0.8)
    features_train, features_val = features[:split_index], features[split_index:]
    target_train, target_val = target[:split_index], target[split_index:]

    # Масштабирование
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(features_train)
    X_val_scaled = scaler_x.transform(features_val)
    y_train_scaled = scaler_y.fit_transform(target_train)
    y_val_scaled = scaler_y.transform(target_val)

    # Создание окон
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, args.window)
    X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, args.window)

    # Нейроэволюция для поиска архитектуры
    blocks = Blocks()
    input_shape = X_train.shape[1:]

    best_bot_pop, best_bot, best_setblockov = evolve_population(
        X_train, y_train,
        X_val, y_val,
        scaler_y,
        population_size=args.population,
        generations=args.generations,
        input_shape=input_shape,
        verbose=True
    )

    builder = WildregressModel(input_shape=input_shape)
    model = builder(best_bot_pop, best_bot, best_setblockov, blocks)

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=args.epochs,
              batch_size=32,
              verbose=1)

    # Прогноз
    y_pred = model.predict(X_val).reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_val_rescaled = scaler_y.inverse_transform(y_val)

    # Визуализация
    plt.figure(figsize=(15, 5))
    plt.plot(y_val_rescaled, label='Истинные значения', linewidth=2)
    plt.plot(y_pred_rescaled, label='Прогноз модели', linestyle='--')
    plt.title(f'📈 Прогноз vs Истина для {ticker}')
    plt.xlabel("Временной шаг")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_vs_actual.png")
    print("✅ График сохранён в prediction_vs_actual.png")


if __name__ == "__main__":
    main()
