import os
import json
import joblib
import argparse
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from src.data_processing.loader import get_sp500_tickers, choose_random_ticker, download_stock_data
from src.data_processing.processor import create_features_targets, create_sequences
from src.modeling.model import WildregressModel
from src.modeling.blocks import Blocks
from src.modeling.evolution import evolve_population


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    ticker = args.ticker or choose_random_ticker(get_sp500_tickers())
    df = download_stock_data(ticker, start="2018-01-01")

    features, target = create_features_targets(df)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    features_scaled = scaler_x.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target)

    X, y = create_sequences(features_scaled, target_scaled, args.window)

    # Эволюция
    best_bot_pop, best_bot, best_setblockov = evolve_population(
        input_shape=X.shape[1:],
        X=X,
        y=y,
        generations=args.generations,
        population_size=args.population,
        ep=args.epochs
    )

    # Сборка лучшей модели
    blocks = Blocks()
    model = WildregressModel(input_shape=X.shape[1:])(best_bot_pop, best_bot, best_setblockov, blocks)

    # Сохранение модели и параметров
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_{ticker}_{timestamp}"
    save_path = f"models/{model_name}"
    os.makedirs(save_path, exist_ok=True)

    model.save(f"{save_path}/model.h5")
    joblib.dump(scaler_y, f"{save_path}/scaler_y.pkl")

    metadata = {
        "ticker": ticker,
        "window": args.window,
        "generations": args.generations,
        "population": args.population,
        "epochs": args.epochs,
        "bot_pop": best_bot_pop,
        "bot": best_bot,
        "setblockov": best_setblockov
    }
    with open(f"{save_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Модель сохранена в {save_path}/")


if __name__ == "__main__":
    main()
