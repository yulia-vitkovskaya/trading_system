import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from src.data_processing.loader import get_sp500_tickers, choose_random_ticker, download_stock_data
from src.data_processing.preprocessor import DataPreprocessor
from src.modeling.model import WildregressModel
from src.modeling.blocks import Blocks
from src.modeling.evolution import evolve_population
from src.modeling.inference_utils import load_saved_model, predict_with_model, evaluate_predictions
import warnings
import joblib
warnings.filterwarnings("ignore")


def create_sequences(X, y, window):
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i-window:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


st.set_page_config(page_title="🧠 NeuroTrade Evolution", layout="wide")
st.title("📈 NeuroTrade Analytics")

tab_train, tab_infer = st.tabs(["🧬 Обучение и эволюция", "📦 Загрузка модели"])

# TAB 1: Обучение и эволюция
with tab_train:
    with st.expander("ℹ️ Описание параметров"):
        st.markdown("""
        **📌 Размер окна (`window`)**  
        Определяет, сколько предыдущих временных шагов учитывается при прогнозе следующего значения.

        **🧬 Количество поколений (`generations`)**  
        Сколько раундов будет проведено для эволюции архитектур.

        **👥 Размер популяции (`population`)**  
        Сколько архитектур будет эволюционировать одновременно.

        **📚 Эпохи (`epochs`)**  
        Сколько раз финальная модель обучается на всех данных.
        """)

    tickers = get_sp500_tickers()
    ticker = st.selectbox("Выберите тикер из S&P 500", ["(случайный)"] + tickers)
    window = st.slider("Размер окна", 10, 100, 60, step=5)
    generations = st.slider("Количество поколений", 1, 10, 2)
    population = st.slider("Размер популяции", 2, 10, 4)
    epochs = st.slider("Эпохи обучения финальной модели", 1, 50, 15)

    save_model = st.checkbox("💾 Сохранять модель и параметры", value=True)
    start_button = st.button("🚀 Запустить обучение")

    if start_button:
        with st.spinner("Загружаем данные..."):
            selected_ticker = choose_random_ticker(tickers) if ticker == "(случайный)" else ticker
            df = download_stock_data(selected_ticker, start='2018-01-01')
            df = DataPreprocessor.clean_dataset(df)
            df = DataPreprocessor.add_all_indicators(df, windows=[5, 10], indicators=['Close'])

            features = df.drop(columns=['Close'])
            target = df[['Close']]
            split_index = int(len(df) * 0.8)
            features_train, features_val = features[:split_index], features[split_index:]
            target_train, target_val = target[:split_index], target[split_index:]

            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_train_scaled = scaler_x.fit_transform(features_train)
            X_val_scaled = scaler_x.transform(features_val)
            y_train_scaled = scaler_y.fit_transform(target_train)
            y_val_scaled = scaler_y.transform(target_val)

            X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, window)
            X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, window)
            input_shape = X_train.shape[1:]

        with st.spinner("🧬 Запускаем нейроэволюцию..."):
            blocks = Blocks()
            best_bot_pop, best_bot, best_setblockov = evolve_population(
                X_train, y_train, X_val, y_val, scaler_y,
                population_size=population,
                generations=generations,
                input_shape=input_shape,
                verbose=True
            )

        with st.spinner("🧠 Финальное обучение модели..."):
            builder = WildregressModel(input_shape=input_shape)
            model = builder(best_bot_pop, best_bot, best_setblockov, blocks)
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train,
                      validation_data=(X_val, y_val),
                      epochs=epochs,
                      batch_size=32,
                      verbose=0)

        with st.spinner("📊 Генерируем прогноз..."):
            y_pred = model.predict(X_val).reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_pred_rescaled = scaler_y.inverse_transform(y_pred)
            y_val_rescaled = scaler_y.inverse_transform(y_val)

            st.subheader(f"📉 Истина vs Прогноз для {selected_ticker}")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(y_val_rescaled, label='Истинные значения', linewidth=2)
            ax.plot(y_pred_rescaled, label='Прогноз модели', linestyle='--')
            ax.set_xlabel("Временной шаг")
            ax.set_ylabel("Цена")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        if save_model:
            st.subheader("💾 Сохраняем модель и параметры")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{selected_ticker}_{timestamp}"
            save_path = f"models/{model_name}"
            os.makedirs(save_path, exist_ok=True)

            model.save(f"{save_path}/model.h5")
            joblib.dump(scaler_y, f"{save_path}/scaler_y.pkl")
            metadata = {
                "ticker": selected_ticker,
                "window": window,
                "generations": generations,
                "population": population,
                "epochs": epochs,
                "bot_pop": best_bot_pop,
                "bot": best_bot,
                "setblockov": best_setblockov
            }
            with open(f"{save_path}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            st.success(f"✅ Модель сохранена в {save_path}/")
            st.json(metadata)

# TAB 2: Загрузка модели
with tab_infer:
    st.subheader("📦 Загрузка ранее обученной модели")

    model_dirs = [f.path for f in os.scandir("models") if f.is_dir()]
    if not model_dirs:
        st.warning("❗ В папке models нет сохранённых моделей")
    else:
        selected_model_dir = st.selectbox("Выберите модель для загрузки", model_dirs)
        load_button = st.button("📥 Загрузить модель")

        if load_button:
            with st.spinner("Загружаем модель..."):
                model, metadata, scaler_y = load_saved_model(selected_model_dir)
                st.json(metadata)

                df = download_stock_data(metadata["ticker"], start='2018-01-01')
                df = DataPreprocessor.clean_dataset(df)
                df = DataPreprocessor.add_all_indicators(df, windows=[5, 10], indicators=['Close'])

                features = df.drop(columns=['Close'])
                target = df[['Close']]
                split_index = int(len(df) * 0.8)
                features_train, features_val = features[:split_index], features[split_index:]
                target_train, target_val = target[:split_index], target[split_index:]

                scaler_x = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_val_scaled = scaler_x.fit_transform(features_val)
                y_val_scaled = scaler_y.fit_transform(target_val)

                window = metadata["window"]
                X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, window)


                y_pred_rescaled = predict_with_model(model, X_val, scaler_y)
                y_val_rescaled = scaler_y.inverse_transform(y_val)

                st.subheader("📈 Прогноз на валидационных данных")
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.plot(y_val_rescaled, label='Истинные значения', linewidth=2)
                ax2.plot(y_pred_rescaled, label='Прогноз', linestyle='--')
                ax2.set_xlabel("Временной шаг")
                ax2.set_ylabel("Цена")
                ax2.legend()
                ax2.grid(True)
                st.pyplot(fig2)

                st.subheader("📊 Метрики качества")
                metrics = evaluate_predictions(y_val_rescaled, y_pred_rescaled)
                st.json(metrics)
