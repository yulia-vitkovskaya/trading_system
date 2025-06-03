import os
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def load_saved_model(model_dir):
    model_path = os.path.join(model_dir, "model.h5")
    metadata_path = os.path.join(model_dir, "metadata.json")
    scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path) or not os.path.exists(scaler_y_path):
        raise FileNotFoundError("Ожидаемые файлы model.h5, metadata.json или scaler_y.pkl не найдены.")

    model = load_model(model_path, compile=False)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    scaler_y = joblib.load(scaler_y_path)

    return model, metadata, scaler_y


def predict_with_model(model, X, scaler_y):
    """
    Делает прогноз и обратную трансформацию
    """
    y_pred = model.predict(X).reshape(-1, 1)
    return scaler_y.inverse_transform(y_pred)


def evaluate_predictions(y_true, y_pred):
    """
    Возвращает основные метрики качества
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4)
    }
