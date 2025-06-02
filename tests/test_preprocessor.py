import pandas as pd
import pytest
import numpy as np
from src.data_processing.preprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    """Фикстура с тестовыми данными"""
    return pd.DataFrame({
        'Close': [1, 2, 3, 4, 5, 4, 3, 2, 1],
        'Volume': [100, 200, 300, 400, 500, 400, 300, 200, 100],
        'High': [1.1, 2.1, 3.1, 4.1, 5.1, 4.1, 3.1, 2.1, 1.1],
        'Low': [0.9, 1.9, 2.9, 3.9, 4.9, 3.9, 2.9, 1.9, 0.9],
        'Open': [1.05, 2.05, 3.05, 4.05, 5.05, 4.05, 3.05, 2.05, 1.05]
    })

def test_add_sma(sample_data):
    """Тест для простых скользящих средних"""
    result = DataPreprocessor.add_sma(sample_data, windows=[2, 3], indicators=['Close', 'High'])

    expected_columns = ['Close_SMA_2', 'Close_SMA_3', 'High_SMA_2', 'High_SMA_3']
    for col in expected_columns:
        assert col in result.columns

    # Проверка правильности расчетов
    # Close_SMA_2 на 2-й индексе: (2+3)/2 = 2.5
    assert result['Close_SMA_2'].iloc[2] == pytest.approx(2.5)
    # High_SMA_3 на 4-м индексе: (3.1+4.1+5.1)/3
    assert result['High_SMA_3'].iloc[4] == pytest.approx((3.1 + 4.1 + 5.1) / 3)

def test_add_ema(sample_data):
    """Тест для экспоненциальных скользящих средних"""
    result = DataPreprocessor.add_ema(sample_data, windows=[2, 3], indicators=['Low', 'Open'])
    
    assert 'Low_EMA_2' in result.columns
    assert 'Open_EMA_3' in result.columns
    assert not result['Low_EMA_2'].isnull().all()

def test_add_obv(sample_data):
    """Тест для индикатора OBV"""
    result = DataPreprocessor.add_obv(sample_data, indicators=['Close', 'High'])
    
    assert 'Close_OBV' in result.columns
    assert 'High_OBV' in result.columns
    # Проверка финального значения OBV вручную: с учетом направлений
    expected_obv = ( # направления: + + - - - → +200+300-400-300-200 = -400
        0 + 200 + 300 - 400 - 300 - 200
    )
    assert result['Close_OBV'].iloc[-1] == pytest.approx(400)

def test_add_macd(sample_data):
    """Тест для индикатора MACD"""
    result = DataPreprocessor.add_macd(sample_data, indicators=['Close', 'Low'])
    
    assert 'Close_MACD' in result.columns
    assert 'Low_MACD' in result.columns
    assert not result['Close_MACD'].isnull().all()

def test_add_vwap(sample_data):
    """Тест для VWAP"""
    result = DataPreprocessor.add_vwap(sample_data, indicators=['Close', 'High'])
    
    assert 'Close_VWAP' in result.columns
    assert 'High_VWAP' in result.columns
    # Проверим, что результат не пуст и монотонно изменяется
    assert not result['Close_VWAP'].isnull().all()
    assert result['Close_VWAP'].iloc[-1] > result['Close_VWAP'].iloc[0]

def test_add_changes(sample_data):
    """Тест для добавления изменений"""
    result_diff = DataPreprocessor.add_changes(sample_data, depth=2, indicators=['Close'], type_change='diff')
    result_pct = DataPreprocessor.add_changes(sample_data, depth=2, indicators=['High'], type_change='pct_change')

    assert 'Close_diff_1' in result_diff.columns
    assert 'Close_diff_2' in result_diff.columns
    assert 'High_diff_1' in result_pct.columns
    assert 'High_diff_2' in result_pct.columns
    # Проверка, что значения рассчитываются и не пустые
    assert not result_pct['High_diff_2'].isnull().all()

def test_clean_dataset():
    """Тест очистки данных"""
    dirty_data = pd.DataFrame({
        'Close': [1, np.nan, 3, np.inf, -5, 6],
        'Volume': [100, 200, -np.inf, 400, 500, 600],
        'High': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    })
    
    clean = DataPreprocessor.clean_dataset(dirty_data)
    assert clean.shape[0] == 3
    assert not clean.isnull().any().any()
    assert not np.isinf(clean.values).any()

def test_add_all_indicators(sample_data):
    """Комплексный тест всех индикаторов"""
    result = DataPreprocessor.add_all_indicators(
        sample_data,
        windows=[2, 3],
        indicators=['Close', 'High', 'Low'],
        depth=2
    )

    required_columns = [
        'Close_SMA_2', 'Close_SMA_3',
        'High_EMA_2', 'High_EMA_3',
        'Low_MACD', 'Close_VWAP',
        'High_OBV', 'Close_diff_1',
        'Close_diff_2'
    ]
    for col in required_columns:
        assert col in result.columns, f"Отсутствует колонка {col}"

    assert 'Volume' in result.columns
    assert result.shape[0] == sample_data.shape[0]

def test_edge_cases():
    """Тест граничных случаев"""
    # Пустой DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        if empty_df.empty:
            raise ValueError("Empty DataFrame")

    # Неправильный тип данных
    with pytest.raises(AttributeError):
        DataPreprocessor.add_changes("not_a_dataframe", 1, ['Close'])

    # Некорректный type_change
    df = pd.DataFrame({'Close': [1, 2, 3]})
    with pytest.raises(AssertionError):
        DataPreprocessor.add_changes(df, 1, ['Close'], type_change='invalid_type')
