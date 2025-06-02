import pandas as pd
import numpy as np
from typing import List, Union
import warnings
warnings.filterwarnings("ignore")

class DataPreprocessor:
    """
    Класс для предобработки финансовых данных.
    Включает методы для добавления технических индикаторов и очистки данных.
    
    Пример использования:
    >>> preprocessor = DataPreprocessor()
    >>> df_processed = preprocessor.add_sma(df, windows=[5, 10], indicators=['Close'])
    """
    
    @staticmethod
    def add_sma(df: pd.DataFrame, windows: List[int], indicators: List[str]) -> pd.DataFrame:
        """Добавляет простые скользящие средние (SMA)"""
        df = df.copy()
        for window in windows:
            for col in indicators:
                df[f'{col}_SMA_{window}'] = df[col].rolling(window=window).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, windows: List[int], indicators: List[str]) -> pd.DataFrame:
        """Добавляет экспоненциальные скользящие средние (EMA)."""
        df = df.copy()
        for window in windows:
            for col in indicators:
                df[f'{col}_EMA_{window}'] = df[col].ewm(span=window, adjust=False).mean()
        return df
    
    @staticmethod
    def add_obv(df: pd.DataFrame, indicators: list):
        """Добавит OBV осцилятор"""
        copy = df.copy()
        for col in indicators:
            copy[f'{col}_OBV'] = (np.sign(copy[col].diff()) * copy["Volume"]).fillna(0).cumsum()
        return copy

    @staticmethod
    def add_macd(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Добавляет индикатор MACD."""
        df = df.copy()
        for col in indicators:
            exp1 = df[col].ewm(span=12, adjust=False).mean()
            exp2 = df[col].ewm(span=26, adjust=False).mean()
            df[f'{col}_MACD'] = exp1 - exp2
        return df

    @staticmethod
    def add_vwap(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Добавляет средневзвешенную цену по объему (VWAP)."""
        df = df.copy()
        q = df['Volume'].values
        for col in indicators:
            p = df[col].values
            df[f'{col}_VWAP'] = df.assign(vwap=(p * q).cumsum() / q.cumsum()).vwap
        return df
    
    @staticmethod
    def add_changes(df: pd.DataFrame, depth: int, indicators: list,
                only_indicators = False, type_change = 'diff'):
        """Добавляет изменения в индикаторах."""
        methods = ('diff','pct_change')
        assert type_change in methods, f'В type_change доступо {methods}'
        copy = df.copy() if not only_indicators else df[indicators].copy()
        for i in range(1, depth + 1):
            indicators_changes = [f'{ind}_diff_{i}' for ind in indicators]
            for indicator_change, indicator in zip(indicators_changes, indicators):
                if type_change == 'diff':
                    copy[indicator_change] = copy[indicator].diff(periods=i)
                elif type_change == 'pct_change':
                    copy[indicator_change] = copy[indicator].pct_change(periods=i)
        return copy

    @staticmethod
    def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        Очищает DataFrame от NaN и бесконечных значений.
        
        Возвращает:
        ----------
        pd.DataFrame
            Очищенный DataFrame
        """
        assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
        return df[indices_to_keep].astype(np.float64)

    @staticmethod
    def add_all_indicators(df: pd.DataFrame, 
                         windows: List[int], 
                         indicators: List[str],
                         depth: int = 25) -> pd.DataFrame:
        """
        Комплексный метод добавления всех индикаторов.
        
        Параметры:
        ----------
        df : pd.DataFrame
            Исходные данные
        windows : List[int]
            Окна для SMA/EMA
        indicators : List[str]
            Колонки для расчета
        depth : int, optional
            Глубина для дифференцирования (по умолчанию 25)
            
        Возвращает:
        ----------
        pd.DataFrame
            DataFrame со всеми индикаторами
        """
        df = DataPreprocessor.add_vwap(df, indicators)
        df = DataPreprocessor.add_obv(df, indicators)
        df = DataPreprocessor.add_changes(df, depth, indicators, type_change='pct_change')
        df = DataPreprocessor.add_sma(df, windows, indicators)
        df = DataPreprocessor.add_ema(df, windows, indicators)
        df = DataPreprocessor.add_macd(df, indicators)
        return df
    

# Пример использования
# if __name__ == "__main__":
#     # Тестовый пример
#     data = {'Close': [1, 2, 3, 4, 5, 4, 3, 2, 1], 
#             'Volume': [100, 200, 300, 400, 500, 400, 300, 200, 100]}
#     df = pd.DataFrame(data)
    
#     processor = DataPreprocessor()
#     processed_df = processor.add_sma(df, windows=[2, 3], indicators=['Close'])
#     print(processed_df.head())