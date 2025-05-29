import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time

st.set_page_config(page_title="NeuroTrade Analytics", layout="wide")

ALPHA_VANTAGE_API = "HEFEC3RJITLG276D"
FINNHUB_API = "cni9t2pr01qjk13q9i10cni9t2pr01qjk13q9i1g"

def get_finnhub_prediction(ticker):
    try:
        url = f"https://finnhub.io/api/v1/stock/price-target?symbol={ticker}&token={FINNHUB_API}"
        response = requests.get(url)
        data = response.json()
        
        if 'targetHigh' in data:
            return {
                'targetHigh': data['targetHigh'],
                'targetLow': data['targetLow'],
                'targetMean': data['targetMean'],
                'targetMedian': data['targetMedian']
            }
        return None
    except Exception as e:
        st.error(f"Finnhub API error: {str(e)}")
        return None

def get_alpha_vantage_prediction(ticker):
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API}"
        response = requests.get(url)
        data = response.json().get('Global Quote', {})
        
        if data:
            return {
                'price': float(data.get('05. price', 0)),
                'change': data.get('10. change percent', '0%')
            }
        return None
    except Exception as e:
        st.error(f"Alpha Vantage API error: {str(e)}")
        return None

def get_mock_prediction(data, days):
    last_price = data['Close'].iloc[-1]
    ma = data['Close'].rolling(window=30).mean().iloc[-1]
    trend = 1 if last_price > ma else -1
    
    pred_dates = pd.date_range(
        start=data.index[-1] + timedelta(days=1),
        periods=days,
        freq='D'
    )
    
    base = last_price
    preds = []
    for i in range(1, days+1):
        change = np.random.normal(0.5 * trend, 0.8)
        preds.append(base * (1 + change/100))
        base = preds[-1]
    
    return pred_dates, np.array(preds)

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date, end_date, retry=3):
    for attempt in range(retry):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=10
            )
            
            if not data.empty:
                return data
                
            variations = [
                ticker,
                f"{ticker}.NS",  # Для индийских акций
                f"{ticker}.AX",  # Для австралийских
                f"{ticker}.L",   # Для лондонских
                f"{ticker}.TO"   # Для канадских
            ]
            
            for variation in variations:
                if variation == ticker:
                    continue
                    
                st.warning(f"Trying {variation}...")
                data = yf.download(
                    variation,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    timeout=10
                )
                
                if not data.empty:
                    return data
                
            st.warning("Falling back to Alpha Vantage...")
            return get_alpha_vantage_data(ticker)
            
        except Exception as e:
            if attempt == retry - 1:
                st.error(f"Final attempt failed: {str(e)}")
                return None
            time.sleep(2)  

    return None

def get_demo_data():
    date_rng = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
    return pd.DataFrame({
        'Open': np.cumsum(np.random.randn(len(date_rng))) + 100,
        'High': np.cumsum(np.random.randn(len(date_rng))) + 105,
        'Low': np.cumsum(np.random.randn(len(date_rng))) + 95,
        'Close': np.cumsum(np.random.randn(len(date_rng))) + 100,
        'Volume': np.random.poisson(1000000, size=len(date_rng))
    }, index=date_rng)


def get_alpha_vantage_data(ticker):
    API_KEY = "HEFEC3RJITLG276D"  
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={API_KEY}&outputsize=full"
    
    try:
        response = requests.get(url)
        data = response.json()
        if "Time Series (Daily)" not in data:
            st.error(f"Alpha Vantage error: {data.get('Note', 'Unknown error')}")
            return None
            
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        st.error(f"Alpha Vantage API error: {str(e)}")
        return None
    
def main():
    st.title('📊 Аналитика прогнозирования NeuroTrade Analytics')
    
    with st.sidebar:
        st.header('Настройки')
        ticker = st.text_input('Биржевой тикер', 'AAPL').strip().upper()
        
        if not ticker.isalnum():
            st.error("Тикер должен содержать только буквы и цифры")
            st.stop()
            
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Начальная Дата', datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input('Конечная Дата', datetime.now())
            
        if start_date >= end_date:
            st.error("Начальная дата должна быть раньше конечной даты")
            st.stop()
            
        use_demo = st.checkbox("Использовать демо-данные", True)

    with st.spinner('Загрузка данных...'):
        data = get_stock_data(ticker, start_date, end_date)
        
        if data is None and use_demo:
            st.warning("Используется демо-данные")
            data = get_demo_data()
        elif data is None:
            st.error("Не получилось загрузить данные. Проверьте тикер и даты.")
            st.stop()
    

    st.subheader('🔮 Предсказание цены')
    days_to_predict = st.slider('Кол-во дней для предсказания', 7, 90, 30)
    
    if st.button('Получить предсказание', type="primary"):
        with st.spinner('Формируем предсказание...'):
            finnhub_pred = get_finnhub_prediction(ticker)
            if finnhub_pred:
                st.subheader("NeuroTrade Analyst Predictions")
                cols = st.columns(4)
                cols[0].metric("High Target", f"${finnhub_pred['targetHigh']:.2f}")
                cols[1].metric("Low Target", f"${finnhub_pred['targetLow']:.2f}")
                cols[2].metric("Mean Target", f"${finnhub_pred['targetMean']:.2f}")
                cols[3].metric("Median Target", f"${finnhub_pred['targetMedian']:.2f}")
            
            av_pred = get_alpha_vantage_prediction(ticker)
            if av_pred:
                st.subheader("Текущие данные Alpha Vantage")
                st.metric("Текущая цена", f"${av_pred['price']:.2f}", av_pred['change'])
            
            st.subheader("Предсказание NeuroTrade Analytics")
            pred_dates, pred_prices = get_mock_prediction(data, days_to_predict)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index[-60:], data['Close'].values[-60:], label='Исторические данные', color='blue')
            ax.plot(pred_dates, pred_prices, label='Прогноз', color='red', linestyle='--')
            ax.set_title(f'{ticker} Прогноз цены ({days_to_predict} days)')
            ax.legend()
            st.pyplot(fig)
            
            forecast_df = pd.DataFrame({
                'Date': pred_dates,
                'Predicted Price': pred_prices,
                'Change %': (pred_prices/pred_prices[0]-1)*100
            }).set_index('Date')
            
            st.dataframe(
                forecast_df.style.format({
                    'Predicted Price': '${:.2f}',
                    'Change %': '{:.2f}%'
                })
            )

if __name__ == '__main__':
    main()