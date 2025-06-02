import pandas as pd
import yfinance as yf
import random

def get_sp500_tickers():
    """ Загружает список тикеров S&P 500 с Википедии """
    url = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    return df['Symbol'].tolist()

def choose_random_ticker(tickers):
    """ Возвращает случайный тикер из списка"""
    return random.choice(tickers)

def download_stock_data(ticker, start='2010-01-01', end=None, interval='1d'):
    """ Загружает исторические данные по тикеру с Yahoo Finance"""
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    df.dropna(inplace=True)
    return df
