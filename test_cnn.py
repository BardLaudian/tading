from binance import Client
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import config
import pandas_ta as ta
import time

client = Client(config.API_KEY, config.API_SECRET)

class Portfolio:
    def __init__(self, usdt_balance, eth_balance=0):
        self.usdt_balance = usdt_balance
        self.eth_balance = eth_balance
        self.initial_usdt = usdt_balance  # Guardar el saldo inicial para referencia

    def buy_eth(self, usdt_amount, eth_price):
        if usdt_amount <= self.usdt_balance:
            eth_bought = usdt_amount / eth_price
            self.eth_balance += eth_bought
            self.usdt_balance -= usdt_amount
            print(f"Comprado {eth_bought} ETH por {usdt_amount} USDT a un precio de {eth_price}")
        else:
            print("Saldo USDT insuficiente para comprar.")

    def sell_eth(self, eth_amount, eth_price):
        if eth_amount <= self.eth_balance:
            usdt_earned = eth_amount * eth_price
            self.usdt_balance += usdt_earned
            self.eth_balance -= eth_amount
            print(f"Vendido {eth_amount} ETH por {usdt_earned} USDT a un precio de {eth_price}")
        else:
            print("Saldo ETH insuficiente para vender.")

def get_recent_data(symbol, interval, limit=30):
    recent_bars = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(recent_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    timestamp = df['timestamp']
    df.set_index('timestamp', inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(float)
    price = df['close']

    # Calcular indicadores técnicos usando la biblioteca ta
    df['RSI']=ta.rsi(df['close'], length=15)
    df['EMAF']=ta.ema(df['close'], length=20)
    df['EMAM']=ta.ema(df['close'], length=100)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)

    # Añadir los resultados al DataFrame original
    df = pd.concat([df, macd], axis=1)

    stoch = ta.stoch(df['high'], df['low'], df['close'], k=5, d=3, smooth_k=3)
    # Añadir los resultados al DataFrame original
    df = pd.concat([df, stoch], axis=1)

    # Eliminar filas con valores NaN en 'Adj Close'
    df.dropna(subset=['EMAM'], inplace=True)

    """df['Adj Close'] = (df['close'] + df['EMAF'] + df['EMAM']) / 3

    # Calcular el valor de Target
    df['Target'] = df['Adj Close'] - df['open']
    df['Target'] = df['Target'].shift(-1)

    # Calcular la columna 'TargetClass'
    df['TargetClass'] = [1 if val > 0 else 0 for val in df['Target']]

    # También puedes calcular 'TargetNextClose' después de eliminar filas nulas en 'Target'
    df['TargetNextClose'] = df['Adj Close'].shift(-1)"""
    df['NextClose'] = df['close'].shift(-1)

    # Eliminar columnas innecesarias
    df.dropna(inplace=True)
    df.reset_index(inplace = True)
    df.drop(['timestamp', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1, inplace=True)

    pd.set_option('display.max_columns', None)

    return df, price, timestamp

def prepare_data_for_prediction(df, backcandles, scaler):
    x = []
    # Asegúrate de incluir todas las columnas que usaste durante el entrenamiento, excepto la última
    data_normalized = scaler.transform(df)
    for i in range(backcandles, len(data_normalized)):
        x.append(data_normalized[i-backcandles:i, :-1])  # Todas las columnas excepto la última
    return np.array(x)

def trading_strategy(predictions, current_price, threshold=0.005):
    actions = []
    for predicted_price in predictions:
        price_difference = predicted_price - current_price
        if price_difference > threshold * current_price:
            actions.append('buy')
        elif price_difference < -threshold * current_price:
            actions.append('sell')
        else:
            actions.append('hold')
    return actions

# Cargar el modelo LSTM entrenado
model = load_model('mi_modelo_cnn.h5')

# Cargar el scaler utilizado durante el entrenamiento
scaler = joblib.load('scaler.save')
# Asegúrate de ajustar el scaler con los mismos datos que usaste para el entrenamiento
# Esto puede requerir cargar los datos de entrenamiento y ajustar el scaler nuevamente

# Obtener datos recientes
symbol = 'ETHUSDT'
interval = '15m'
limit = 500  # Debe coincidir con backcandles
backcandles = 30

# Inicializar la cartera con 200 USDT y 0 ETH
portfolio = Portfolio(usdt_balance=200)

# Obtener el precio actual de ETH para realizar la primera compra
df, price, timestamp = get_recent_data(symbol, interval, limit)
current_eth_price = price.iloc[-1]
portfolio.buy_eth(usdt_amount=50, eth_price=current_eth_price)  # Compra inicial de 50 USDT en ETH

result = pd.DataFrame(columns=['Precio actual', 'Precio proximo', 'Decision', 'Saldo USDT', 'Saldo ADA'])

while True:
    try:
        df, price, timestamp = get_recent_data(symbol, interval, limit)
        X_recent = prepare_data_for_prediction(df, backcandles, scaler)
        predictions = model.predict(X_recent)
        # Transformar las predicciones al espacio original
        temp_array = np.zeros((predictions.shape[0], 13))
        temp_array[:, -1] = predictions.ravel()
        y_pred_original_full = scaler.inverse_transform(temp_array)
        y_pred_original = y_pred_original_full[:, -1]

        current_eth_price = price.iloc[-1]
        decision = trading_strategy(y_pred_original, current_eth_price, threshold=0.01)[-1]

        if decision == 'buy' and portfolio.usdt_balance >= 50:
            portfolio.buy_eth(usdt_amount=50, eth_price=current_eth_price)
        elif decision == 'sell' and portfolio.eth_balance * current_eth_price >= 50:
            eth_amount_to_sell = 50 / current_eth_price
            portfolio.sell_eth(eth_amount=eth_amount_to_sell, eth_price=current_eth_price)

        # Guardar el DataFrame actualizado en un archivo CSV
        result.loc[len(result)] = [current_eth_price, y_pred_original[-1], decision, portfolio.usdt_balance, portfolio.eth_balance]
        result.to_csv('resultados_cnn.csv', index=False)

        time.sleep(900)
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        time.sleep(900)