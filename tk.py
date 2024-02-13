from binance import Client
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import config
import pandas_ta as ta
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

client = Client(config.API_KEY, config.API_SECRET)

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

def trading_strategy(predictions, current_price, threshold=0.01):
    """
    Toma decisiones de trading (comprar, vender, mantener) basadas en las predicciones del modelo.

    :param predictions: Lista de precios futuros predichos por el modelo.
    :param current_price: Precio actual del activo.
    :param threshold: Umbral para decidir si la diferencia entre el precio predicho y el actual justifica una acción.
    :return: Lista de acciones recomendadas basadas en las predicciones ('buy', 'sell', 'hold').
    """

    actions = []
    for predicted_price in predictions:
        price_difference = predicted_price - current_price

        # Si el precio predicho es significativamente mayor que el precio actual, considerar comprar
        if price_difference > threshold * current_price:
            actions.append('buy')
        # Si el precio predicho es significativamente menor que el precio actual, considerar vender
        elif price_difference < -threshold * current_price:
            actions.append('sell')
        # Si no hay una diferencia significativa, mantener
        else:
            actions.append('hold')
    
    return actions

# Cargar el modelo LSTM entrenado
model = load_model('mi_modelo.h5')

# Cargar el scaler utilizado durante el entrenamiento
scaler = joblib.load('scaler.save')
# Asegúrate de ajustar el scaler con los mismos datos que usaste para el entrenamiento
# Esto puede requerir cargar los datos de entrenamiento y ajustar el scaler nuevamente

# Obtener datos recientes
symbol = 'ETHUSDT'
interval = '1m'
limit = 500  # Debe coincidir con backcandles
backcandles = 30

# Inicializar listas para almacenar datos
real_prices = []
predicted_prices = []

# Clase de la aplicación
class TradingApp:
    def __init__(self, root):
        self.root = root
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.update_button = tk.Button(self.root, text="Iniciar", command=self.start_loop)
        self.update_button.pack(side=tk.BOTTOM)

        self.running = False

    def start_loop(self):
        self.running = True
        self.update_button.config(text="Detener", command=self.stop_loop)
        self.update_graph()

    def stop_loop(self):
        self.running = False
        self.update_button.config(text="Iniciar", command=self.start_loop)

    def update_graph(self):
        if self.running:
            try:
                df, price, timestamp = get_recent_data(symbol, interval, limit)
                X_recent = prepare_data_for_prediction(df, backcandles, scaler)
                predictions = model.predict(X_recent)
                temp_array = np.zeros((predictions.shape[0], 13))
                temp_array[:, -1] = predictions.ravel()
                y_pred_original_full = scaler.inverse_transform(temp_array)
                y_pred_original = y_pred_original_full[:, -1]

                real_prices.append(price.iloc[-1])
                predicted_prices.append(y_pred_original[-1])

                self.ax.clear()
                print(price.iloc[-1])
                print(y_pred_original[-1])
                self.ax.plot(real_prices, label='Precio Real', color='blue', marker='o')
                self.ax.plot(predicted_prices, label='Predicciones', color='red', linestyle='--', marker='x')
                self.ax.legend()
                self.canvas.draw()

                self.root.after(60000, self.update_graph)  # Actualizar cada 2 segundos
            except Exception as e:
                print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()