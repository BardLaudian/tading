from binance import Client
import pandas as pd
import pandas_ta as ta
import config

client = Client(config.API_KEY, config.API_SECRET)

# Función para obtener datos históricos y convertirlos en DataFrame
def get_historical_data(symbol, interval, start_str, end_str=None):
    all_bars = []
    limit = 1000  # Límite máximo de datos por solicitud
    while True:
        bars = client.get_historical_klines(symbol, interval, start_str, end_str, limit=limit)
        if len(bars) == 0:
            break  # Sale del bucle si no hay más datos
        all_bars += bars
        last_bar = bars[-1]
        start_str = last_bar[0] + 1  # Actualiza start_str para la siguiente solicitud

    # Convertir los datos en DataFrame
    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(float)
    return df

# Obtener datos históricos
symbol = 'ETHUSDT'
interval = '1m'  # Velas de 1 minuto
start_str = '12 hour ago UTC'  # 5 años de datos históricos

df = get_historical_data(symbol, interval, start_str)

# Calcular indicadores técnicos usando la biblioteca ta
df['RSI'] = ta.rsi(df['close'])
df['SMA_50'] = ta.sma(df['close'], length=50)
df['SMA_200'] = ta.sma(df['close'], length=200)
df['EMA_50'] = ta.ema(df['close'], length=50)
df['EMA_200'] = ta.ema(df['close'], length=200)
macd = ta.macd(df['close'])
df = pd.concat([df, macd], axis=1)
# Calcular Bandas de Bollinger
bollinger = ta.bbands(df['close'])

# Asegúrate de que las columnas de Bandas de Bollinger tienen los nombres esperados
# Los nombres predeterminados pueden variar, así que ajusta los nombres de las columnas según sea necesario
expected_bollinger_columns = ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']
actual_bollinger_columns = bollinger.columns.tolist()

# Cambiar los nombres de las columnas de las Bandas de Bollinger si no coinciden con los esperados
if set(expected_bollinger_columns) != set(actual_bollinger_columns):
    rename_dict = {actual: expected for actual, expected in zip(actual_bollinger_columns, expected_bollinger_columns)}
    bollinger.rename(columns=rename_dict, inplace=True)

# Añadir las Bandas de Bollinger al DataFrame principal
df = pd.concat([df, bollinger], axis=1)

# Preparar el DataFrame para el modelo
df['next_close'] = df['close'].shift(-1)  # Precio de cierre del próximo minuto
df.dropna(inplace=True)  # Eliminar filas con valores NaN

# Seleccionar características relevantes
features = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'next_close']
df = df[features]

# Guardar el conjunto de datos procesado
df.to_csv('processed_dataset.csv')
