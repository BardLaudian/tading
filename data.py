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

# Obtener datos históricos (puedes ajustar las fechas según tus necesidades)
symbol = 'ETHUSDT'
interval = '1m'  # O '15m' para velas de 15 minutos
start_str = '1 days ago UTC'  # Ajusta según la cantidad de datos históricos que necesites

df = get_historical_data(symbol, interval, start_str)

# Calcular indicadores técnicos usando la biblioteca ta
df['RSI'] = ta.rsi(df['close'], length=10)
macd = ta.macd(df['close'], fast=5, slow=35, signal=5)
# Añadir los resultados al DataFrame original
df = pd.concat([df, macd], axis=1)

# Eliminar filas con valores NaN en 'Adj Close'
df.dropna(subset=['MACDh_5_35_5'], inplace=True)

#buy_threshold_rsi, sell_threshold_rsi, macd_diff = [68.41852, 25.855667, -4.553406]
#buy_threshold_rsi, sell_threshold_rsi, macd_diff = [63.31822087984028, 57.383478518064685, 1.7732106095533613]
buy_threshold_rsi, sell_threshold_rsi, macd_diff = [40.96290750887992, 35.914535267567864, -13.96354288595775]
# Crear la columna de etiquetas basada en el precio de cierre y los indicadores técnicos
def crear_etiqueta(row):
    # Comprobar condiciones para "Comprar"
    if row['RSI'] < buy_threshold_rsi and (row['MACD_5_35_5'] - row['MACDs_5_35_5']) > macd_diff:
        return 1
    # Comprobar condiciones para "Vender"
    elif row['RSI'] > sell_threshold_rsi and (row['MACD_5_35_5'] - row['MACDs_5_35_5']) < -macd_diff:
        return 2
    # Si no se cumplen las condiciones anteriores, "Mantener"
    else:
        return 0

# Aplicar la función para crear la columna de etiquetas
df['Etiqueta'] = df.apply(crear_etiqueta, axis=1)

# Eliminar columnas innecesarias
df.dropna(inplace=True)
df.reset_index(inplace = True)
df.drop(['timestamp', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1, inplace=True)

pd.set_option('display.max_columns', None)

df.to_csv('test_complete_dataset.csv')

print(df.head(20))
# Contar el número de filas para cada etiqueta
conteo_etiquetas = df['Etiqueta'].value_counts()

# Imprimir el conteo de etiquetas
print("Conteo de filas por etiqueta:")
print(conteo_etiquetas)