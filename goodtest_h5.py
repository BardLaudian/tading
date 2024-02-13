import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm

# Definir la clase Portfolio
class Portfolio:
    def __init__(self, usdt_balance, eth_balance=0):
        self.usdt_balance = usdt_balance
        self.eth_balance = eth_balance

    def buy_eth(self, eth_amount, eth_price):
        usdt_required = eth_amount * eth_price
        if usdt_required <= self.usdt_balance:
            self.eth_balance += eth_amount
            self.usdt_balance -= usdt_required
            return eth_amount
        else:
            return 0

    def sell_eth(self, eth_amount, eth_price):
        if eth_amount <= self.eth_balance:
            usdt_earned = eth_amount * eth_price
            self.usdt_balance += usdt_earned
            self.eth_balance -= eth_amount
            return usdt_earned
        else:
            return 0

# Cargar el conjunto de datos
df = pd.read_csv('complete_dataset.csv', parse_dates=True, index_col=0)

# Cargar el modelo y el scaler
model = load_model('mi_modelo_pca.h5')
scaler = joblib.load('scaler.save')

# Preparar los datos para la predicción
def prepare_data_for_prediction(df, backcandles, scaler):
    data_normalized = scaler.transform(df)
    X = np.array([data_normalized[-backcandles:, :-1]])  # Excluye las dos últimas columnas
    return X

# Estrategia de trading actualizada
def trading_strategy(prediction):
    if prediction == 1:
        return 'buy'
    elif prediction == 2:
        return 'sell'
    else:
        return 'hold'

# Inicializar la cartera
portfolio = Portfolio(usdt_balance=300)  # Comienza con 1000 USDT
backcandles = 60
transactions = []

# Iterar a través del conjunto de datos
for i in tqdm(range(backcandles, len(df) - 1), desc='Procesando'):
    current_df = df.iloc[(i - backcandles):i]
    X_recent = prepare_data_for_prediction(current_df, backcandles, scaler)
    prediction = model.predict(X_recent, verbose=0)[0]
    decision = trading_strategy(np.argmax(prediction))  # Use np.argmax for categorical predictions

    current_price = df.iloc[i]['close']
    eth_amount = 0.1

    if decision == 'buy' and portfolio.usdt_balance >= current_price * eth_amount:
        portfolio.buy_eth(eth_amount, current_price)
        transactions.append(('buy', df.index[i], current_price))
    elif decision == 'sell' and portfolio.eth_balance >= eth_amount:
        portfolio.sell_eth(eth_amount, current_price)
        transactions.append(('sell', df.index[i], current_price))

last_price = df['close'].iloc[-1]
total_value = portfolio.usdt_balance + (portfolio.eth_balance * last_price)

print("Extracto final del Portfolio:")
print(f"Balance en USDT: {portfolio.usdt_balance:.2f} USDT")
print(f"Balance en ETH: {portfolio.eth_balance:.4f} ETH")
print(f"Valor total de la cartera: {total_value:.2f} USDT")

print("\nTransacciones realizadas:")
for transaction in transactions:
    action, date, price = transaction
    print(f"{date}: {action.capitalize()} a {price:.2f} USDT")

prices = df['close'][backcandles:len(df) - 1]
plt.figure(figsize=(14, 7))
plt.plot(prices, label='Precio de ETH', color='blue')
for transaction in transactions:
    if transaction[0] == 'buy':
        plt.scatter(transaction[1], transaction[2], color='green', marker='^', label='Compra')
    elif transaction[0] == 'sell':
        plt.scatter(transaction[1], transaction[2], color='red', marker='v', label='Venta')
plt.title('Precio de ETH y transacciones')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.show()