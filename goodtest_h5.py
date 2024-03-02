import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from tqdm import tqdm

class Portfolio:
    def __init__(self, usdt_balance, eth_balance=0):
        self.usdt_balance = usdt_balance
        self.eth_balance = eth_balance

    def execute_action(self, action, eth_price):
        if action == 1:  # Comprar ETH
            if self.usdt_balance > 0:
                eth_bought = self.usdt_balance / eth_price
                self.eth_balance += eth_bought
                self.usdt_balance = 0
                return eth_bought, 0
        elif action == 2:  # Vender ETH
            if self.eth_balance > 0:
                usdt_earned = self.eth_balance * eth_price
                self.usdt_balance += usdt_earned
                self.eth_balance = 0
                return 0, usdt_earned
        return 0, 0

# Cargar el conjunto de datos
df = pd.read_csv('year_complete_dataset.csv', index_col=0)

# Cargar el modelo DQN
model = load_model('dqn_trading_model_simple.h5')

# Inicializar la cartera
portfolio = Portfolio(usdt_balance=300)
transactions = []

# Iterar a través del conjunto de datos
for i in tqdm(range(len(df)), desc='Procesando transacciones'):
    current_state = df.iloc[i].to_numpy()
    # Asegúrate de que tenga la forma adecuada, rellenando hasta 25 características
    current_state = np.pad(current_state, (0, max(0, 18 - len(current_state))), 'constant')
    current_state = current_state.reshape(1, -1)  # Redimensionar para que sea compatible con el modelo

    try:
        action = np.argmax(model.predict(current_state, verbose=0)[0])  # Decidir la acción
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Error al predecir la acción en el paso {i}: {e}")
        action = 0

    current_price = df.iloc[i]['close']
    eth_bought, usdt_earned = portfolio.execute_action(action, current_price)

    if eth_bought > 0:
        transactions.append(('buy', df.index[i], current_price, eth_bought))
    elif usdt_earned > 0:
        transactions.append(('sell', df.index[i], current_price, usdt_earned))

last_price = df['close'].iloc[-1]
total_value = portfolio.usdt_balance + (portfolio.eth_balance * last_price)

print("Extracto final del Portfolio:")
print(f"Balance en USDT: {portfolio.usdt_balance:.2f} USDT")
print(f"Balance en ETH: {portfolio.eth_balance:.4f} ETH")
print(f"Valor total de la cartera: {total_value:.2f} USDT")

print("\nTransacciones realizadas:")
for transaction in transactions:
    action, date, price, amount = transaction
    if action == 'buy':
        print(f"{date}: Compra de {amount:.4f} ETH a {price:.2f} USDT")
    elif action == 'sell':
        print(f"{date}: Venta, ganancia de {amount:.2f} USDT a {price:.2f} USDT")

prices = df['close']
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
