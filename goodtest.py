import pandas as pd
import joblib
from tqdm import tqdm

class Portfolio:
    def __init__(self, usdt_balance, eth_balance=0):
        self.usdt_balance = usdt_balance
        self.eth_balance = eth_balance
        self.buy_price = []  # Lista para almacenar precios de compra
        self.purchase_percentage = 0.1  # Usar el 10% del saldo de USDT para compras
        self.update_purchase_amount()  # Calcula el monto de compra inicial
        self.commission_rate = 0.0010

    def update_purchase_amount(self):
        self.purchase_amount = self.usdt_balance * self.purchase_percentage  # Actualiza el monto de compra

    def buy_eth(self, eth_price):
        if self.purchase_amount <= self.usdt_balance:
            commission = self.purchase_amount * self.commission_rate
            effective_purchase_amount = self.purchase_amount - commission
            eth_amount = effective_purchase_amount / eth_price
            self.eth_balance += eth_amount
            self.usdt_balance -= self.purchase_amount
            self.buy_price.append((eth_price, eth_amount))  # Guarda el precio y la cantidad comprada
            return eth_amount
        return 0

    def sell_eth(self, eth_price):
        if self.eth_balance > 0 and self.buy_price:
            eth_amount = self.eth_balance
            usdt_earned = eth_amount * eth_price
            commission = usdt_earned * self.commission_rate
            net_usdt_earned = usdt_earned - commission
            self.usdt_balance += net_usdt_earned
            self.eth_balance -= eth_amount
            self.update_purchase_amount()  # Actualiza el monto de compra después de cada venta
            # Calcula el resultado de la operación
            avg_buy_price = sum([price * amount for price, amount in self.buy_price]) / sum([amount for price, amount in self.buy_price])
            trade_outcome = eth_price > avg_buy_price
            self.buy_price = []  # Limpia la lista de precios de compra
            return usdt_earned, trade_outcome
        return 0, None

# Cargar el conjunto de datos
df = pd.read_csv('test_complete_dataset.csv', parse_dates=True, index_col=0)

# Cargar el modelo de Bosque Aleatorio
rf_model = joblib.load('rf_model.joblib')

def trading_strategy(prediction):
    if prediction == 1:
        return 'buy'
    elif prediction == 2:
        return 'sell'
    else:
        return 'hold'

# Inicializar variables para rastrear estadísticas de operaciones
total_trades = 0
winning_trades = 0
losing_trades = 0

portfolio = Portfolio(usdt_balance=300)
monto_compra = portfolio.usdt_balance / 5
transactions = []

for i in tqdm(range(len(df)), desc='Procesando'):
    current_row = df.iloc[i:i+1]
    X_recent = current_row.drop(columns=['Etiqueta'])
    prediction = rf_model.predict(X_recent)[0]
    decision = trading_strategy(prediction)

    current_price = df.iloc[i]['close']

    if decision == 'buy' and portfolio.usdt_balance >= monto_compra:  # Usar monto_compra en la condición
        eth_amount = portfolio.buy_eth(current_price)  # Usar monto_compra en la compra
        if eth_amount > 0:
            transactions.append(('buy', df.index[i], current_price))
    elif decision == 'sell' and portfolio.eth_balance > 0:
        usdt_earned, trade_outcome = portfolio.sell_eth(current_price)  # Vender todo el ETH disponible
        transactions.append(('sell', df.index[i], current_price))
        if trade_outcome is not None:
            total_trades += 1
            if trade_outcome:
                winning_trades += 1
            else:
                losing_trades += 1

# Imprimir estadísticas de las operaciones
print(f"Total de operaciones: {total_trades}")
print(f"Operaciones ganadoras: {winning_trades}")
print(f"Operaciones perdedoras: {losing_trades}")
if total_trades > 0:
    win_rate = (winning_trades / total_trades) * 100
    print(f"Tasa de éxito: {win_rate:.2f}%")
else:
    print("No se realizaron operaciones suficientes para calcular la tasa de éxito.")

last_price = df['close'].iloc[-1]
total_value = portfolio.usdt_balance + (portfolio.eth_balance * last_price)

print("Extracto final del Portfolio:")
print(f"Balance en USDT: {portfolio.usdt_balance:.2f} USDT")
print(f"Balance en ETH: {portfolio.eth_balance:.4f} ETH")
print(f"Valor total de la cartera: {total_value:.2f} USDT")

"""prices = df['close']
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
plt.show()"""