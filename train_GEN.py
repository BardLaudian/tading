import random
from deap import base, creator, tools, algorithms
import pandas as pd
from multiprocessing import Pool
import time
import numpy as np

# Cargar el DataFrame
df = pd.read_csv('complete_dataset.csv', index_col=0, parse_dates=True)
df = df.drop(columns=['Etiqueta'])

evaluation_cache = {}

class Portfolio:
    def __init__(self, usdt_balance, eth_balance=0, purchase_percentage=0.1):
        self.usdt_balance = usdt_balance
        self.eth_balance = eth_balance
        self.buy_price = []
        self.purchase_percentage = purchase_percentage
        self.update_purchase_amount()
        self.commission_rate = 0.0010  # Tasa de comisión de Binance del 0.1000%

    def update_purchase_amount(self):
        self.purchase_amount = self.usdt_balance * self.purchase_percentage  # Recalcula el monto de compra

    def buy_eth(self, eth_price):
        if self.purchase_amount <= self.usdt_balance:
            commission = self.purchase_amount * self.commission_rate
            effective_purchase_amount = self.purchase_amount - commission
            eth_amount = effective_purchase_amount / eth_price
            self.eth_balance += eth_amount
            self.usdt_balance -= self.purchase_amount  # Deduce el monto total incluyendo la comisión
            self.buy_price.append((eth_price, eth_amount))  # Guarda el precio y la cantidad comprada
            return True
        return False

    def sell_eth(self, eth_price):
        if self.eth_balance > 0 and self.buy_price:
            gross_usdt_earned = self.eth_balance * eth_price
            commission = gross_usdt_earned * self.commission_rate
            net_usdt_earned = gross_usdt_earned - commission
            self.usdt_balance += net_usdt_earned
            self.eth_balance = 0  # Asume que vendes todo el ETH disponible
            self.buy_price = []  # Limpia la lista de precios de compra
            self.update_purchase_amount()  # Recalcula el monto de compra después de vender
            return True
        return False

def simulate_trading_strategy(individual, df):
    # Convertir el individuo a una tupla para usarlo como clave de caché
    individual_tuple = tuple(individual)
    
    # Verificar si el individuo ya está en la caché
    if individual_tuple in evaluation_cache:
        return evaluation_cache[individual_tuple]
    
    portfolio = Portfolio(usdt_balance=300)
    buy_threshold_rsi, sell_threshold_rsi, macd_diff = individual

    # Identificar señales de compra y venta
    buy_signals = (df['RSI'] < buy_threshold_rsi) & ((df['MACD_5_35_5'] - df['MACDs_5_35_5']) > macd_diff)
    sell_signals = (df['RSI'] > sell_threshold_rsi) & ((df['MACD_5_35_5'] - df['MACDs_5_35_5']) < -macd_diff)

    # Obtener índices de señales de compra y venta
    buy_indices = np.where(buy_signals)[0]
    sell_indices = np.where(sell_signals)[0]

    # Iterar solo a través de señales de compra y venta
    for i in sorted(np.union1d(buy_indices, sell_indices)):
        current_price = df.iloc[i]['close']
        if i in buy_indices:
            portfolio.buy_eth(current_price)
        elif i in sell_indices:
            portfolio.sell_eth(current_price)

    last_price = df['close'].iloc[-1]
    total_value = portfolio.usdt_balance + (portfolio.eth_balance * last_price)
    
    # Almacenar el resultado en la caché antes de retornarlo
    evaluation_cache[individual_tuple] = (total_value,)
    return total_value,

if __name__ == '__main__':
    start_time = time.time()  # Tiempo de inicio
    # Configuración del algoritmo genético
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Paralelización con multiprocessing
    pool = Pool()
    toolbox.register("map", pool.map)  # Usa el pool de procesos para la evaluación en paralelo

    toolbox.register("attr_float", random.uniform, 20, 80)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", simulate_trading_strategy, df=df)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Inicializar la población
    population = toolbox.population(n=50)

    # Ejecutar el algoritmo genético
    ngen = 100
    result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, verbose=True)

    best_individual = tools.selBest(population, 1)[0]
    print("Mejor estrategia:", best_individual)
    print("Mejor fitness:", best_individual.fitness.values[0])

    pool.close()  # No olvides cerrar el pool al final para liberar recursos

    end_time = time.time()  # Tiempo de finalización
    total_time = end_time - start_time  # Cálculo del tiempo total de ejecución
    print(f"Tiempo total de ejecución: {total_time / 60} minutos")