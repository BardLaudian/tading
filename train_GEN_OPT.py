import random
from deap import base, creator, tools, algorithms
import pandas as pd
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool

# Cargar el DataFrame
df = pd.read_csv('complete_dataset.csv', index_col=0, parse_dates=True)
df = df.drop(columns=['Etiqueta'])

class Portfolio:
    def __init__(self, usdt_balance, eth_balance=0):
        self.usdt_balance = usdt_balance
        self.eth_balance = eth_balance
        self.buy_price = []

    def buy_eth(self, usdt_amount, eth_price):
        if usdt_amount <= self.usdt_balance:
            eth_amount = usdt_amount / eth_price
            self.eth_balance += eth_amount
            self.usdt_balance -= usdt_amount
            self.buy_price.append(eth_price)
            return True
        return False

    def sell_eth(self, eth_amount, eth_price):
        if eth_amount <= self.eth_balance and self.buy_price:
            usdt_earned = eth_amount * eth_price
            self.usdt_balance += usdt_earned
            self.eth_balance -= eth_amount
            self.buy_price = []
            return True
        return False

def simulate_trading_strategy(individual, df):
    portfolio = Portfolio(usdt_balance=300)  # Inicializar el portfolio con un saldo de USDT
    buy_threshold_rsi, sell_threshold_rsi, macd_diff = individual

    for index, row in df.iterrows():
        current_price = row['close']
        if row['RSI'] < buy_threshold_rsi and (row['MACD_5_35_5'] - row['MACDs_5_35_5']) > macd_diff:
            portfolio.buy_eth(50, current_price)  # Suponiendo que cada compra usa 50 USDT
        elif row['RSI'] > sell_threshold_rsi and (row['MACD_5_35_5'] - row['MACDs_5_35_5']) < -macd_diff:
            portfolio.sell_eth(portfolio.eth_balance, current_price)  # Vender todo el ETH disponible

    last_price = df['close'].iloc[-1]
    total_value = portfolio.usdt_balance + (portfolio.eth_balance * last_price)
    return (total_value,)

def main():
    # Configuración inicial del algoritmo genético
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 20, 80)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
    toolbox.register("population_n", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", simulate_trading_strategy, df=df)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Define el espacio de búsqueda de los hiperparámetros
    param_grid = {
        'cxpb': [0.5, 0.7, 1.0],  # Probabilidad de cruce
        'mutpb': [0.1, 0.2, 0.5],  # Probabilidad de mutación
        'pop_size': [20, 50, 100]  # Tamaño de la población
    }

    grid = ParameterGrid(param_grid)

    best_score = float('-inf')
    best_params = None
    best_individual = None

    for params in grid:
        population = toolbox.population_n(n=params['pop_size'])
        
        with Pool() as pool:
            toolbox.register("map", pool.map)
            result = algorithms.eaSimple(population, toolbox, cxpb=params['cxpb'], mutpb=params['mutpb'], ngen=40, verbose=False)
        
        top_individual = tools.selBest(population, 1)[0]
        top_score = top_individual.fitness.values[0]

        if top_score > best_score:
            best_score = top_score
            best_params = params
            best_individual = top_individual

    print(f"Mejores parámetros: {best_params}")
    print(f"Mejor rendimiento: {best_score}")
    print(f"Mejor individuo: {best_individual}")

if __name__ == '__main__':
    main()