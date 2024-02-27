import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from collections import deque

# Cargar datos desde CSV
df = pd.read_csv('processed_dataset.csv', index_col=0)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Definir el tamaño del buffer y el tamaño del lote
buffer_size = 10000
batch_size = 32
replay_buffer = ReplayBuffer(buffer_size)

# Definición del Entorno de Trading
class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = 0
        self.total_balance = self.initial_balance
        self.eth_held = 0
        self.initial_price = self.data['close'][self.current_step]
        self.current_price = self.initial_price
        self.done = False
        return self._get_state()

    def step(self, action):
        self.current_price = self.data['close'][self.current_step]

        if action == 1:  # Comprar
            self.eth_held += self.total_balance / self.current_price
            self.total_balance = 0
        elif action == 2 and self.eth_held > 0:  # Vender
            self.total_balance += self.eth_held * self.current_price
            self.eth_held = 0

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        new_total_balance = self.total_balance + self.eth_held * self.current_price
        reward = new_total_balance - self.initial_balance
        next_state = self._get_state()
        return next_state, reward, self.done

    def _get_state(self):
        market_state = self.data.iloc[self.current_step].to_numpy()
        portfolio_state = [self.total_balance, self.eth_held * self.current_price]
        return np.concatenate([market_state, portfolio_state])

# Definición del Modelo DQN
def create_dqn_model(input_shape, action_space):
    model = Sequential([
        Dense(64, input_dim=input_shape, activation='relu'),
        Dense(32, activation='relu'),
        Dense(action_space, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# Función de Entrenamiento
def train_dqn(env, episodes=100, update_frequency=20, batch_size=16):
    action_space = 3  # Mantener, Comprar, Vender
    state_size = env._get_state().shape[0]
    model = create_dqn_model(state_size, action_space)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0

        while not env.done:
            action = np.random.choice(action_space)  # Estrategia de acción aleatoria
            next_state, reward, done = env.step(action)

            # Almacenar la experiencia en el buffer
            replay_buffer.add((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward
            step_count += 1

            # Actualizar el modelo con menos frecuencia y con lotes más pequeños
            if step_count % update_frequency == 0 and len(replay_buffer.buffer) >= batch_size:
                minibatch = replay_buffer.sample(batch_size)
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                        target = reward + 0.95 * np.amax(model.predict(next_state.reshape([1, state_size]), verbose=0)[0])
                    target_f = model.predict(state.reshape([1, state_size]), verbose=0)
                    target_f[0][action] = target
                    model.fit(state.reshape([1, state_size]), target_f, epochs=1, verbose=0)

        if episode % 10 == 0:
            print(f'Episodio: {episode + 1}, Recompensa total: {total_reward}')

    # Guardar el modelo entrenado
    model.save('dqn_trading_model_optimized.h5')
    print("Modelo DQN optimizado guardado como 'dqn_trading_model_optimized.h5'.")

# Inicialización y Entrenamiento
env = TradingEnvironment(df)
train_dqn(env)