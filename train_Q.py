import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.initializers import HeUniform
from collections import deque

# Cargar datos
df = pd.read_csv('complete_dataset.csv')

# Convertir 'timestamp' de cadena a datetime y extraer características de tiempo
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Eliminar la columna 'timestamp' original
df.drop(columns=['timestamp'], inplace=True)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class TradingEnvironment:
    def __init__(self, data, initial_balance=1000, min_episode_length=200):
        self.data = data
        self.initial_balance = initial_balance
        self.min_episode_length = min_episode_length
        self.current_price = self.data['close'].iloc[0]
        self.reset()
    
    def reset(self):
        max_start_point = len(self.data) - self.min_episode_length
        self.current_step = random.randint(0, max_start_point) if max_start_point > 0 else 0
        self.total_balance = self.initial_balance
        self.eth_held = 0
        self.current_price = self.data['close'].iloc[self.current_step]
        self.done = False
        return self._get_state()
    
    def step(self, action):
        self.current_price = self.data['close'][self.current_step]
        reward = 0
        commission_rate = 0.001

        if action == 1 and self.total_balance > 0:
            # Combinar cálculos para reducir la sobrecarga
            self.eth_held += (self.total_balance * (1 - commission_rate)) / self.current_price
            self.last_purchase_price = self.current_price
            self.total_balance = 0

        elif action == 2 and self.eth_held > 0:
            # Simplificar el cálculo de la recompensa y la actualización del balance
            reward = (self.current_price - self.last_purchase_price) * self.eth_held * (1 - commission_rate)
            self.total_balance = reward
            self.eth_held = 0

        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        return self._get_state(), reward, self.done
    
    def _get_state(self):
        market_state = self.data.iloc[self.current_step].to_numpy()
        portfolio_state = [self.total_balance, self.eth_held * self.current_price]
        return np.concatenate([market_state, portfolio_state])

def create_dqn_model(input_shape, action_space, dropout_rate=0.1, regularization_factor=0.01):
    model = Sequential([
        Dense(64, input_dim=input_shape, activation='relu', kernel_regularizer=l2(regularization_factor), kernel_initializer=HeUniform()),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(32, activation='relu', kernel_regularizer=l2(regularization_factor), kernel_initializer=HeUniform()),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(action_space, activation='linear')  # No BatchNormalization before the output layer.
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def train_dqn(env, episodes=100, update_target_every=10):
    action_space = 3
    state_size = len(env._get_state())
    model = create_dqn_model(state_size, action_space)
    target_model = create_dqn_model(state_size, action_space)
    target_model.set_weights(model.get_weights())

    replay_buffer = ReplayBuffer(10000)
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    gamma = 0.95

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        while True:
            action = random.choice([0, 1, 2]) if np.random.rand() <= epsilon else np.argmax(model.predict(state, verbose=0)[0])
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if done:
                epsilon = max(epsilon_min, epsilon_decay * epsilon)
                break

            if len(replay_buffer.buffer) >= batch_size:
                minibatch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

                # Asegurar que los estados tengan la forma correcta
                states = np.squeeze(states)
                next_states = np.squeeze(next_states)

                targets = rewards + gamma * np.amax(target_model.predict_on_batch(next_states), axis=1) * (1 - dones)
                target_f = model.predict_on_batch(states)
                for i, action in enumerate(actions):
                    target_f[i][action] = targets[i]

                model.train_on_batch(states, target_f)

        if episode % update_target_every == 0:
            target_model.set_weights(model.get_weights())

        print(f'Episode: {episode+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}')

    model.save('dqn_trading_model_optimized.h5')

# Suponiendo que tienes un 'env' listo para usar
env = TradingEnvironment(df)
train_dqn(env, 500)