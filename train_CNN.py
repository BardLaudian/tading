import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import tensorflow as tf

# Comprobar disponibilidad de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Establecer la asignación de memoria dinámica
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("CUDA (GPU support) is available and enabled!")
else:
    print("CUDA (GPU support) is not available. Falling back to CPU.")

# Importar el archivo CSV
df = pd.read_csv('complete_dataset.csv', index_col=0, parse_dates=True)

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(df)

# Definir función para preparar los datos
def prepare_data(data, backcandles):
    X, y = [], []
    for i in range(backcandles, len(data)):
        X.append(data[i-backcandles:i, :-1])  # Todas las columnas excepto la última
        y.append(data[i, -1])  # Última columna (variable a predecir)
    return np.array(X), np.array(y)

# Configuración de hiperparámetros
backcandles = 30
input_size = data_normalized.shape[1] - 1  # Número de características (excluyendo la variable a predecir)
hidden_layer_size = 100
output_size = 1

# Preparar datos
X, y = prepare_data(data_normalized, backcandles)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir el modelo LSTM
def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_size))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Número de splits para K-Fold
n_splits = 5

# Preparar KFold
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Definir Early Stopping
early_stopping = EarlyStopping(patience=5, monitor='val_loss', mode='min', restore_best_weights=True)

# Bucle sobre los folds
fold_no = 1
for train, test in kfold.split(X, y):
    model = create_model((backcandles, input_size))

    # Entrenar el modelo
    print(f'Training on fold {fold_no}...')
    history = model.fit(X[train], y[train], epochs=100, batch_size=64, verbose=1, validation_data=(X[test], y[test]), callbacks=[early_stopping])

    # Evaluación del modelo
    train_loss = model.evaluate(X[train], y[train], verbose=0)
    test_loss = model.evaluate(X[test], y[test], verbose=0)
    print(f'Fold {fold_no} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    # Incrementar el número de fold
    fold_no += 1

# Evaluación del modelo
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Train Loss: {train_loss:.4f}')
print(f'Test Loss: {test_loss:.4f}')

model.save('mi_modelo_cnn.h5')

import joblib

# Guardar el scaler ajustado para usarlo más tarde
joblib.dump(scaler, 'scaler.save')

# Realizar la predicción
y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
temp_array = np.zeros((y_test.shape[0], 13))
temp_array[:, -1] = y_test.ravel()
y_pred_original_full = scaler.inverse_transform(temp_array)
y_test = y_pred_original_full[:, -1]

temp_array = np.zeros((y_pred.shape[0], 13))
temp_array[:, -1] = y_pred.ravel()
y_pred_original_full = scaler.inverse_transform(temp_array)
y_pred = y_pred_original_full[:, -1]
# Graficar la solución en comparación con la predicción
plt.figure(figsize=(16,8))
plt.plot(y_test[-20:], color='black', label='Test')
plt.plot(y_pred[-20:], color='green', label='Pred')
plt.legend()
plt.show()