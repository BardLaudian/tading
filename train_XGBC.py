import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from joblib import dump

# Cargar el dataset
df = pd.read_csv('complete_dataset.csv', index_col=0, parse_dates=True)

# Dividir el dataset en características (X) y etiquetas (y)
X = df.drop(columns=['Etiqueta'])
y = df['Etiqueta']

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuración actualizada de parámetros para XGBoost, incluyendo el uso de la GPU
params = {
    'tree_method': 'hist',  # Cambiado a 'hist'
    'device': 'cuda',  # Especifica el uso de CUDA para entrenamiento en GPU
    'n_estimators': 200,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 1,
    'colsample_bytree': 0.8
}

# Crear y entrenar el modelo XGBoost con soporte para GPU y los parámetros actualizados
xgb_model = xgb.XGBClassifier(**params)
xgb_model.fit(X_train, y_train)

# Realizar predicciones y evaluar el modelo
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Precisión del modelo XGBoost con GPU:", accuracy)
print("\nMatriz de Confusión:")
print(conf_matrix)

# Exportar el modelo XGBoost entrenado a un archivo
dump(xgb_model, 'xgb_model_gpu.joblib')
