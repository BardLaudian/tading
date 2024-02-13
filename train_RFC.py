from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from joblib import dump

# Cargar el dataset
df = pd.read_csv('complete_dataset.csv', index_col=0, parse_dates=True)

# Dividir el dataset en características (X) y etiquetas (y)
X = df.drop(columns=['Etiqueta'])
y = df['Etiqueta']

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Bosque Aleatorio con los mejores parámetros obtenidos
rf_model = RandomForestClassifier(
    n_estimators=100,  # Número de árboles
    max_depth=None,  # Profundidad máxima de los árboles
    min_samples_split=10,  # Número mínimo de muestras requeridas para dividir un nodo interno
    min_samples_leaf=4,  # Número mínimo de muestras requeridas para estar en un nodo hoja
    max_features='sqrt',  # Número de características a considerar al buscar la mejor división
    random_state=42,
    class_weight='balanced'  # Ajuste de pesos para clases desequilibradas
)

# Entrenar el modelo con los datos de entrenamiento
rf_model.fit(X_train, y_train)

# Realizar predicciones con el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Evaluar el rendimiento del modelo con el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Imprimir la precisión, la matriz de confusión y el reporte de clasificación
print("Precisión del modelo:", accuracy)
print("\nMatriz de Confusión:")
print(conf_matrix)
print("\nReporte de Clasificación:")
print(class_report)

# Exportar el modelo entrenado a un archivo
dump(rf_model, 'rf_model.joblib')
