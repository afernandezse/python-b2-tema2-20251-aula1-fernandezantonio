"""
Enunciado:
Desarrolla un conjunto de funciones enfocadas en el entrenamiento, almacenamiento, carga y evaluación de un modelo de
clasificación utilizando el conjunto de datos de vinos y la implementación de un clasificador de bosque aleatorio de la
biblioteca Scikit-learn, destacando la persistencia de modelos mediante Pickle. 

Funciones a desarrollar:
- train_model(    X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42) ->
(RandomForestClassifier, np.ndarray, np.ndarray):
    Descripción:
    Entrena un modelo de clasificación de bosque aleatorio dividido en entrenamiento y prueba.
    Parámetros:
        - X (np.ndarray): Datos de las características.
        - y (np.ndarray): Etiquetas objetivo.
        - test_size (float): Proporción del conjunto de datos a utilizar como conjunto de prueba.
        - random_state (int): Semilla para la generación de números aleatorios para reproducibilidad.

- save_model(model: BaseEstimator, filename: str) -> bool:
    Descripción:
    Guarda el modelo entrenado en un archivo utilizando el módulo Pickle para su posterior recuperación.
    Parámetros:
        - model (RandomForestClassifier): Modelo entrenado.
        - filename (str): Ruta del archivo donde se guardará el modelo.

- load_model_and_predict(filename: str, X_test: np.ndarray) -> np.ndarray:
    Descripción:
    Carga un modelo previamente guardado y realiza predicciones sobre un conjunto de datos de prueba proporcionado.
    Parámetros:
        - filename (str): Ruta del archivo donde se encuentra el modelo guardado.
        - X_test (np.ndarray): Datos de prueba para realizar predicciones.

- plot_feature_importance(model: BaseEstimator, feature_names: List[str], figsize: Tuple[int, int] = (12, 8))
-> matplotlib.figure.Figure:
    Descripción:
    Genera y visualiza un gráfico de las características determinadas por el modelo de bosque aleatorio.
    Parámetros:
        - model (RandomForestClassifier): Modelo del cual se extrae las características.
        - feature_names (List[str]): Nombres de las características del conjunto de datos.
        - figsize (tuple): Dimensiones de la figura a generar.

Ejemplo:
    model, X_test, y_test = train_model(X, y)
    filename = 'data/wine_model.pickle'
    save_model(model, filename)
    predictions = load_model_and_predict(filename, X_test)
    fig = plot_feature_importance(model, wine.feature_names)

Salida esperada:
- Modelo de clasificación de bosque aleatorio entrenado, capaz de predecir las clases del conjunto de datos de
vinos con una determinada precisión.
- Persistencia del modelo entrenado utilizando Pickle.
- Visualización de las características según el modelo al realizar predicciones.
"""

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Any
from sklearn.base import BaseEstimator


def train_model(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42
) -> Tuple[BaseEstimator, np.ndarray, np.ndarray]:
    model = RandomForestClassifier(random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def save_model(model: BaseEstimator, filename: str) -> bool:
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def load_model_and_predict(filename: str, X_test: np.ndarray) -> np.ndarray:
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        predictions = model.predict(X_test)
        return predictions
    except Exception as e:
        print(f"Error loading model or predicting: {e}")
        return np.array([])


def plot_feature_importance(
    model: BaseEstimator, feature_names: List[str], figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(importances)), importances[indices], align='center')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
    ax.set_title('Feature Importances')
    ax.set_ylabel('Importance Score')
    plt.tight_layout()
    return fig


# Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    wine = load_wine()
    X = wine.data
    y = wine.target

    model, X_test, y_test = train_model(X, y)
    filename = "wine_model.pickle"
    is_saved = save_model(model, filename)
    print("Model saved:", is_saved)
    fig = plot_feature_importance(model, wine.feature_names)
    plt.show()
