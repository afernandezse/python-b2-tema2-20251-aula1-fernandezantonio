"""
Enunciado:
Desarrolla un conjunto de funciones para realizar y visualizar un análisis de regresión utilizando
`sklearn.ensemble.RandomForestRegressor` en el conjunto de datos de viviendas. Este conjunto de datos contiene varias
características de viviendas en áreas suburbanas de Boston, y el objetivo es predecir el valor medio de las viviendas
(`MEDV`) a partir de estas características.

Las funciones a desarrollar son:
1. Preparar los datos: `prepare_data_for_regression(file_path: str)` que lee el archivo CSV, divide los datos en
características (X) y el objetivo (y) siendo `MEDV`. Además, realiza un split de entrenamiento y prueba con una
proporción de 80/20. Al leer el archivo, se deben omitir las primeras 14 filas que contienen información adicional.
2. Realizar la regresión con bosque aleatorio: `perform_random_forest_regression(X_train, y_train)` que entrena un
modelo de bosque aleatorio con los datos de entrenamiento.
3. Evaluar el modelo: `evaluate_regression_model(model, X_test, y_test)` que evalúa el rendimiento del modelo con los
datos de prueba, devolviendo métricas como el R^2 y RMSE.

Parámetros:
    file_path (str): Ruta al archivo CSV que contiene los datos del dataset de viviendas.

Ejemplo de uso:
    model = perform_random_forest_regression(X_train, y_train)
    r_squared, rmse = evaluate_regression_model(model, X_test, y_test)

Salida esperada:
    Los valores de R^2 y RMSE indicando el rendimiento del modelo en el conjunto de prueba.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def prepare_data_for_regression(file_path: str) -> Tuple:
    df = pd.read_csv(file_path, skiprows=14)
    X = df.loc[:, df.columns != 'MEDV']
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def perform_random_forest_regression(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(model, X_test, y_test) -> Tuple[float, float]:
    y_pred = model.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r_squared, rmse


# Para probar el código, debes descomentar las siguientes líneas:
if __name__ == '__main__':
    current_dir = Path(__file__).parent
    file_path = current_dir / 'data/housing.csv'
    X_train, X_test, y_train, y_test = prepare_data_for_regression(file_path)
    model = perform_random_forest_regression(X_train, y_train)
    r_squared, rmse = evaluate_regression_model(model, X_test, y_test)
    print(f'R^2 del modelo: {r_squared}, RMSE: {rmse}')
