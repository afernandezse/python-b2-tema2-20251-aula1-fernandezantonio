"""
Enunciado:
Desarrolla una función para analizar la correlación entre dos características del conjunto de datos de viviendas
utilizando `scipy.stats.pearsonr`. Se proporciona un archivo CSV 'housing.csv' que contiene datos sobre características
de viviendas La tarea es completar la función para calcular el coeficiente de correlación de Pearson, lo cual ayudará a
determinar si existe una relación lineal entre las dos variables seleccionadas.

La función a desarrollar es:
    - Calcular el coeficiente de correlación de Pearson: calculate_pearson_correlation(file_path, variable_1, var2) que lee
    un archivo CSV y calcula la correlación entre dos variables dadas.

Parámetros:
    file_path (str): Ruta del archivo CSV que contiene los datos de vivienda.
    variable_1 (str): Nombre de la primera variable para el cálculo de la correlación.
    var2 (str): Nombre de la segunda variable para el cálculo de la correlación.

La función debe devolver el valor del coeficiente de correlación de Pearson y el valor p asociado, permitiendo a los
estudiantes comprender la relación entre las dos variables seleccionadas. Las descripciones que corresponde a cada
columna se encuentra en las primeras 14 filas del archivo CSV, por lo tanto, deberás saltar estas filas al leer el
archivo.


Ejemplo:
    housing_csv_path = 'path_a_tu_archivo_housing.csv'
    correlation, p_value = calculate_pearson_correlation(housing_csv_path, 'MEDV', 'RM')

    # Mostrar el coeficiente de correlación de Pearson y el valor p
    print(f'Correlación de Pearson: {correlation}, Valor p: {p_value}')

Salida esperada:
    El valor del coeficiente de correlación de Pearson y el valor p entre las dos variables seleccionadas.
"""

from pathlib import Path
import pandas as pd
from scipy import stats


def calculate_pearson_correlation(
    file_path: str, var1: str, var2: str
) -> (float, float):
    df = pd.read_csv(file_path, skiprows=14)
    df_clean = df[[var1, var2]].dropna()
    correlation, p_value = stats.pearsonr(df_clean[var1], df_clean[var2])
    return correlation, p_value


# Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    HOUSING_CSV_PATH = current_dir / 'data/housing.csv'
    variable_1 = 'MEDV'
    variable_2 = 'RM'
    correlation, p_value = calculate_pearson_correlation(HOUSING_CSV_PATH, variable_1, variable_2)

    # Mostrar el coeficiente de correlación de Pearson y el valor p
    print(f'Columnas comparadas: {variable_1} y {variable_2}')
    print(f'Correlación de Pearson: {correlation}, Valor p: {p_value}')
