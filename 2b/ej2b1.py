"""
Enunciado:
Desarrolla un conjunto de funciones para leer y procesar datos de archivos CSV utilizando Pandas, abordando distintos
escenarios comunes en el análisis de datos mediante Pandas y la funsion "read_csv()". Se proporciona un archivo CSV y se
deberán completar las funciones para leer y formatear los datos.

Las funciones y escenarios a desarrollar son:
    - Leer datos desde un archivo CSV básico: read_csv_basic(file_path) que lee un archivo CSV sencillo y devuelve un
    DataFrame de Pandas.
    - Leer CSV con encabezado no estándar: read_csv_header_issue(file_path, header_row) que lee un archivo CSV donde los
    datos comienzan después de la cuarta fila.
    - Leer CSV con multi-índice: read_csv_multi_index(file_path, index_cols) que lee un archivo CSV y utiliza múltiples
    columnas como índices.
    - Leer CSV con separador no estándar: read_csv_custom_separator(file_path, separator) que lee un archivo CSV
    utilizando un separador diferente al predeterminado y separador decimal con coma. Además, esta función debe
    convertir la columna 'Stars' a tipo flotante.

Parámetros:
    file_path (str): Ruta del archivo CSV.
    header_row (int): Número de fila donde comienzan los datos (para read_csv_header_issue).
    index_cols (list): Lista de columnas para usar como índices (para read_csv_multi_index).
    separator (str): Separador utilizado en el CSV (para read_csv_custom_separator).

Cada función debe devolver un DataFrame de Pandas, permitiendo a los estudiantes ver el efecto de diferentes modos de lectura de archivos CSV.

Ejemplo:
    df_basic = read_csv_basic(basic_csv_path)
    df_header_issue = read_csv_header_issue(header_issue_csv_path, header_row=3)
    df_multi_index = read_csv_multi_index(multi_index_csv_path, index_cols=['Brand', 'Style'])
    df_semicolon = read_csv_custom_separator(semicolon_csv_path, separator=';', decimal=',')

    # Mostrar los primeros registros de cada DataFrame
    print(df_basic.head(), df_header_issue.head(), df_multi_index.head(), df_semicolon.head())

Salida esperada:
    Un DataFrame de Pandas para cada función.
"""

import pandas as pd
import typing as t
from pathlib import Path


def read_csv_basic(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def read_csv_header_issue(file_path: str, header_row: int) -> pd.DataFrame:
    return pd.read_csv(file_path, header=header_row)


def read_csv_multi_index(file_path: str, index_cols: t.List[str]) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col=index_cols)

def read_csv_custom_separator(
    file_path: str, separator: str, decimal: str
) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=separator, decimal=decimal)
    df['Stars'] = pd.to_numeric(df['Stars'], errors='coerce')
    return df


# Para probar el código, descomenta las siguientes líneas
current_dir = Path(__file__).parent
BASIC_CSV_PATH = current_dir / "data/ej2b1/ramen-ratings.csv"
HEADER_ISSUE_CSV_PATH = current_dir / "data/ej2b1/ramen_ratings_with_header_issue.csv"
MULTI_INDEX_CSV_PATH = current_dir / "data/ej2b1/ramen_ratings_multi_index.csv"
SEMICOLON_CSV_PATH = current_dir / "data/ej2b1/ramen_ratings_decimal_comma.csv"

df_basic = read_csv_basic(BASIC_CSV_PATH)
df_header_issue = read_csv_header_issue(HEADER_ISSUE_CSV_PATH, header_row=3)
df_multi_index = read_csv_multi_index(
    MULTI_INDEX_CSV_PATH, index_cols=["Brand", "Style"]
)
df_semicolon = read_csv_custom_separator(SEMICOLON_CSV_PATH, separator=";", decimal=",")

# Mostrar los primeros registros de cada DataFrame
print(
    df_basic.head(), df_header_issue.head(), df_multi_index.head(), df_semicolon.head()
)
