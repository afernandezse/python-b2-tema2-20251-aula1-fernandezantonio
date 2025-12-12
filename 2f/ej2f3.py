"""
Enunciado:
Desarrolla un conjunto de funciones para exportar un DataFrame de Pandas a diferentes formatos de archivo, incluyendo
JSON, CSV y Excel, utilizando Pandas.

Funciones a desarrollar:

- df_to_json(df: pd.DataFrame, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any])
    Descripción:
    Exporta un DataFrame a un archivo JSON y luego lo carga nuevamente, utilizando parámetros específicos que son 
    orient='records' y lines=True, que son retornados para asegurar la consistencia de los datos.
    Parámetros:
        - df (pd.DataFrame): DataFrame a exportar.
        - filename (str): Nombre del archivo de destino para la exportación en formato JSON.

- df_to_csv(df: pd.DataFrame, filename: str) -> (pd.DataFrame, dict):
    Descripción:
    Exporta un DataFrame a un archivo CSV y luego lo carga nuevamente, utilizando parámetros específicos que retornan
    para controlar el delimitador, el encabezado y la codificación. Usa sep=';', header=None, y encoding='utf-8'.
    Parámetros:
        - df (pd.DataFrame): DataFrame a exportar.
        - filename (str): Nombre del archivo de destino para la exportación en formato CSV.

- df_to_excel(df: pd.DataFrame, filename: str) -> (pd.DataFrame, dict):
    Descripción:
    Exporta un DataFrame a un archivo Excel y luego lo carga nuevamente, especificando el nombre de la hoja para la 
    exportación y retornandolo en la función. Utiliza sheet_name='Pandas to Excel'
    Parámetros:
        - df (pd.DataFrame): DataFrame a exportar.
        - filename (str): Nombre del archivo de destino para la exportación en formato Excel.

Ejemplo:
    df_from_json, used_params_json = df_to_json(df_sales, 'data/df_to_json_sales.json')
    df_from_csv, used_params_csv = df_to_csv(df_sales, 'data/df_to_csv_sales.csv')
    df_from_excel, used_params_excel = df_to_excel(df_sales, 'data/sales.xlsx')

Salida esperada:
- Tres DataFrames, cada uno cargado desde un archivo en uno de los formatos especificados: JSON, CSV y Excel. 
Además, se mostrarán los parámetros utilizados para la exportación e importación de cada formato, demostrando cómo los
parámetros específicos influyen en la manipulación de archivos de datos.
"""

from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, Any


def df_to_json(df: pd.DataFrame, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        df.to_json(filename, orient='records', lines=True)
        df_loaded = pd.read_json(filename, orient='records', lines=True)
        used_params = {'orient': 'records', 'lines': True}
        return df_loaded, used_params
    except Exception as e:
        print(f"Error exporting or importing JSON: {e}")
        return pd.DataFrame(), {}


def df_to_csv(df: pd.DataFrame, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        df.to_csv(filename, sep=';', header=None, encoding='utf-8', index=False)
        df_loaded = pd.read_csv(filename, sep=';', header=None, encoding='utf-8')
        used_params = {'sep': ';', 'header': None, 'encoding': 'utf-8'}
        return df_loaded, used_params
    except Exception as e:
        print(f"Error exporting or importing CSV: {e}")
        return pd.DataFrame(), {}


def df_to_excel(df: pd.DataFrame, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        df.to_excel(filename, sheet_name='Pandas to Excel', index=False)
        df_loaded = pd.read_excel(filename, sheet_name='Pandas to Excel')
        used_params = {'sheet_name': 'Pandas to Excel'}
        return df_loaded, used_params
    except Exception as e:
        print(f"Error exporting or importing Excel: {e}")
        return pd.DataFrame(), {}


# Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    path_csv = current_dir / "data/sales.csv"
    df_sales = pd.read_csv(path_csv)

    df_from_json, used_params = df_to_json(
        df_sales, current_dir / "data/df_to_json_sales.json"
    )
    df_from_csv, used_params_csv = df_to_csv(
        df_sales, current_dir / "data/df_to_csv_sales.csv"
    )
    df_from_excel, used_params_excel = df_to_excel(
        df_sales, current_dir / "data/sales.xlsx"
    )

    print(df_from_json.head())
    print(df_from_csv.head())
    print(df_from_excel.head())
