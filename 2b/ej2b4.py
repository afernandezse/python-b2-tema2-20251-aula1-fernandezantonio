"""
Enunciado:
Desarrolla un conjunto de funciones para leer y procesar datos de archivos Excel utilizando Pandas, abordando
distintos escenarios comunes en el análisis de datos. Se proporciona un archivo Excel y se deberán completar las
funciones para leer y formatear los datos.

Las funciones y escenarios a desarrollar son:
    - Leer datos desde una hoja específica: read_excel_sheet(file_path) que lee todos los registros de la hoja Sheet1
    en el archivo Excel y devuelve un DataFrame de Pandas.
    - Leer y limpiar datos de la hoja Sheet2 con encabezados desplazados, filas y columnas adicionales:
    read_excel_custom_sheet(file_path) que lee la hoja, ajusta los encabezados de las columnas y omite las
    filas y columnas sin valor, adicional, evitar las filas al final de la hoja.

Parámetros:
    - file_path (str): Ruta al archivo Excel.

Cada función debe devolver un DataFrame de Pandas, permitiendo a los estudiantes ver el efecto de diferentes modos de
lectura de datos desde un archivo Excel.

Ejemplo:
    df_from_sheet1 = read_excel_sheet(excel_file_path)
    df_from_sheet2 = read_excel_custom_sheet(excel_file_path)

Salida esperada:
    Un DataFrame de Pandas para cada función.
"""

import pandas as pd
from pathlib import Path


def read_excel_sheet(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path, sheet_name='Sheet1')



def read_excel_custom_sheet(file_path: str) -> pd.DataFrame:
    def is_not_empty_column(col):
        return not col.isnull().all()
    df = pd.read_excel(file_path, sheet_name="Sheet2", header=3, skipfooter=4)
    use_cols = [
        col for col in df.columns if is_not_empty_column(df[col])
    ]
    df = df[use_cols]
    return df


# Para probar el código, descomenta las siguientes líneas
file_path = "data/ej2b4/ramen-ratings.xlsx"
current_dir = Path(__file__).parent
excel_file_path = current_dir / file_path

df_from_sheet1 = read_excel_sheet(excel_file_path)
df_from_sheet2 = read_excel_custom_sheet(excel_file_path)

# Mostrar la cantidad de registros y los nombres de las columnas
print(f"Registros en la hoja 1: {len(df_from_sheet1)}")
print(f"Nombres de columnas en la hoja 1: {df_from_sheet1.columns.tolist()}")
print(f"Registros en la hoja 2: {len(df_from_sheet2)}")
print(f"Nombres de columnas en la hoja 2: {df_from_sheet2.columns.tolist()}")
