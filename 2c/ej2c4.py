"""
Enunciado:
Desarrolla un conjunto de funciones para realizar análisis agrupado y estandarización de columnas en un conjunto de 
datos de productos. Este conjunto de datos contiene varias características de productos, como nombre del producto,
marca, categoría, precio y calificación. El objetivo es agrupar los datos por categorías y marcas para entender mejor
la distribución de precios y calificaciones dentro de estos grupos, y luego estandarizar las calificaciones dentro de
cada grupo para compararlas de manera justa.

Las funciones a desarrollar son:
- Agrupar y agregar: `group_and_aggregate(df, group_columns, agg_dict)` que agrupa el DataFrame por las columnas
especificadas en `group_columns` y aplica las operaciones de agregación especificadas en `agg_dict`.
- Estandarizar columnas por grupo: `standardize_column_by_group(df, group_columns, column_to_standardize)` que
estandariza los valores de la columna especificada en `column_to_standardize` dentro de cada grupo definido por
`group_columns`, añadiendo una nueva columna al DataFrame con los valores estandarizados.

Parámetros:
- df (pd.DataFrame): DataFrame de Pandas que contiene los datos de los productos.
- group_columns (List[str]): Lista de columnas por las que agrupar el DataFrame.
- agg_dict (Dict[str, List[str]]): Diccionario con las columnas a agregar y las funciones de agregación a aplicar.
- column_to_standardize (str): Nombre de la columna cuyos valores se quieren estandarizar.

Ejemplo de uso:
    aggregated_df = group_and_aggregate(df, ['Category', 'Brand'], {'Price': ['min','max', 'sum'], 'Rating': ['mean']})
    df_standardized = standardize_column_by_group(df, ['Category', 'Brand'], 'Rating')

Salida esperada:
    Un DataFrame agrupado y agregado que muestra la mínima, máxima y suma de precios, así como la media de las 
    calificaciones por cada combinación de categoría y marca. Además, el DataFrame original con una nueva columna que
    muestra las calificaciones estandarizadas dentro de cada grupo de categoría y marca.
"""

from pathlib import Path
import pandas as pd


def group_and_aggregate(df, group_columns, agg_dict):
    aggregated_df = df.groupby(group_columns).agg(agg_dict)
    return aggregated_df

def standardize_column_by_group(df, group_columns, column_to_standardize):
    def standardize(group):
        mean = group[column_to_standardize].mean()
        std = group[column_to_standardize].std(ddof=0)
        group[f"{column_to_standardize}_Standardized"] = (group[column_to_standardize] - mean) / std
        return group

    df_standardized = df.groupby(group_columns).apply(standardize)
    return df_standardized


# Para probar el código, descomenta las siguientes líneas y asegúrate de que el path al archivo sea correcto
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    FILE_PATH = current_dir / "data/products.csv"
    df = pd.read_csv(FILE_PATH)
    group_columns = ["Category", "Brand"]
    agg_dict = {"Price": ["min", "max", "sum"], "Rating": ["mean"]}

    aggregated_df = group_and_aggregate(df, group_columns, agg_dict)
    print(aggregated_df)

    group_columns = ["Category", "Brand"]
    column_to_standardize = "Rating"

    df_standardized = standardize_column_by_group(
        df, group_columns, column_to_standardize
    )
    print(df_standardized.head())
