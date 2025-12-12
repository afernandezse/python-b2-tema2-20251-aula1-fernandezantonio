"""
Enunciado:
Desarrolla un conjunto de funciones para realizar análisis y visualización de datos sobre el conjunto de datos Iris,
utilizando Pandas y Matplotlib. El objetivo es explorar las características de las especies de iris mediante gráficos.

diccionario_colores = {0: 'green', 1: 'red', 2: 'blue'}.

Funciones a desarrollar:
    
- plot_area_graph(df, column_name, ax=None) -> None
    Descripción: Genera un gráfico de área para visualizar la distribución de un atributo específico entre las diferentes
    especies de iris. El eje X representa el índice del DataFrame, y el eje Y representa el valor del atributo especificado
    por column_name. Las áreas son coloreadas según diccionario_colores.
    Parámetros:
    df: DataFrame que contiene los datos.
    column_name: Nombre de la columna cuyos datos se quieren visualizar en el gráfico de área.

- plot_scatter_graph(df, column_name_x, column_name_y, ax=None) -> None
    Descripción: Crea un gráfico de dispersión para comparar dos atributos entre las diferentes especies de iris. Los ejes
    representan los valores de los atributos especificados por column_name_x y column_name_y, con los puntos coloreados según
    diccionario_colores.
    Parámetros:
    df: DataFrame que contiene los datos.
    column_name_x: Nombre de la columna que se quiere visualizar en el eje X del gráfico de dispersión.
    column_name_y: Nombre de la columna que se quiere visualizar en el eje Y del gráfico de dispersión.

Ejemplo:
    plot_area_graph(dataframe, 'petal length (cm)')
    plot_scatter_graph(dataframe, 'sepal length (cm)', 'sepal width (cm)')

Salida esperada:
- Un gráfico de área que muestre la distribución de la longitud del pétalo para las diferentes especies de iris.
- Un gráfico de dispersión que compare la longitud y el ancho del sépalo entre las diferentes especies de iris.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_area_graph(df, column_name, ax=None):
    df_sorted = df.sort_values(by=column_name)
    species = df_sorted['target'].unique()
    area_data = {specie: df_sorted[df_sorted['target'] == specie][column_name].values for specie in species}
    area_df = pd.DataFrame(area_data)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    area_df.plot.area(ax=ax, color=['green', 'red', 'blue'])
    ax.set_title(f'Area Graph of {column_name}')
    ax.set_xlabel('Index')
    ax.set_ylabel(column_name)
    return fig, ax


def plot_scatter_graph(df, column_name_x, column_name_y, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    color_map = {0: 'green', 1: 'red', 2: 'blue'}
    species = sorted(df['target'].unique())
    for specie in species:
        subset = df[df['target'] == specie]
        ax.scatter(
            subset[column_name_x],
            subset[column_name_y],
            label=str(specie),
            color=color_map[specie]
        )
    ax.set_title(f'Scatter Plot of {column_name_x} vs {column_name_y}')
    ax.set_xlabel(column_name_x)
    ax.set_ylabel(column_name_y)
    ax.legend()
    return fig, ax



# Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    path_csv = current_dir / "data/iris_dataset.csv"
    dataframe = pd.read_csv(path_csv)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    plot_area_graph(dataframe, "petal length (cm)", ax=axs[0])
    plot_scatter_graph(dataframe, "sepal length (cm)", "sepal width (cm)", ax=axs[1])

    plt.tight_layout()
    plt.show()
