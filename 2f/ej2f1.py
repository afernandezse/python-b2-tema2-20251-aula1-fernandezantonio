"""
Enunciado:
Desarrolla un conjunto de funciones que permitan cargar, guardar y visualizar el conjunto de datos de vinos mediante
Pandas, Matplotlib, Seaborn y Pickle.

Funciones a desarrollar:
- create_histograms(df: DataFrame, features: List[str]) -> matplotlib.figure.Figure:
    Descripción:
    Genera histogramas para un conjunto de datos de vinos, diferenciando las muestras según su clase de vino.
    Parámetros:
        - df (pd.DataFrame): DataFrame que contiene los datos del conjunto de vinos.
        - features (List[str]): Lista de nombres de las características para las cuales se generarán los histogramas.

- save_img_pickle(fig: Figure, filename: str) -> None:
    Descripción:
    Guarda una figura de Matplotlib en un archivo utilizando Pickle, lo que permite su recuperación y visualización
    posterior sin necesidad de regenerar el gráfico.
    Parámetros:
        - fig (matplotlib.figure.Figure): Figura que se desea guardar.
        - filename (str): Ruta del archivo donde se guardará la figura.

- load_and_display_figure(filename: str) -> matplotlib.figure.Figure:
    Descripción:
    Carga y muestra una figura guardada previamente desde un archivo.
    Parámetros:
        - filename (str): Ruta del archivo donde se encuentra guardada la figura.

Ejemplo:
    df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df_wine['target'] = pd.Categorical.from_codes(wine.target, wine.target_names)

    fig_histograms = create_histograms(df_wine, df_wine.columns[:6])
    save_figure(fig_histograms, 'data/histograms_wine.pickle')

Salida esperada:
- Visualización de histogramas para las primeras seis características del conjunto de datos de vinos, con
diferenciación por clase de vino.
- Guardar en un archivo la figura que contiene el histograma, permitiendo su posterior recuperación y visualización
sin regenerar los gráficos.
"""

from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import List, Union
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame


def create_histograms(df: DataFrame, features: List[str]) -> Figure:
    num_features = len(features)
    cols = 3
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.histplot(data=df, x=feature, hue='target', multiple='stack', ax=axes[i])
        axes[i].set_title(f'Histogram of {feature}')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig


def save_img_pickle(fig: Figure, filename: str) -> None:
    try:
        with open(filename, 'wb') as f:
            pickle.dump(fig, f)
        return True
    except Exception as e:
        print(f"Error saving figure: {e}")
        return False


def load_and_display_figure(filename: str) -> Figure:
    try:
        with open(filename, 'rb') as f:
            fig = pickle.load(f)
        fig.show()
        return fig
    except Exception as e:
        print(f"Error loading figure: {e}")
        return None 


# Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    wine = load_wine()
    df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df_wine["target"] = pd.Categorical.from_codes(wine.target, wine.target_names)

    fig_histograms = create_histograms(df_wine, df_wine.columns[:6])
    is_saved = save_img_pickle(fig_histograms, "data/histograms_wine.pickle")
    print("Figure saved:", is_saved)
    fig_loaded = load_and_display_figure("data/histograms_wine.pickle")
    plt.show()
