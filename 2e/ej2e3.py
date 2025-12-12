"""
Enunciado:
Desarrolla un conjunto de funciones para visualizar el conjunto de datos Iris utilizando visualizaciones de pares y
gráficos de densidad, mediante Seaborn, Matplotlib, y Pandas.

Funciones a desarrollar:
- data_processing( data: np.ndarray, feature_names: List[str], target: Optional[np.ndarray] = None,
target_names: Optional[List[str]] = None, target_feature_name: str = "species") -> pd.DataFrame:
  - Descripción: 
    Prepara un DataFrame de Pandas asignando nombres a las características y etiquetas de clase.
  - Parámetros: 
    - data (np.ndarray): Datos de entrada.
    - feature_names (List[str]): Nombres de las características.
    - target (np.ndarray, opcional): Etiquetas de clase.
    - target_names (List[str], opcional): Nombres de las clases.
    - target_feature_name (str): Nombre de la columna de clase en el DataFrame.

- pairplot_graphic( df: pd.DataFrame, columns: Optional[List[str]] = None, **viz_params: Dict[str, str]) -> 
sns.PairGrid:
  - Descripción: 
    Visualiza gráficos de pares, mediante Seaborn para crear una matriz de gráficos de pares para visualizar las
    relaciones entre características y clases. Recibe columnas específicas o el dataset completo para visualizar.
  - Parámetros: 
    - df (pd.DataFrame): DataFrame de Pandas con los datos a visualizar.
    - columns (List[str], opcional): Columnas específicas o DataFrame para visualizar.
    - **viz_params: Parámetros para personalizar la visualización, como el color por clase, tipo de gráfico
    en la diagonal, etc.

Ejemplo de Uso:
    columns_to_visualize = None # Opcional: ['sepal length (cm)', 'sepal width (cm)']
    plot = pairplot_graphic(df_iris, columns=columns_to_visualize, **viz_params)
    plt.show()

Salida Esperada:
- Gráfica de pares que muestran las relaciones entre cada par de características, con gráficos de densidad en la
diagonal. Las especies se distinguen por colores, para identificar patrones y diferencias entre ellas.
"""

from typing import List, Optional, Dict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


def data_processing(
    data, feature_names, target=None, target_names=None, target_feature_name="species"
):
    df = pd.DataFrame(data, columns=feature_names)
    if target is not None:
        if target_names is not None:
            target_mapped = [target_names[i] for i in target]
            df[target_feature_name] = target_mapped
        else:
            df[target_feature_name] = target
    return df


def pairplot_graphic( df: pd.DataFrame, columns: Optional[List[str]] = None, 
                     **viz_params: Dict[str, str]) -> sns.PairGrid:
    if columns is not None:
        df_to_plot = df[columns + [viz_params.get("hue", "species")]]
    else:
        df_to_plot = df
    pairplot = sns.pairplot(df_to_plot, **viz_params)
    return pairplot


# Para probar el código, descomenta las siguientes líneas
if __name__ == "__main__":
    iris = load_iris()
    df_iris = data_processing(
        iris.data, iris.feature_names, iris.target, iris.target_names
    )

    viz_params = {
        "hue": "species",
        "diag_kind": "kde",
        "kind": "scatter",
        "palette": "husl",
        "corner": True,
    }
    columns_to_visualize = None
    plot = pairplot_graphic(df_iris, columns=columns_to_visualize, **viz_params)
    plt.show()

