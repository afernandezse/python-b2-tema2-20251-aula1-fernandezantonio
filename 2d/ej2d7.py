"""
Enunciado:
Desarrolla un conjunto de funciones para realizar y visualizar un análisis de clustering utilizando
`sklearn.cluster.KMeans` en el conjunto de datos de crédito alemán. Este conjunto de datos contiene varias
características financieras y personales de los solicitantes de crédito, y el objetivo es agrupar a los solicitantes
en k clusters basados en sus características para identificar patrones y perfiles de riesgo.

Las funciones a desarrollar son:
1. Preparar los datos: `prepare_data_for_clustering(file_path: str)` que lee el archivo CSV y prepara los datos para
   el clustering, realizando cualquier limpieza o preprocesamiento necesario.
2. Realizar el clustering con KMeans: `perform_kmeans_clustering(data, n_clusters: int)` que aplica el algoritmo KMeans
   a los datos preparados, utilizando un número específico de clusters.
3. Visualizar los resultados: `visualize_clusters(data, labels, is_testing_execution)` que visualiza los resultados del
   clustering, idealmente utilizando una reducción de dimensionalidad para representar los datos en 2D o 3D si es
   posible.

Parámetros:
    file_path (str): Ruta al archivo CSV que contiene los datos del dataset de crédito alemán.
    n_clusters (int): Número de clusters a utilizar en el análisis KMeans.

Ejemplo:
    data = prepare_data_for_clustering('./data/german_credit_data.csv')
    labels = perform_kmeans_clustering(data, 5)
    visualize_clusters(data, labels, is_testing_execution)

Salida esperada:
    Una visualización de los clusters formados por el análisis KMeans, mostrando cómo se agrupan los solicitantes de
    crédito según sus características.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def prepare_data_for_clustering(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    num_df = df.select_dtypes(include=[np.number]).dropna()
    mean = num_df.mean(axis=0)
    std = num_df.std(axis=0, ddof=1)
    data_scaled = (num_df - mean) / std
    df_scaled = pd.DataFrame(data_scaled, columns=num_df.columns)
    return df_scaled


def perform_kmeans_clustering(data: np.ndarray, n_clusters: int) -> np.ndarray:
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return labels


def visualize_clusters(
    data: np.ndarray, labels: np.ndarray, is_testing_execution: bool = False
) -> Tuple[np.ndarray, plt.Figure, plt.Axes]:
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_title('KMeans Clustering Visualization')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')

    if not is_testing_execution:
        plt.show()

    return data_reduced, fig, ax


# Para probar el código, desconmenta las siguientes líneas
if __name__ == '__main__':
    current_dir = Path(__file__).parent
    file_path = current_dir / 'data/german_credit_data.csv'
    data = prepare_data_for_clustering(file_path)
    labels = perform_kmeans_clustering(data, 5)
    data_reduced, fig, ax = visualize_clusters(data, labels, False)
