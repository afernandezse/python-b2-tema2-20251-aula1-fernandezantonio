"""
Enunciado:
Desarrolla un conjunto de funciones para realizar una clasificación de especies de flores Iris utilizando el algoritmo
Support Vector Machine (SVM) de la biblioteca sklearn. El archivo CSV 'iris_dataset.csv' contiene los datos relevantes
para este análisis, incluyendo características como el largo y ancho del sépalo y pétalo, y la especie de la flor Iris
como objetivo de clasificación.

Las funciones a desarrollar son:
1. Preparar los datos: `prepare_data(file_path: str)` que lee el archivo CSV, divide los datos en características (X)
y objetivo (y), y realiza un split de entrenamiento y prueba con una proporción de 80/20.
2. Realizar la clasificación con SVM: `perform_svm_classification(X_train, y_train, X_test, y_test)` que entrena un
modelo SVM con los datos de entrenamiento y evalúa su rendimiento con los datos de prueba, devolviendo las métricas
de precisión y un reporte de clasificación.
3. Entrenar el clasificador SVM: train_svm_classifier(X_train, y_train) toma las características y las etiquetas de
entrenamiento como entradas, entrena un clasificador de vectores de soporte (SVM) usando un kernel lineal, y devuelve
el clasificador entrenado.
4. Predecir especies: predict_species(clf, features, feature_names) recibe un clasificador SVM entrenado, un conjunto
de características numéricas y sus nombres correspondientes. Crea un DataFrame con estas características, realiza una
predicción usando el clasificador, y devuelve el nombre de la especie predicha basándose en el índice de la predicción.

Parámetros:
    file_path (str): Ruta al archivo CSV que contiene los datos del dataset Iris.
    X_train, X_test, y_train, y_test: Datos divididos para entrenamiento y pruebas.

Ejemplo de uso:
    X_train, X_test, y_train, y_test = prepare_data('iris_dataset.csv')
    accuracy, classification_report = perform_svm_classification(X_train, y_train, X_test, y_test)

Salida esperada:
    La precisión del modelo SVM en el conjunto de prueba y un reporte de clasificación que incluye precision, recall
    y f1-score para cada clase.
"""

from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def prepare_data(file_path: str) -> Tuple:
    df = pd.read_csv(file_path)
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def perform_svm_classification(X_test, y_test, clf) -> Tuple[float, str]:
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


def train_svm_classifier(X_train, y_train) -> SVC:
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    return clf


def predict_species(clf: SVC, features: List[float], feature_names: List[str]) -> str:
    input_data = pd.DataFrame([features], columns=feature_names)
    prediction = clf.predict(input_data)
    class_idx = clf.classes_.tolist().index(prediction[0])
    species_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    species = species_names[class_idx]
    return species


target_names = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}

# Para probar el código, descomenta este código
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    file_path = current_dir / 'data/iris_dataset.csv'
    X_train, X_test, y_train, y_test = prepare_data(file_path)
    clf = train_svm_classifier(X_train, y_train)

    # Precisión y reporte del modelo
    accuracy, report = perform_svm_classification(X_test, y_test, clf)
    print(f'Precisión del modelo: {accuracy}')
    print('Reporte de clasificación:')
    print(report)

    # Realizar una predicción con un nuevo conjunto de características
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    features = [5.1, 3.5, 1.4, 0.2]  # Ejemplo de características
    species = predict_species(clf, features, feature_names)
    print(f'La especie predicha es: {species}')
