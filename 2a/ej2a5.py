"""
Enunciado:
Explora el análisis de datos mediante la realización de una regresión lineal y la interpolación de un conjunto de datos.
Este ejercicio se centra en el uso de scipy.optimize para llevar a cabo una regresión lineal y en la aplicación de
scipy.interpolate para la interpolación de datos.

Implementa la función linear_regression_and_interpolation(data_x, data_y) que realice lo siguiente:
    - Regresión Lineal: Ajustar una regresión lineal a los datos proporcionados.
    - Interpolación: Crear una interpolación lineal de los mismos datos.

Adicionalmente, implementa la función plot_results(data_x, data_y, results) para graficar los datos originales,
la regresión lineal y la interpolación.

Parámetros:
    - data_x (List[float]): Lista de valores en el eje x.
    - data_y (List[float]): Lista de valores en el eje y correspondientes a data_x.
    - results (Dict): Resultados de la regresión lineal e interpolación.

Ejemplo:
    - Entrada:
        data_x = np.linspace(0, 10, 100)
        data_y = 3 - data_x + 2 + np.random.normal(0, 0.5, 100) # Datos con tendencia lineal y algo de ruido
    - Ejecución:
        results = linear_regression_and_interpolation(data_x, data_y)
        plot_results(data_x, data_y, results)
    - Salida:
        Un gráfico mostrando los datos originales, la regresión lineal y la interpolación.
"""

from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import typing as t
from sklearn.linear_model import LinearRegression



def linear_regression_and_interpolation(
    data_x: t.List[float], data_y: t.List[float]
) -> t.Dict[str, t.Any]:
    
    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y,dtype=float)

    order = np.argsort(x)
    x_sorted = x[order].reshape(-1, 1)
    y_sorted = y[order]

    lin_reg = LinearRegression()
    model = lin_reg.fit(x_sorted, y_sorted)
    x_new = np.linspace(x_sorted.min(), x_sorted.max(), 20).reshape(-1, 1)
    y_pred = model.predict(x_new)
    slope = lin_reg.coef_[0]
    intercept = lin_reg.intercept_

    interpolated_data = model.predict(x.reshape(-1, 1)) 

    results = {
        "linear_regression": {"slope": slope, "intercept": intercept},
        "interpolated_data": interpolated_data,

        "x_new": x_new.ravel(),
        "y_pred": y_pred,
    }

    return results


def plot_results(data_x: t.List[float], data_y: t.List[float], results: t.Dict):

    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y,dtype=float)

    order = np.argsort(x)
    x_sorted = x[order].reshape(-1, 1)
    y_sorted = y[order]

    lr = results.get("linear_regression")
    if lr and isinstance(lr, dict):
        slope = float(lr.get("slope"))
        intercept = float(lr.get("intercept"))
    else:
        slope = results.get("slope")
        intercept = results.get("intercept")

    x_line = np.linspace(float(x_sorted.min()), float(x_sorted.max()), 100)
    y_line = slope * x_line + intercept

    x_new = results["x_new"]
    y_pred = results["y_pred"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_sorted, y_sorted, s=40, label="Datos", zorder=3)
    ax.plot(x_line, y_line, linewidth=2, label=f"Regresión (y={slope:.3f}x+{intercept:.3f})")
    ax.scatter(x_new, y_pred, s=55, marker="x", label="Predicciones", zorder=4)

    # R2 si viene en results
    r2 = results.get("r2", None)
    if r2 is not None:
        ax.text(0.02, 0.98, f"$R^2$ = {float(r2):.3f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Datos, regresión lineal y predicciones")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")

    plt.show()
    return fig, ax



# Si quieres probar tu código, descomenta las siguientes líneas y ejecuta el script
data_x = np.linspace(0, 10, 100)
data_y = 3 * data_x + 2 + np.random.normal(0, 2, 100)
results = linear_regression_and_interpolation(data_x, data_y)
plot_results(data_x, data_y, results)
