"""
Enunciado:

Explora el análisis avanzado de datos y la aplicación de ajustes no lineales mediante el uso de SciPy. Este ejercicio se
centra en ajustar una función gaussiana a un conjunto de datos usando el módulo scipy.optimize.curve_fit y en calcular
la integral de esta curva con scipy.integrate.quad.

Implementar la función gaussian_fit_and_integration(data_x, data_y) que realice lo siguiente:
    Ajuste de Curva Gaussiana: Utilizar scipy.optimize.curve_fit para ajustar una curva gaussiana a los datos.
    Integración Numérica: Calcular la integral de la curva gaussiana ajustada sobre el rango de data_x utilizando
    scipy.integrate.quad.

Además, implementar la función plot_gaussian_fit(data_x, data_y, gaussian_params) para visualizar los datos originales
y la curva gaussiana ajustada.

Parámetros:
    data_x (List[float]): Lista de valores en el eje x.
    data_y (List[float]): Lista de valores en el eje y correspondientes a data_x.
    gaussian_params (Tuple[float]): Parámetros (amplitud, centro, ancho) de la curva gaussiana ajustada.

Ejemplo:
    Entrada:
        data_x = np.linspace(-5, 5, 100)
        data_y = 3 * np.exp(-(data_x - 1)**2 / (2 * 1.5**2)) + np.random.normal(0, 0.2, 100)
    Ejecución:
        gaussian_params, integral = gaussian_fit_and_integration(data_x, data_y)
        plot_gaussian_fit(data_x, data_y, gaussian_params)
    Salida:
        Un gráfico mostrando los datos originales y la curva gaussiana ajustada.
        La integral de la curva gaussiana ajustada.
"""

from scipy import optimize, integrate
import numpy as np
import matplotlib.pyplot as plt
import typing as t


def gaussian(x: float, amplitude: float, mean: float, stddev: float) -> float:
    return amplitude * np.exp(-(x-mean)**2 / (2*stddev**2))

def gaussian_fit_and_integration(
    data_x: t.List[float], data_y: t.List[float]
) -> t.Tuple[t.Tuple[float], float]:

    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)

    initial_guess = [np.max(y), x[np.argmax(y)], 1.0]
    params, _ = optimize.curve_fit(gaussian, x, y, p0=initial_guess)

    amplitude, mean, stddev = params

    integral, _ = integrate.quad(gaussian, x.min(), x.max(), args=(amplitude, mean, stddev))

    return params, integral


def plot_gaussian_fit(
    data_x: t.List[float], data_y: t.List[float], gaussian_params: t.Tuple[float]
):
    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)

    amplitude, mean, stddev = gaussian_params

    plt.scatter(x, y, label="Data", color="blue", s=10)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = gaussian(x_fit, amplitude, mean, stddev)
    plt.plot(x_fit, y_fit, label="Gaussian Fit", color="red")
    plt.title("Gaussian Fit to Data")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()


# Si quieres probar tu código, descomenta las siguientes líneas y ejecuta el script
data_x = np.linspace(-5, 5, 100)
data_y = 3 * np.exp(-(data_x - 1) ** 2 / (2 * 1.5 ** 2)) + np.random.normal(0, 0.2, 100)
gaussian_params, integral = gaussian_fit_and_integration(data_x, data_y)
print("Integral de la curva gaussiana ajustada:", integral)
plot_gaussian_fit(data_x, data_y, gaussian_params)
