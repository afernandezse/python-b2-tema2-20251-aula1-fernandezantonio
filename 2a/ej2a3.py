"""
Enunciado:
Desarrolla una función en Python llamada create_meshgrid, que utilice la función np.meshgrid de NumPy para generar una
cuadrícula de puntos. La función debe:

- Recibir tres parámetros: start, end, y step, que definirán los rangos de los arrays x e y.
- Crear dos arrays x e y utilizando los parámetros proporcionados.
- Utilizar np.meshgrid para generar una cuadrícula de puntos a partir de los arrays x e y.
- Modificar la fila 0 de la matriz X remplazar por el valor 99 utilizando indexación.

Parámetros:
    start (int): El valor inicial para los arrays x e y.
    end (int): El valor final para los arrays x e y.
    step (int): El paso entre cada valor en los arrays x e y.

Ejemplo:
    Entrada: create_and_modify_meshgrid(-5, 5, 1)
    Salida: Dos arrays modificados que representan una cuadrícula de puntos en el espacio 2D.
"""

import numpy as np
import typing as t


def create_and_modify_meshgrid(
    start: int, end: int, step: int
) -> t.Tuple[np.ndarray, np.ndarray]:
    x = np.arange(start, end +1, step)
    y = np.arange(start, end +1, step)
    X, Y = np.meshgrid(x, y)
    X[0, :] = 99
    return X, Y



# Para probar tu código, puedes usar los siguientes parámetros:
X, Y = create_and_modify_meshgrid(-5, 5, 1)
print(X)
