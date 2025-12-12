"""
Enunciado:
Desarrolla la función enhanced_compare_monthly_sales para visualizar y analizar datos de ventas de tres años distintos
utilizando la biblioteca Matplotlib. Esta función debe crear gráficos para comparar las ventas mensuales de dos años y
mostrar la distribución de las ventas de un tercer año.

Detalles de la Implementación:

            Eje X: Nombres de los meses.
            Eje Y izquierdo: Ventas mensuales.
            Eje Y derecho: Ventas acumuladas.
        Leyendas para diferenciar cada año y las ventas acumuladas.

    Gráfico de Pastel:
        Título: "Distribución de Ventas Mensuales para 2022"
        Etiquetas para cada segmento del pastel, mostrando el porcentaje de ventas por mes.

Ejemplo:
    Entrada:
        sales_2020 = [120, 150, 180, 200, ...] # Ventas para cada mes en 2020
        sales_2021 = [130, 160, 170, 210, ...] # Ventas para cada mes en 2021
        sales_2022 = [140, 155, 175, 190, ...] # Ventas para cada mes en 2022
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Ejecución:
        enhanced_compare_monthly_sales(sales_2020, sales_2021, sales_2022, months)
    Salida:
        Dos gráficos dentro de la misma figura, uno combinando barras y líneas para 2020 y 2021, y otro en forma de
        pastel para 2022.
"""

import matplotlib.pyplot as plt
import numpy as np
import typing as t


def compare_monthly_sales(
    sales_year1: list,
    sales_year2: list,
    sales_year3: list,
    months: list
) -> t.Tuple[plt.Figure, plt.Axes, plt.Axes]:
    

    y1 = np.asarray(sales_year1)
    y2 = np.asarray(sales_year2)
    y3 = np.asarray(sales_year3)

    if len(months) != 12:
        raise ValueError("months must have exactly 12 entries.")

    csum1 = np.cumsum(y1)
    csum2 = np.cumsum(y2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    x = np.arange(12)
    width = 0.42
    b1 = ax1.bar(x - width/2, y1, width=width, label="2020")
    b2 = ax1.bar(x + width/2, y2, width=width, label="2021")

    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    ax1.set_ylabel("Monthly Sales")

    ax1_t = ax1.twinx()
    l1, = ax1_t.plot(x, csum1, marker="o", linestyle="-", label="2020 Cumulative")
    l2, = ax1_t.plot(x, csum2, marker="o", linestyle="-", label="2021 Cumulative")
    ax1_t.set_ylabel("Cumulative Sales")

    ax1.set_title("Monthly Sales Comparison: 2020 vs 2021")

    handles = [b1, b2, l1, l2]
    labels = ["2020", "2021", "2020 Cumulative", "2021 Cumulative"]
    ax1.legend(handles, labels, loc="upper left")

    total3 = float(y3.sum())
    if total3 == 0:
        ax2.text(0.5, 0.5, "No Sales in 2022", ha="center", va="center", fontsize=11)
        ax2.axis("off")
    else:
        ax2.pie(y3, labels=months, autopct=lambda p: f"{p:.1f}%", startangle=90)
        ax2.axis("equal")

    # TÍTULO EXACTO que pide el test
    ax2.set_title("2022 Monthly Sales Distribution")

    return fig, ax1, ax2



# Para probar el código, descomenta las siguientes líneas
sales_2020 = np.random.randint(100, 500, 12)
sales_2021 = np.random.randint(100, 500, 12)
sales_2022 = np.random.randint(100, 500, 12)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


if __name__ == "__main__":
    fig, ax1, ax2 = compare_monthly_sales(sales_2020, sales_2021, sales_2022, months)
    plt.show()
