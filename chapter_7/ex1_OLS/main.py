import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess


def ols_fit(x, y, transformations: list[callable]) -> list[float]:
    all_transformed = []
    for xelement in x:
        all_transformed.append([tf(xelement) for tf in transformations])
    reg = LinearRegression().fit(all_transformed, y)
    # return parameter [a, b, c]
    return reg.coef_


def smoothing(x, y):
    lowess_frac = 0.15  # size of data (%) for estimation =~ smoothing window
    lowess_it = 0
    x_smooth = x
    y_smooth = lowess(y, x, is_sorted=False, frac=lowess_frac, it=lowess_it, return_sorted=False)
    return x_smooth, y_smooth

def compute_y_prediction(x, a, b, c):
    output = np.empty(len(x))
    for i in range(len(x)):
        output[i] = a * math.pow(x[i], 2) + b * math.pow(x[i], 5) + c * math.sin(x[i])
    return output

def main():
    data = pd.read_csv("__files/data.csv", header=0, sep=",").sort_values(by=["x"], ascending=True)
    transformations = [lambda x: x**2, lambda x: x ** 5, math.sin]
    parameter_list = ols_fit(data.x, data.y, transformations)
    # TODO: need to change the digit format in txt
    with open('coeffs.txt', 'w') as f:
        for number in (range(0, parameter_list.size - 1)):
            f.write(f"{float(number)} ")
        f.write(f"{float(parameter_list[parameter_list.size - 1])}")
    a, b, c = parameter_list
    x = data['x']
    y = data['y']
    x_range = (data['x'].min(), data['x'].max())
    x_eval = np.linspace(*x_range, 100)
    y_eval = compute_y_prediction(x_eval, a, b, c)
    plt.scatter(x, y, label='data', zorder=1)
    plt.plot(x_eval, y_eval, label='OLS fit', color="green", linestyle="-", zorder=2)
    plt.title(f"OLS fit, a = {a}, b={b}, c={c}")
    plt.xlim(-10.0, 10.0)
    plt.ylim(-5.0, 50.0)
    plt.legend(loc="upper left")
    plt.savefig('plot.pdf')
    plt.show()


if __name__ == "__main__":
    main()