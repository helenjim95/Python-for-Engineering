import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def breit_wigner(x, a, b, c):
    output = np.empty(len(x))
    for i in range(len(x)):
        output[i] = a / (np.power(b - x[i], 2) + c)
    return output


def func(pars, x, data=None):
    a, b, c = pars['a'], pars['b'], pars['c']
    model = a * np.exp(-b * x) + c
    if data is None:
        return model
    return model - data


def eval_errors(x_all: np.array, y_all, func, params: list):
    return y_all - func(x_all, *list(params))


def dfunc(pars, x, data=None):
    a, b = pars['a'], pars['b']
    v = np.exp(-b * x)
    return np.array([v, -a * x * v, np.ones(len(x))])


# TODO: evaluate the “jacobian” across all input points and
# calculate all necessary derivatives using central differences
def eval_jacobian(x, func, params, h=0.0001):
    global func_calls

    # number of data points
    m = len(breit_wigner(x, *params))
    # number of parameters
    n = len(params)

    # initialize Jacobian to Zero
    ps = params
    J = np.zeros((m, n))
    del_ = np.zeros((n, 1))

    # START --- loop over all parameters
    for j in range(n):
        # parameter perturbation
        del_[j, 0] = h * (1 + abs(params[j, 0]))
        # perturb parameter p(j)
        params[j, 0] = ps[j, 0] + del_[j, 0]

        if del_[j, 0] != 0:
            y1 = breit_wigner(x, *params)
            func_calls = func_calls + 1

            if h < 0:
                # backwards difference
                J[:, j] = (y1 - func) / del_[j, 0]
            else:
                # central difference, additional func call
                params[j, 0] = ps[j, 0] - del_[j]
                J[:, j] = (y1 - breit_wigner(x, *params)) / (2 * del_[j, 0])
                func_calls = func_calls + 1

        # restore p(j)
        params[j, 0] = ps[j, 0]

    return J


def lin2d(X, a, b):
    return X[:, 0] * a + X[:, 1] * b


# TODO with numpy:
def get_params():
    pass


# TODO:
def _lma_quality_measure(x_all, y_all, func, params, delta_params, jac, lma_lambda):
    pass


# TODO:
# def lma(X_all, y_all, func, param_guess, kwargs) -> list(float):
#     pass


def main():
    x = np.array([[0], [1]])
    params = [0.5, 0.2, 1]
    y = breit_wigner(x, *params)

    # TODO: for testing
    jacobian = eval_jacobian(np.array([[0], [1]]), breit_wigner, [0.5, 0.2, 1])

    # x_all = np.array([[0], [1]])
    # func = breit_wigner
    # params = [0.5, 0.2, 1]
    # param_change = 1
    # STEPSIZE = 0.00001
    # # TODO: What is yall?
    # while (param_change > STEPSIZE):
    #     jac = eval_jacobian(x_all, func, params)
    #     y_all = breit_wigner(x_all, *params)
    #     param_change = np.linalg.norm(delta_params) / np.linalg.norm(params)
    #     # calculate lambda if not set
    #     if lma_lambda is None:
    #         lma_lambda = np.linalg.norm(jac.T @ jac)
    #     e = eval_errors(x_all, y_all, func, params)
    #     delta_params = get_params()  # TODO: Implement that with numpy!
    #     # calculate the quality measure
    #     lma_rho = _lma_quality_measure(x_all, y_all, func, params, delta_params, jac, lma_lambda)
    #     if lma_rho > 0.75:
    #         lma_lambda /= 3
    #     elif lma_rho < 0.25:
    #         lma_lambda *= 2
    #     else:
    #         lma_lambda = lma_lambda
    #     # only change parameters if the quality measure is  greater than 0
    #     if lma_rho > 0:
    #         params = [x + d for x, d in zip(params, delta_params)]
    #
    # jac.to_csv("breit_wigner.csv")

    # plot from breit_wigner.csv
    data = pd.read_csv("breit_wigner.csv")
    x = data[["x"]].to_numpy()
    y = data["g"].to_numpy()
    # fit = # [a, b, c]
    parameter_list = lma(x, y, breit_wigner, np.random.rand(3) * 1)  # WILL NOT ALWAYS CONVERGE STILL!
    a, b, c = parameter_list
    x_range = (data['x'].min(), data['x'].max())
    x_eval = np.linspace(*x_range, 100)
    plt.scatter(x, y, zorder=1)
    plt.plot(x, y, zorder=2)
    plt.title(f"Breit-Wigner function for params a = {a}, b={b}, c={c}")
    plt.xlim(0, 200)
    plt.ylim(0, 90)
    plt.savefig('plot.pdf')
    plt.show()


if __name__ == "__main__":
    main()
