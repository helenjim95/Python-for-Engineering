import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge


# TODO: how to update y_pred for each iteration of the loop?
y_pred = 1
model = 0
test_r2 = 0
test_mae = 0
test_mse = 0
train_mae = 0
train_mse = 0
train_r2 = 0


def main():
    dataframe = pd.read_csv("__files/nitride_compounds.csv", header=0, index_col=0)
    # print(dataframe.head())
    data = dataframe.values
    X = data[:, 1:27]
    y = data[:, 28] #HSE band gap
    n_sample = X.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8)

    global y_pred, model, test_r2, test_mae, test_mse, train_mae, train_mse, train_r2

    alphas = [1e0, 0.1, 1e-2, 1e-3]
    gammas = np.logspace(-2, 2, 5)
    while test_r2 < 0.7:
        for (alpha, gamma) in zip(alphas, gammas):
            kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma, degree=3, coef0=1, kernel_params=None)
            hypermodel = GridSearchCV(kr, param_grid=dict(alpha=alphas, gamma=gammas), cv=5, scoring='neg_mean_squared_error')
            hypermodel.fit(X_train, y_train)
            y_pred = hypermodel.predict(X_test)
            best_params = hypermodel.best_params_
            score = r2_score(y_test, y_pred)
            train_r2 = r2_score(y_train[0:y_pred.shape[0]], y_pred)
            # print("best_estimator_.alpha", hypermodel.best_estimator_.alpha)
            # print("best_estimator_.gamma", hypermodel.best_estimator_.gamma)
            # print(f"alpha: {alpha}, gamma: {gamma}, R2: {score}")
            test_mae = mean_absolute_error(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            train_mae = mean_absolute_error(y_train[0:y_pred.shape[0]], y_pred)
            train_mse = mean_squared_error(y_train[0:y_pred.shape[0]], y_pred)
            if score > test_r2:
                test_r2 = score
                model = kr

    s = joblib.dump(hypermodel, "model.joblib")

    train_sizes_abs_mse, train_scores_mse, test_scores_mse = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_mean_squared_error",
        cv=5,
    )

    train_sizes_abs_r2, train_scores_r2, test_scores_r2 = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="r2",
        cv=5,
    )

    df_mse = pd.DataFrame(columns=['train_sizes_abs_mse', '-test_scores_mse.mean'])
    df_mse['train_sizes_abs_mse'] = train_sizes_abs_mse
    df_mse['-test_scores_mse.mean'] = -test_scores_mse.mean(axis=1)
    df_mse = df_mse.sort_values('train_sizes_abs_mse', axis=0, ascending=True)
    data_mse = df_mse.values

    df_r2 = pd.DataFrame(columns=['train_sizes_abs_r2', 'test_scores_r2.mean'])
    df_r2['train_sizes_abs_r2'] = train_sizes_abs_r2
    df_r2['test_scores_r2.mean'] = test_scores_r2.mean(axis=1)
    df_r2 = df_r2.sort_values('train_sizes_abs_r2', axis=0, ascending=True)
    data_mse = df_r2.values

    # TODO: second plot put into dataframe and sort
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    ax1_2 = ax1.twinx()
    ax1.plot(np.linspace(0.1, 1.0, 10), df_mse['-test_scores_mse.mean'], color="b", label="Test MSE score")
    ax1_2.plot(np.linspace(0.1, 1.0, 10), df_r2['test_scores_r2.mean'], color="r", label="r2 score")
    ax1.set(title="Learning curves", xlabel="fraction of training data used", ylabel='MSE')
    ax1_2.set(ylabel='R Square')
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    ax2.plot([p1, p2], [p1, p2], 'b-')
    training_dots = ax2.scatter(y_train[0:y_pred.shape[0]], y_pred, color="b", label='training data')
    test_dots = ax2.scatter(y_test, y_pred,  color="r", label='test data')
    ax2.legend()
    ax2.set(title=f"Model R square:{test_r2:.3f}, MAE:{test_mae:.3f}", xlabel="Calculated gap", ylabel='Model gap')

    plt.savefig("plot.pdf")
    plt.show()


if __name__ == "__main__":
    main()
