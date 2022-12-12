import json
import os

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def main():
    dataframe = pd.read_csv("__files/wave.csv", header=0)
    data = dataframe.values
    X = data[:, 0]
    y = data[:, -1]
    n_sample = X.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8)

    test_r2 = 0
    test_mae = 0
    test_mse = 0
    train_mae = 0
    train_mse = 0
    train_r2 = 0
    # alphas = [1e0, 0.1, 1e-2, 1e-3]
    # gammas = np.logspace(-2, 2, 5)
    alphas = [0.001]
    gammas = [1.0]
    best_estimator_alpha = 0.001
    best_estimator_gamma = 1.0
    alpha = 0.001
    gamma = 10.0
    # for (alpha, gamma) in zip(alphas, gammas):
    hypermodel = GridSearchCV(KernelRidge(kernel="rbf", alpha=0.001, gamma=10.0, degree=5, coef0=1, kernel_params=None), param_grid=dict(alpha=alphas, gamma=gammas), cv=5, scoring='neg_mean_squared_error')
    hypermodel.fit(X_train.reshape(-1, 1), y_train)
    y_pred = hypermodel.predict(X_test.reshape(-1, 1))
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


    df_training = pd.DataFrame(columns=['X_train', 'y_train'])
    df_training['X_train'] = X_train
    df_training['y_train'] = y_train
    df_training.to_csv("train.csv", index=False)
    df_training = df_training.sort_values('X_train', axis=0, ascending=True)
    data_training = df_training.values

    df_test = pd.DataFrame(columns=['X_test', 'y_test'])
    df_test['X_test'] = X_test
    df_test['y_test'] = y_test
    df_test.to_csv("test.csv", index=False)
    df_test['y_pred'] = y_pred
    df_test = df_test.sort_values('X_test', axis=0, ascending=True)
    data_test = df_test.values

    s = joblib.dump(hypermodel, "model.joblib")

    file_list = os.listdir("__files")
    with open('scores.json', 'w') as output:
        data = {"test_mae": test_mae, "test_mse": test_mse, "test_r2": test_r2, "train_mae": train_mae, "train_mse": train_mse, "train_r2": train_r2}
        json.dump(data, output, sort_keys=False)

    fig, ax = plt.subplots()
    ax.scatter(data_training[:, 0], data_training[:, 1], color='blue', label='Training data')
    ax.scatter(data_test[:, 0], data_test[:, 1], color='red', label='Test data')
    ax.plot(data_test[:, 0], data_test[:, 1], label="f", color='red')
    ax.plot(data_test[:, 0], data_test[:, 2], label="predicted f", color='blue')
    ax.legend(loc="lower left")
    ax.set(title=f"MSE:{test_mse:.3f}, MAE:{test_mae:.3f}, R Square:{test_r2:.3f}", xlim=(-10.0, 10.0), ylim=(-2, 1))
    plt.savefig("plot.pdf")
    plt.show()


if __name__ == "__main__":
    main()
