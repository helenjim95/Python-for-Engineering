import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge


# TODO: how to update y_pred for each iteration of the loop?
y_pred = 0
def main():
    dataframe = pd.read_csv("__files/nitride_compounds.csv", header=0, index_col=0)
    # print(dataframe.head())
    data = dataframe.values
    y = data[:, 28] #HSE band gap
    X = data[:, 1:27]
    n_sample = X.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8)
    global y_pred
    test_r2 = 0.7
    test_mae = 0
    test_mse = 0
    train_mae = 0
    train_mse = 0
    train_r2 = 0
    alphas = [1e0, 0.1, 1e-2, 1e-3]
    gammas = np.logspace(-2, 2, 5)
    model = 0
    hypermodel = 0
    while test_r2 < 0.7:
        for (alpha, gamma) in zip(alphas, gammas):
            kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma, degree=5, coef0=1, kernel_params=None)
            hypermodel = GridSearchCV(kr, param_grid=dict(alpha=alphas, gamma=gammas), cv=5, scoring='neg_mean_squared_error')
            hypermodel.fit(X_train.reshape(-1, 1), y_train)
            y_pred = hypermodel.predict(X_test.reshape(-1, 1))
            print("y_pred", y_pred)
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
    # training_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # for (size) in training_size:

    # TODO: something is wrong with X, need to create a dataframe for R square for plotting
    # train_sizes_abs, train_scores_kr, test_scores_kr = learning_curve(
    #     model,
    #     X_train.reshape(-1, 1),
    #     y_train,
    #     train_sizes=np.linspace(0.1, 1, 10),
    #     scoring="neg_mean_squared_error",
    #     cv=10,
    # )

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    # ax1.plot(train_sizes_abs, -train_scores_kr.mean(axis=1), "o-", color="r", label="Training MSE score")
    ax1.set(title="Learning curves", xlabel="fraction of training data used", ylabel='MSE')
    ax2.plot(y_test, y_pred, label='gap')
    ax2.scatter(y_train, y_pred, label='training data', color="b")
    ax2.scatter(y_test, y_pred, label='training data', color="r")
    ax2.set(title=f"Model R square:{test_r2}, MAE:{test_mae}", xlabel="Calculated gap", ylabel='Model gap')
    ax2.legend("upper left")
    plt.savefig("plot.pdf")
    plt.show()

# finding:
# y = band gap (PBE Eg (eV),HSE Eg (eV)
# After comparison among different machine learning techniques,
# when elemental properties are taken as features,
# support vector regression (SVR) with radial kernel performs best for
# predicting both the band gap and band offset with a prediction root mean square error (RMSE)
# of 0.298 eV and 0.183 eV, respectively.

if __name__ == "__main__":
    main()
