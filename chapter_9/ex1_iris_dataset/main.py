import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def find_hyperparams(base_model, paramgrid, features, targets, cv=5, **kwargs):
    hypermodel = GridSearchCV(base_model, param_grid=paramgrid, cv=cv, scoring='neg_mean_squared_error', n_jobs=5, **kwargs)
    hypermodel.fit(features, targets)
    return hypermodel

def main():
    iris = load_iris()

    data = iris.data
    X = iris.data[:, :2]
    y = iris.target
    names = iris.target_names
    df = pd.DataFrame(data, columns=iris.feature_names)
    df['species'] = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


    knn = KNeighborsClassifier()
    linear_svc = SVC(kernel='linear')
    poly_svc = SVC(kernel='poly', degree=5)
    rbf_svc = SVC(kernel='rbf')

    svms = [linear_svc, poly_svc, rbf_svc]
    param_grid_knn = {'n_neighbors': np.arange(1, 25)}
    param_grid_svc = {'C': [0.1, 1, 10, 100]}

    evaluation = pd.DataFrame(columns=['model', 'c', 'accuracy_train', 'accuracy_test'])
    evaluation['model'] = svms

    hypermodel_knn = find_hyperparams(knn, param_grid_knn, X_train, y_train, cv=5)
    best_params_knn = hypermodel_knn.best_params_ # {'n_neighbors': 15}
    y_pred = hypermodel_knn.predict(X_test)
    accuracy_train_knn = accuracy_score(y_train[0:y_pred.shape[0]], y_pred)
    accuracy_test_knn = accuracy_score(y_test, y_pred)

    for svm in svms:
        hypermodel = find_hyperparams(svm, param_grid_svc, X_train, y_train, cv=5)
        best_params_ = hypermodel.best_params_
        # print(best_params_)
        y_pred = hypermodel.predict(X_test)

        evaluation.loc[evaluation['model'] == svm, 'c'] = best_params_['C']
        evaluation.loc[evaluation['model'] == svm, 'accuracy_train'] = accuracy_score(y_train[0:y_pred.shape[0]], y_pred)
        evaluation.loc[evaluation['model'] == svm, 'accuracy_test'] = accuracy_score(y_test, y_pred)

    knn_fit = KNeighborsClassifier(n_neighbors=best_params_knn['n_neighbors']).fit(X_train, y_train)
    lin_fit = SVC(kernel='linear', C=evaluation.loc[evaluation['model'] == linear_svc, 'c']).fit(X_train, y_train)
    rbf_fit = SVC(kernel='rbf', C=evaluation.loc[evaluation['model'] == rbf_svc, 'c']).fit(X_train, y_train)
    poly_fit = SVC(kernel='poly', degree=5, C=evaluation.loc[evaluation['model'] == poly_svc, 'c']).fit(X_train, y_train)

    # df_setosa = df[df['species'] == 0]
    # df_versicolor = df[df['species'] == 1]
    # df_virginica = df[df['species'] == 2]
    titles = [f'KNN, train {accuracy_train_knn:.2f}, test {accuracy_test_knn:.2f}\n {best_params_knn}',
              f'linear SVC, train {evaluation.loc[evaluation["model"] == linear_svc, "accuracy_train"].values[0]:.2f}, '
              f'test {evaluation.loc[evaluation["model"] == linear_svc, "accuracy_test"].values[0]:.2f}\n'
              f'C: {evaluation.loc[evaluation["model"] == linear_svc, "c" ].values[0]:.2f}, kernel: linear',
              f'poly SVC, train {evaluation.loc[evaluation["model"] == poly_svc, "accuracy_train"].values[0]:.2f}, '
              f'test {evaluation.loc[evaluation["model"] == poly_svc, "accuracy_test"].values[0]:.2f}\n'
              f'C: {evaluation.loc[evaluation["model"] == poly_svc, "c" ].values[0]:.2f}, '
              f'degree:5, kernel: poly',
              f'rbf SVC, train {evaluation.loc[evaluation["model"] == rbf_svc, "accuracy_train"].values[0]:.2f}, '
              f'test {evaluation.loc[evaluation["model"] == rbf_svc, "accuracy_test"].values[0]:.2f}\n'
              f'C: {evaluation.loc[evaluation["model"] == rbf_svc, "c" ].values[0]:.2f}, kernel: rbf',
              ]

    print("creating plots")
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    models = [knn_fit, lin_fit, poly_fit, rbf_fit]

    for clf, title, ax in zip(models, titles, sub.flatten()):
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel=iris.feature_names[0],
            ylabel=iris.feature_names[1],
        )
        scatter = ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_title(title, fontsize=8)
        handles, labels = scatter.legend_elements()
        ax.legend(handles=handles, labels=["setosa", "versicolor", 'virginica'], loc="upper right", fontsize=8)
    plt.savefig("plot.pdf")
    plt.show()


if __name__ == "__main__":
    main()
