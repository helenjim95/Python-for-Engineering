import os
from tempfile import TemporaryDirectory
from joblib import dump, load
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import platform
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer
import joblib
from sklearn.metrics import accuracy_score

test_folder = "__files"
n_neighbors_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_mnist(folder=test_folder, train=True):
    file_list = os.listdir(folder)
    if train:
        if not platform.system() == 'Windows':
            X, y = loadlocal_mnist(
                images_path=f"{folder}/train-images-idx3-ubyte",
                labels_path=f"{folder}/train-labels-idx1-ubyte")

        else:
            X, y = loadlocal_mnist(
                images_path=f"{folder}/train-images.idx3-ubyte",
                labels_path=f"{folder}/train-labels.idx1-ubyte")
        return X, y
    if not train:
        if not platform.system() == 'Windows':
            X, y = loadlocal_mnist(
                images_path=f"{folder}/t10k-images-idx3-ubyte",
                labels_path=f"{folder}/t10k-labels-idx1-ubyte")

        else:
            X, y = loadlocal_mnist(
                images_path=f"{folder}/t10k-images.idx3-ubyte",
                labels_path=f"{folder}/t10k-labels.idx1-ubyte")
        return X, y


def image_show(number, data, label):
    fig, axes = plt.subplots(2, 2, figsize=(8, 4))
    for row in range(0, 2):
        for col in range(0, 2):
            x = data[row + col]  # get the vectorized image
            x = x.reshape((28, 28))  # reshape it into 28x28 format
            axes[row][col] = x
    print("saving numbers.pdf")
    plt.savefig("numbers.pdf")
    plt.show()
    # plt.imshow(x, cmap='gray')


# def plot_knn(model):
#     fig, axes = plt.subplots(1, 1, figsize=(8, 4))
#     axes.errorbar(
#         x=n_neighbors_list,
#         y=model.cv_results_["mean_test_score"],
#         yerr=model.cv_results_["std_test_score"],
#     )
#     axes.set(xlabel="n_neighbors", title="Classification accuracy")
#     # axes[1].errorbar(
#     #     x=n_neighbors_list,
#     #     y=grid_model.cv_results_["mean_fit_time"],
#     #     yerr=grid_model.cv_results_["std_fit_time"],
#     #     color="r",
#     # )
#     # axes[1].set(xlabel="n_neighbors", title="Fit time (with caching)")
#     # fig.tight_layout()
#     print("saving knn.pdf")
#     plt.savefig("knn.pdf")
#     plt.show()


def main():
    # print("start the main method")
    X_train, y_train = load_mnist(folder=test_folder, train=True)
    # print("training data/labels loaded")
    X_test, y_test = load_mnist(folder=test_folder, train=False)
    # print("test data/labels loaded")
    accuracy = 0
    knn = KNeighborsClassifier(n_neighbors=3)
    while accuracy < 0.8:
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        # print(accuracy)
    # print("accuracy > 80%, dumping model.sk")
    s = joblib.dump(knn, "model.sk")

    no_neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(no_neighbors))
    test_accuracy = np.empty(len(no_neighbors))

    for i, k in enumerate(no_neighbors):
        # print(f"n_neighbors={k}")
        # We instantiate the classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        # Fit the classifier to the training data
        knn.fit(X_train, y_train)
        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)
        # Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)
    # plot k values vs accuracy
    # print("plotting")
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(no_neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(no_neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Nearest Neighbors')
    plt.ylabel('Test Accuracy')
    # print("saving plot.pdf")
    plt.savefig("plot.pdf")
    plt.show()
    # image_show(4, X_train, y_train)
    # plot_knn(grid_model)
    # graph_model = KNeighborsTransformer(n_neighbors=max(n_neighbors_list), mode="distance")
    # classifier_model = KNeighborsClassifier(metric="precomputed")
    # with TemporaryDirectory(prefix="sklearn_graph_cache_") as tmpdir:
    #     full_model = Pipeline(
    #         steps=[("graph", graph_model), ("classifier", classifier_model)], memory=tmpdir
    #     )
    #     param_grid = {"classifier__n_neighbors": n_neighbors_list}
    #     grid_model = GridSearchCV(full_model, param_grid)
    #     print("grid_model done")
    #     grid_fit = grid_model.fit(X_train, y_train)
    #     print("grid_fit done")
    #     y_pred = grid_model.predict(X_test)
    #     print(y_pred)
    #     # accuracy = grid_model.score(X_test, y_train)
    #     accuracy = metrics.accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    main()
