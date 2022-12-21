import os
import gzip
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import LinearSVC

test_folder = "__files"
n_neighbors_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_mnist(train_data=True, test_data=False):
    """
    Get mnist data from the official website and
    load them in binary format.

    Parameters
    ----------
    train_data : bool
        Loads
        'train-images-idx3-ubyte.gz'
        'train-labels-idx1-ubyte.gz'
    test_data : bool
        Loads
        't10k-images-idx3-ubyte.gz'
        't10k-labels-idx1-ubyte.gz'

    Return
    ------
    tuple
    tuple[0] are images (train & test)
    tuple[1] are labels (train & test)

    """
    RESOURCES = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz']

    if (os.path.isdir(f'{test_folder}') == 0):
        os.mkdir(f'{test_folder}')

    return get_images(train_data, test_data), get_labels(train_data, test_data)

def get_images(train_data=True, test_data=False):

    to_return = []

    if train_data:
        with gzip.open(f'{test_folder}/train-images-idx3-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            train_images = np.frombuffer(image_data, dtype=np.uint8)\
                .reshape((image_count, row_count, column_count))
            to_return.append(np.where(train_images > 127, 1, 0))

    if test_data:
        with gzip.open(f'{test_folder}/t10k-images-idx3-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of images
            image_count = int.from_bytes(f.read(4), 'big')
            # third 4 bytes is the row count
            row_count = int.from_bytes(f.read(4), 'big')
            # fourth 4 bytes is the column count
            column_count = int.from_bytes(f.read(4), 'big')
            # rest is the image pixel data, each pixel is stored as an unsigned byte
            # pixel values are 0 to 255
            image_data = f.read()
            test_images = np.frombuffer(image_data, dtype=np.uint8)\
                .reshape((image_count, row_count, column_count))
            to_return.append(np.where(test_images > 127, 1, 0))

    return to_return

def get_labels(train_data=True, test_data=False):

    to_return = []

    if train_data:
        with gzip.open(f'{test_folder}/train-labels-idx1-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            train_labels = np.frombuffer(label_data, dtype=np.uint8)
            to_return.append(train_labels)
    if test_data:
        with gzip.open('__files/t10k-labels-idx1-ubyte.gz', 'r') as f:
            # first 4 bytes is a magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            # second 4 bytes is the number of labels
            label_count = int.from_bytes(f.read(4), 'big')
            # rest is the label data, each label is stored as unsigned byte
            # label values are 0 to 9
            label_data = f.read()
            test_labels = np.frombuffer(label_data, dtype=np.uint8)
            to_return.append(test_labels)
    return to_return

def plot_image(X_test, y_test, y_pred):
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.2)
    for index, (image, test, pred) in enumerate(zip(X_test[0:10], y_test[0:10], y_pred[0:10])):
        ax = plt.subplot(2, 5, index + 1)
        ax.imshow(image.reshape(28, 28), cmap=plt.cm.gray)
        ax.set_title(f'Pred:{pred}, True:{test}', fontsize=10)
    plt.savefig("plot.pdf")
    plt.show()


def main():
    X_train, y_train = load_mnist(train_data=True, test_data=False)
    # print("X_train shape: ", X_train[0].shape)
    # print("y_train shape: ", y_train[0].shape)
    X_test, y_test = load_mnist(train_data=False, test_data=True)
    # print("X_test shape: ", X_test[0].shape)
    # print("y_test shape: ", y_test[0].shape)

    nsamples, nx, ny = X_train[0].shape
    X_train_reshape = X_train[0].reshape((nsamples, nx * ny))
    nsamples_, nx_, ny_ = X_test[0].shape
    X_test_reshape = X_test[0].reshape((nsamples_, nx_ * ny_))
    df = pd.DataFrame(columns=['y_train', 'y_test', 'y_pred'])
    linear_svm = LinearSVC()
    linear_svm.fit(X_train_reshape, y_train[0])
    y_pred = linear_svm.predict(X_test_reshape)
    # print("Accuracy: ", np.mean(y_pred == y_test[0]))
    linear_svm_score = linear_svm.score(X_test_reshape, y_test[0])
    print(f"Linear SVM score: {linear_svm_score}")
    df['y_train'] = y_train[0][0:10]
    df['y_test'] = y_test[0][0:10]
    df['y_pred'] = y_pred[0:10]

    # plot
    plot_image(X_test_reshape, df['y_test'], df['y_pred'])

    joblib.dump(linear_svm, "model.joblib")

if __name__ == "__main__":
    main()
