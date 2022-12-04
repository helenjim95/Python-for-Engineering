import os
import struct
import loader
from numpy import array
import pathlib

test_folder = "C:/repos/Python for Engineering Data Analysis/ge32goy/chapter_7/ex3_MNIST_KNN/__files"


def load_mnist(folder=test_folder, train=True):
    file_list = os.listdir(folder)
    mnist = loader.MNIST()
    batch_number = 0
    batch_size = file_list.size()
    for path_img in file_list:
        filepath = f"{test_folder}/{path_img}"
        if file_list is not None:
            if type(file_list) is not list or len(file_list) is not 2:
                raise ValueError('batch should be a 1-D list'
                                 '(start_point, batch_size)')

        with open(folder, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        if file_list is not None:
            image_data = image_data[batch_number * rows * cols: \
                                    (batch_number + batch_size) * rows * cols]
            labels = labels[batch_number: batch_size + batch_size]
            size = batch_size

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

            # for some reason EMNIST is mirrored and rotated
            # if self.emnistRotate:
            #     x = image_data[i * rows * cols:(i + 1) * rows * cols]
            #
            #     subs = []
            #     for r in range(rows):
            #         subs.append(x[(rows - r) * cols - cols:(rows - r) * cols])
            #
            #     l = list(zip(*reversed(subs)))
            #     fixed = [item for sublist in l for item in sublist]
            #
            #     images[i][:] = fixed

        return images, labels


def main():
    images, labels = load_mnist(folder="__files", train=True)
    print(images)
    print(labels)
    print("it is done")


if __name__ == "__main__":
    main()
