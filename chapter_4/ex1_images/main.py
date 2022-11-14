import os
import numpy as np
from PIL import Image, ImageFilter
from scipy.signal import convolve2d


def subsample(imagearray):
    # return the resulting array (np.ndarray)
    height, width, color_ = imagearray.shape
    cut = imagearray[:height:2, :width:2]
    outimg = Image.fromarray(cut)
    outimg.save("TUM_small.png")
    return cut


def boxblur(imagearray, size=5):
    # Apply a convolution, using scipy.signal.convolve2d.
    # set the mode-kw-argument to ’same’ so that the output has the same size as the image.
    # TODO: where to input kernel size?
    kernel1 = (1 / 9.0) * np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])
    # TODO: what input should be in convolve2d
    imagearray_blurred = imagearray  # need to delete
    # imagearray_blurred = convolve2d(imagearray, kernel1, mode='same')
    # How do you handle different channels? 
    # One of the simplest convolution kernels you could use would be a box blur. 
    # Choose a kernel of size 5 for outputting the TUM-image to TUM_blur.png.
    outimg = Image.fromarray(imagearray_blurred)
    outimg.save("TUM_blur.png")
    # return the resulting array (np.ndarray)
    # return imagearray_blurred


def frame(imagearray, frame=(15, 15, 15, 15), color=(48, 112, 179)) -> np.ndarray:
    # “frame” the image, output to TUM_frame.png with
    # a frame of 15 pixels in TUM-blue (RGB: (48, 112, 179)) at each “side”.
    height, width, color_ = imagearray.shape
    left, right, top, bottom = frame
    R, G, B = color
    imagearray[:top, :] = [R, G, B] #top
    imagearray[:, :left] = [R, G, B] #left
    imagearray[height - bottom:, :] = [R, G, B] #bottom
    imagearray[:, width - right:] = [R, G, B] #right
    print(imagearray)
    imagearray_framed = imagearray
    # Output to TUM_frame.png.
    outimg = Image.fromarray(imagearray_framed)
    outimg.save("TUM_frame.png")
    #     # # return the resulting array (np.ndarray) , where frame is the absolute number of pixels from the
    #     # respective edge being coloured!
    return imagearray_framed



if __name__ == "__main__":
    # process the image supplied in the zip as TUM_old.jpg
    # read the image
    global height, width, color
    file_list = os.listdir("__files")
    with Image.open("__files/TUM_old.jpg", "r") as imagearray:
        imagearray = np.array(imagearray)
        # print(imagearray.shape) # (124, 200, 3)
        imagearray_subsampled = subsample(imagearray)
        imagearray_blurred = boxblur(imagearray_subsampled, size=5)
        blue_color = (48, 112, 179)
        frame_size = 15
        # TODO: need to change input to imagearray_blurred once boxblur function is fixed
        image_framed = frame(imagearray_subsampled, frame=(frame_size, frame_size, frame_size, frame_size),
                                                color=blue_color)
    pass
