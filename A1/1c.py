import numpy as np
from scipy import ndimage
from skimage import io
from matplotlib import pyplot as plt
import time


def faster_convolution(img, filter_matrix):
    """
    Write a faster convolution function leveraging the fact that
    the filter is separable.
    """

    # apply SVD to the filter
    u, sigma, v = np.linalg.svd(filter_matrix)

    # The separated horizontal vertical filters
    horizontal = np.sqrt(sigma[0]) * np.asmatrix(u[:, 0])
    vert = np.sqrt(sigma[0]) * np.asmatrix(v[0])

    # convolve image by the horizontal filter
    horizontal_output = ndimage.convolve(img, horizontal)
    # convolve image by the vertical filter.
    result = ndimage.convolve(horizontal_output, vert.T)

    return result


if __name__ == '__main__':
    start_time = time.time()

    # choose a separable filter
    input_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    input_img = io.imread('waldo.png', as_gray=True)
    output = faster_convolution(input_img, input_filter)

    print("--- faster convolution takes %s seconds ---" %
          (time.time() - start_time))

    plt.imshow(output, cmap='gray')
    plt.show()
