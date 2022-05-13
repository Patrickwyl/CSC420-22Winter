import numpy as np
from skimage import io
from matplotlib import pyplot as plt


def cross_correlation(image, filter_matrix):
    """
    Implement cross-correlation.
    """

    height, width = image.shape
    frame, col_pad, row_pad = zero_pad(image, filter_matrix)

    result = np.empty_like(frame)
    # traverse all pixels and calculate the correlation
    for i in range(col_pad, col_pad + height):
        for j in range(row_pad, row_pad + width):
            result[i, j] = correlation(frame, filter_matrix, i, j)

    return result[col_pad: col_pad + height, row_pad: row_pad + width]


def correlation(image, filter_matrix, i, j):
    """
    Implement correlation.
    """
    height, width = filter_matrix.shape

    m = int((height - 1) / 2)
    n = int((width - 1) / 2)

    img = image[i - m: i + m + 1, j - n: j + n + 1].flatten()
    filter_matrix = filter_matrix.flatten()

    return np.dot(img, filter_matrix)


def zero_pad(img, fil):
    """
    Helper function to apply zero padding
    """
    i, j = fil.shape

    pad_axis_1 = (i, i)
    pad_axis_2 = (j, j)

    padding = (pad_axis_1, pad_axis_2)
    frame = np.pad(img, padding, mode='constant', constant_values=0)

    return frame, i, j


if __name__ == '__main__':
    input_filter = np.array([[0, 0.125, 0], [0.5, 0.5, 0.125], [0, 0.5, 0]])
    input_img = io.imread('waldo.png', as_gray=True)
    output = cross_correlation(input_img, input_filter)

    plt.imshow(output, cmap='gray')
    plt.show()
