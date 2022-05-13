import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import time


def convolution(img, filter_matrix):
    """
    Write a function for computing convolution of the 2D (grayscale) image
    and a 2D filter.
    """

    # flip the filter vertically and horizontally
    filter_matrix = filter_matrix[::-1, ::-1]
    filter_shape = filter_matrix.shape[0]
    padding = filter_shape - 1

    # zero padding
    padded = np.zeros((img.shape[0] + padding, img.shape[1] + padding))

    # copy in the image matrix between the padding in the matrix
    for row in range(padding//2, padded.shape[0] - (padding//2)):
        for col in range(padding//2, padded.shape[1] - (padding//2)):
            padded[row, col] = img[row - (padding//2), col - (padding//2)]

    # make the output matrix be the same size as the input image
    result = np.zeros(img.shape)

    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            img_section = padded[row: (row + filter_shape),
                                 col: (col + filter_shape)]
            # convolution calculation
            result[row, col] = img_section.flatten().dot(filter_matrix.flatten())

    return result


if __name__ == '__main__':
    start_time = time.time()

    input_filter = np.array([[0, 0.125, 0], [0.5, 0.5, 0.125], [0, 0.5, 0]])
    input_img = io.imread('waldo.png', as_gray=True)
    output = convolution(input_img, input_filter)

    print("--- convolution takes %s seconds ---" % (time.time() - start_time))

    plt.imshow(output, cmap='gray')
    plt.show()
