import numpy as np
from skimage import io
from scipy import ndimage
from matplotlib import pyplot as plt


def gaussian_filter(size, std):
    """
    Generate Gaussian filter with kernel size and std (sigma) as input.
    """

    # template
    gauss = np.zeros((size, size))
    # index of mean
    mu = (size - 1) / 2

    for row in range(size):
        for col in range(size):
            power = np.exp(-((row - mu) ** 2 + (col - mu) ** 2)
                           / (2 * (std ** 2)))
            gauss[row, col] = power / (2 * np.pi * (std ** 2))

    return gauss


if __name__ == '__main__':
    img_matrix = io.imread('waldo.png')

    # use a gaussian filter with kernel size=3 and std=1
    gauss_filter = gaussian_filter(3, 1)
    plt.imshow(gauss_filter)
    plt.show()

    # Convolve each RGB dimension
    img_matrix[:, :, 0] = ndimage.convolve(img_matrix[:, :, 0], gauss_filter)
    img_matrix[:, :, 1] = ndimage.convolve(img_matrix[:, :, 1], gauss_filter)
    img_matrix[:, :, 2] = ndimage.convolve(img_matrix[:, :, 2], gauss_filter)

    plt.imshow(img_matrix)
    plt.show()
