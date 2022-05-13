import numpy as np
from skimage import io
from scipy import ndimage
from matplotlib import pyplot as plt


def magnitude_gradient(img):
    """
    Compute magnitude of gradients for the input image.
    """
    img_matrix = io.imread(img, as_gray=True)

    # choose the sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # use the sobel filter for the horizontal gradient
    horizontal = ndimage.convolve(img_matrix, sobel_x)
    # use the sobel filter for the vertical gradient
    vertical = ndimage.convolve(img_matrix, sobel_y)

    # get square root of the sum of squares of vertical and horizontal gradients
    result = np.sqrt(horizontal ** 2 + vertical ** 2)

    return result


def grid_localize(filter_img, img):
    """
    Write a function that localizes the input filter image in the input image
    based on the magnitude of gradients.
    """

    image = io.imread(img)  # colored image
    # using function from part 3a to compute the gradient magnitudes
    img_gradient = magnitude_gradient(img)
    filter_gradient = magnitude_gradient(filter_img)

    frame, col_pad, row_pad = zero_pad(img_gradient, filter_gradient)

    height, width = img_gradient.shape
    result = np.empty_like(frame)

    for i in range(col_pad, col_pad + height):
        for j in range(row_pad, row_pad + width):
            result[i, j] = normalized_correlation(frame, filter_gradient, i, j)

    similarity = result[col_pad: col_pad + height, row_pad: row_pad + width]
    max_index = similarity.argmax()

    i, j = np.unravel_index(max_index, similarity.shape)
    corners = np.array([[i, j],
                        [i + col_pad - 1, j],
                        [i + col_pad - 1, j + row_pad - 1],
                        [i, j + row_pad - 1],
                        [i, j]])

    # visualize the normalized cross-correlation
    plt.imshow(similarity)
    plt.show()

    # visualize the template matching
    plt.plot(corners[:, 1], corners[:, 0], 'b')
    plt.imshow(image)
    plt.show()


def normalized_correlation(img, filter_matrix, i, j):
    """
    helper function to conduct normalized correlation
    """
    height, width = filter_matrix.shape

    # flatten images
    img = img[i: i+height, j: j+width].flatten()
    filter_matrix = filter_matrix.flatten()

    dot = np.dot(img, filter_matrix)
    norm_img = np.linalg.norm(img)
    norm_filter = np.linalg.norm(filter_matrix)

    if (norm_img * norm_filter) == 0:
        result = 0
    else:
        result = dot/(norm_img * norm_filter)

    return result


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

    # 3a compute and visualize the magnitude of gradients
    plt.imshow(magnitude_gradient('waldo.png'), cmap='gray')
    plt.show()

    plt.imshow(magnitude_gradient('template.png'), cmap='gray')
    plt.show()

    # 3b localize the template in the image
    grid_localize('template.png', 'waldo.png')
