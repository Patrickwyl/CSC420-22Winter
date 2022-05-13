import numpy as np
from skimage import io
from scipy import ndimage
from matplotlib import pyplot as plt


def canny_edge_detector(img):
    """
    Implement the Canny edge detector, performing non-maxima suppression.
    """
    # load the image as matrix
    img_matrix = io.imread(img, as_gray=True)

    # Apply Gaussian filter
    img_matrix = ndimage.gaussian_filter(img_matrix, sigma=1, order=0)

    # choose the sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    horizontal = ndimage.convolve(img_matrix, sobel_x)
    vertical = ndimage.convolve(img_matrix, sobel_y)

    # get square root of the sum of squares of vertical and horizontal gradients
    gradient = np.sqrt(horizontal ** 2 + vertical ** 2)

    # compute angles for each pixel
    angles = np.zeros(img_matrix.shape)

    for row in range(angles.shape[0]):
        for col in range(angles.shape[1]):
            current_angle = np.arctan2(vertical[row, col], horizontal[row, col])
            # convert to degrees
            current_angle = current_angle * 180 / np.pi
            if current_angle < 0:
                current_angle += 180
            angles[row, col] = current_angle

    # apply non-maximum suppression
    non_max = np.zeros(img_matrix.shape)
    for row in range(1, non_max.shape[0]-1):
        for col in range(1, non_max.shape[1]-1):
            # edge direction
            direction = 45 * round(angles[row, col] / 45)
            # save the neighbor edge strengths
            if direction == 0 or direction == 180:
                left = gradient[row, col - 1]
                right = gradient[row, col + 1]
            elif direction == 45:
                left = gradient[row + 1, col - 1]
                right = gradient[row - 1, col + 1]
            elif direction == 90:
                left = gradient[row - 1, col]
                right = gradient[row + 1, col]
            else:
                left = gradient[row - 1, col - 1]
                right = gradient[row + 1, col + 1]

            # compare current pixel's edge strengths with neighbors
            if left < gradient[row, col] and right < gradient[row, col]:
                non_max[row, col] = gradient[row, col]
            else:
                non_max[row, col] = 0

    return non_max


if __name__ == '__main__':
    canny_edge = canny_edge_detector('waldo.png')
    plt.imshow(canny_edge, cmap='gray')
    plt.show()
