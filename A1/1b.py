import numpy as np


def separable(input_filter):
    """
    Write a function to verify the input filter is separable or not.
    """

    # apply SVD
    u, sigma, v = np.linalg.svd(input_filter)
    print(sigma)

    # Check if the 2nd diagonal element of the Sigma matrix is zero,
    # considering rounding errors, choose to round to 14 digits
    if round(sigma[1], 14) == 0:
        print("This is a separable filter")
        return True
    print("This is NOT a separable filter")
    return False


if __name__ == '__main__':
    filter_input = np.array([[0, 0.125, 0], [0.5, 0.5, 0.125], [0, 0.5, 0]])
    separable(filter_input)
