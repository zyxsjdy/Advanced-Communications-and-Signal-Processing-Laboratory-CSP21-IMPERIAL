# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import math
import matplotlib.pyplot as plt


"____________________________________________________Sub Functions____________________________________________________"


def compute_error(w_k, w_o):  # Normalised weight error vector norm
    numerator = np.sum((w_k - w_o) ** 2)
    denominator = np.sum(w_o ** 2)
    return numerator / denominator


# LMS algorithm
def LMS(x_in, d_in, m, L):  # x_in is the adaptive filter input, d_in is the desired response signal
    S = len(x_in)           # m is the adaptation gain, L is the length of the adaptive filter
    e = np.zeros(S)  # error array
    w = [[0] * L for _ in range(S+1)]  # weights, should be N*L matrix, the additional row is for the initial w
    zs = np.zeros(L-1)
    x_add0 = np.concatenate((zs, x_in))  # add L-1 zeros to the beginning of input, start with [0 0 0 0 0 0 0 0 x1]

    for k in range(S):
        x_k = x_add0[k-len(x_add0)-1+L:k-len(x_add0)-1:-1]  # take k to k+L-1 x_add0, and reverse their order
        e[k] = d_in[k] - x_k.dot(w[k])  # calculate the error between response signal and desired response signal
        w[k+1] = np.add(w[k], 2 * m * x_k * e[k])  # update the weight

    return e, w[1:S+1]  # return N*L weights matrix


"____________________________________________________Main function____________________________________________________"


def main(L_f):
    size = 200  # size of data array
    L_f_un = 9  # length of the unknown filter
    mu = 1 / (3 * L_f)  # adaptation gain, input signal has a unit power

    x = np.random.randn(size)  # generate the input signal with unit power and zero mean

    # This is the theoretical optimized w, use it to generate desired response signal
    opt_w = np.zeros(L_f_un)
    for i in range(L_f_un):
        opt_w[i] = 1/(i+1) * math.exp(-((i-4)**2)/4)
    dn = np.convolve(x, opt_w, mode="full")  # desired response signal

    # Calculate
    e, w = LMS(x, dn, mu, L_f)

    # Compute the learning rate
    w_e = np.zeros(size)
    opt_w_fit = np.zeros(L_f)  # If the length of unknown filter and adaptive filter are not the same, they cannot
    for i in range(L_f):       # be used in 'compute_error', hence fit the length of opt_w with adaptive filter
        opt_w_fit[i] = 1 / (i + 1) * math.exp(-((i - 4) ** 2) / 4)
    for i in range(size):
        w_e[i] = compute_error(w[i], opt_w_fit)

    return e, w_e  # return N*L weights matrix


"____________________________________________________Plot Diagrams____________________________________________________"


if __name__ == "__main__":
    L_filter = 9  # length of the adaptive filter
    error_equal, weight_error_equal = main(L_filter)

    L_filter = 5  # length of the adaptive filter
    error_low, weight_error_low = main(L_filter)

    L_filter = 11  # length of the adaptive filter
    error_high, weight_error_high = main(L_filter)

    plt.figure(1)
    plt.semilogy(error_equal**2, label="Filter Length = 9")
    plt.semilogy(error_low**2, label="Filter Length = 5")
    plt.semilogy(error_high**2, label="Filter Length = 11")
    plt.xlabel("Sample Number")
    plt.ylabel("Power of error")
    plt.title("Error between response signal and desired response signal")
    plt.legend()

    plt.figure(2)
    plt.semilogy(weight_error_equal, label="Filter Length = 9")
    plt.semilogy(weight_error_low, label="Filter Length = 5")
    plt.semilogy(weight_error_high, label="Filter Length = 11")
    plt.xlabel("Sample Number")
    plt.ylabel("Error Rate")
    plt.title("Normalised weight error vector norm")
    plt.legend()

    plt.show()