# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io

"____________________________________________________Sub Functions____________________________________________________"


def compute_error(w_k, w_o):  # Normalised weight error vector norm
    numerator = np.sum((w_k - w_o) ** 2)
    denominator = np.sum(w_o ** 2)
    return numerator / denominator


# LMS algorithm
def LMS(x_in, d_in, m, L):  # x_in is the adaptive filter input, d_in is the desired response signal
    S = len(x_in)  # m is the adaptation gain, L is the length of the adaptive filter
    e = np.zeros(S)  # error array
    w = [[0] * L for _ in range(S + 1)]  # weights, should be N*L matrix, the additional row is for the initial w
    zs = np.zeros(L - 1)
    x_add0 = np.concatenate((zs, x_in))  # add L-1 zeros to the beginning of input, start with [0 0 0 0 0 0 0 0 x1]

    for k in range(S):
        x_k = x_add0[k - len(x_add0) - 1 + L:k - len(x_add0) - 1:-1]  # take k to k+L-1 x_add0, and reverse their order
        e[k] = d_in[k] - np.dot(w[k], x_k)  # calculate the error between response signal and desired response signal
        w[k + 1] = np.add(w[k], 2 * m * x_k * e[k])  # update the weight

    return e, w[1:S + 1]  # return N*L weights matrix


def NLMS(x_in, d_in, m, L):  # x_in is the adaptive filter input, d_in is the desired response signal
    S = len(x_in)  # m is the adaptation gain, L is the length of the adaptive filter
    e = np.zeros(S)  # error array
    w = [[0] * L for _ in range(S + 1)]  # weights, should be N*L matrix, the additional row is for the initial w
    zs = np.zeros(L - 1)
    x_add0 = np.concatenate((zs, x_in))  # add L-1 zeros to the beginning of input, start with [0 0 0 0 0 0 0 0 x1]

    for k in range(S):
        x_k = x_add0[k - len(x_add0) - 1 + L:k - len(x_add0) - 1:-1]  # take k to k+L-1 x_add0, and reverse their order

        temp = np.linalg.norm(x_k)
        if temp != 0:
            x_normalized = x_k / temp ** 2
        else:
            x_normalized = x_k

        e[k] = d_in[k] - np.dot(w[k], x_k)  # calculate the error between desired response signal and response signal
        w[k + 1] = np.add(w[k], 2 * m * x_normalized * e[k])  # update the weight

    return e, w[1:S + 1]  # return N*L weights matrix


"____________________________________________________Main function____________________________________________________"


def main(x):
    size = len(x)
    L_f = 9
    L_f_un = 9

    P_x = np.sum(x ** 2) / size

    mu = 1 / (0.5592 * L_f)  # adaptation gain, input signal has a unit power

    # This is the theoretical optimized w, use it to generate desired response signal
    opt_w = np.zeros(L_f_un)
    for i in range(L_f_un):
        opt_w[i] = 1 / (i + 1) * math.exp(-((i - 4) ** 2) / 4)
    dn = np.convolve(x, opt_w, mode="full")  # desired response signal

    # Calculate
    e_LMS, w_LMS = LMS(x, dn, mu, L_f)
    e_NLMS, w_NLMS = NLMS(x, dn, mu, L_f)

    # Compute the normalised weight error vector norm
    lr_LMS = np.zeros(size)
    lr_NLMS = np.zeros(size)
    for i in range(size):
        lr_LMS[i] = compute_error(w_LMS[i], opt_w)
        lr_NLMS[i] = compute_error(w_NLMS[i], opt_w)

    return e_LMS, lr_LMS, e_NLMS, lr_NLMS  # return N*L weights matrix


"____________________________________________________Plot Diagrams____________________________________________________"

if __name__ == "__main__":
    s1 = scipy.io.loadmat('s1.mat')
    s1 = s1['s1']
    s1 = np.array(s1).flatten()

    all_error_LMS, all_learning_rate_LMS, all_error_NLMS, all_learning_rate_NLMS = main(s1)

    plt.figure(1)
    plt.semilogy(all_error_LMS**2, label="LMS")
    plt.semilogy(all_error_NLMS**2, label="NLMS")
    plt.xlabel("Sample Number")
    plt.ylabel("Power of error")
    plt.title("Error between response signal and desired response signal")
    plt.legend()

    plt.figure(2)
    plt.semilogy(all_learning_rate_LMS, label="LMS")
    plt.semilogy(all_learning_rate_NLMS, label="NLMS")
    plt.xlabel("Sample Number")
    plt.ylabel("Error Rate")
    plt.title("Normalised weight error vector norm")
    plt.legend()

    plt.show()
