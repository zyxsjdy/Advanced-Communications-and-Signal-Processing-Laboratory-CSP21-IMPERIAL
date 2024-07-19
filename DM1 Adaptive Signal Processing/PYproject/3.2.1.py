# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import math
import matplotlib.pyplot as plt

"____________________________________________________Sub Functions____________________________________________________"


def compute_error(x_in, d_in, w):
    return d_in - np.dot(w, x_in)


def compute_sse(w_k, w_o):  # Normalised weight error vector norm
    numerator = np.sum((w_k - w_o) ** 2)
    denominator = np.sum(w_o ** 2)
    return numerator / denominator


def compute_kalman_gain(x_in, last_rinv, ff):
    x_in = x_in[:, np.newaxis]
    a = np.matmul(last_rinv, x_in)
    return a / (ff + np.matmul(x_in.T, a))


def compute_rinv(x_in, last_rinv, kalman_gain, ff):
    x_in = x_in[:, np.newaxis]
    a = np.matmul(x_in.T, last_rinv)
    b = np.matmul(kalman_gain, a)
    return (1 / ff) * (last_rinv - b)


def RLS(x_in, d_in, lamb, L):
    S = len(x_in)
    e = np.zeros(S)  # error array
    w = [[0] * L for _ in range(S + 1)]  # weights, should be N*L matrix, the additional row is for the initial w
    zs = np.zeros(L - 1)
    x_add0 = np.concatenate((zs, x_in))
    rinv = np.eye(L)
    for k in range(S):
        x_k = x_add0[k - len(x_add0) - 1 + L:k - len(x_add0) - 1:-1]  # take k to k+L-1 x_add0, and reverse their order
        e[k] = compute_error(x_k, d_in[k], w[k])
        kalman_gain = compute_kalman_gain(x_k, rinv, lamb)
        w[k+1] = w[k] + kalman_gain.flatten() * e[k]
        rinv = compute_rinv(x_k, rinv, kalman_gain, lamb)
    return e, w[1:S + 1]


"____________________________________________________Main function____________________________________________________"


if __name__ == "__main__":
    L_f = 9
    L_f_un = 9
    forget_factor = 1
    Size = 200

    Loop = 1
    all_error_equal = np.zeros([Loop, Size])
    all_misad_equal = np.zeros([Loop, Size])
    all_error_low = np.zeros([Loop, Size])
    all_misad_low = np.zeros([Loop, Size])
    all_error_high = np.zeros([Loop, Size])
    all_misad_high = np.zeros([Loop, Size])

    for q in range(Loop):
        x = np.random.randn(Size)  # generate the input signal with unit power and zero mean

        # This is the theoretical optimized w, use it to generate desired response signal
        opt_w = np.zeros(L_f_un)
        for i in range(L_f_un):
            opt_w[i] = 1 / (i + 1) * math.exp(-((i - 4) ** 2) / 4)
        dn = np.convolve(x, opt_w, mode="full")  # desired response signal

        L_f_equal = 9  # length of the adaptive filter
        all_error_equal[q, :], w_RLS_equal = RLS(x, dn, forget_factor, L_f_equal)

        L_f_low = 5  # length of the adaptive filter
        all_error_low[q, :], w_RLS_low = RLS(x, dn, forget_factor, L_f_low)

        L_f_high = 11  # length of the adaptive filter
        all_error_high[q, :], w_RLS_high = RLS(x, dn, forget_factor, L_f_high)

        # Compute the misadjustment (normalised weight error vector norm)
        opt_w_low = np.zeros(L_f_low)
        for i in range(L_f_low):
            opt_w_low[i] = 1 / (i + 1) * math.exp(-((i - 4) ** 2) / 4)
        dn = np.convolve(x, opt_w_low, mode="full")  # desired response signal

        opt_w_high = np.zeros(L_f_high)
        for i in range(L_f_high):
            opt_w_high[i] = 1 / (i + 1) * math.exp(-((i - 4) ** 2) / 4)
        dn = np.convolve(x, opt_w_high, mode="full")  # desired response signal

        for i in range(Size):
            all_misad_equal[q, i] = compute_sse(w_RLS_equal[i], opt_w)
            all_misad_low[q, i] = compute_sse(w_RLS_low[i], opt_w_low)
            all_misad_high[q, i] = compute_sse(w_RLS_high[i], opt_w_high)

    mean_error_equal = np.mean(all_error_equal**2, 0)
    mean_error_low = np.mean(all_error_low**2, 0)
    mean_error_high = np.mean(all_error_high**2, 0)

    mean_misad_equal = np.mean(all_misad_equal, 0)
    mean_misad_low = np.mean(all_misad_low, 0)
    mean_misad_high = np.mean(all_misad_high, 0)

    plt.figure(1)
    plt.semilogy(mean_error_equal, label="L_f = 9")
    plt.semilogy(mean_error_low, label="L_f = 5")
    plt.semilogy(mean_error_high, label="L_f = 11")
    plt.xlabel("Sample Number")
    plt.ylabel("Power of error")
    plt.title("Error between response signal and desired response signal")
    plt.legend()

    plt.figure(2)
    plt.semilogy(mean_misad_equal, label="L_f = 9")
    plt.semilogy(mean_misad_low, label="L_f = 5")
    plt.semilogy(mean_misad_high, label="L_f = 11")
    plt.xlabel("Sample Number")
    plt.ylabel("Error Rate")
    plt.title("Normalised weight error vector norm")
    plt.legend()

    plt.show()


