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

    Loop = 200
    all_error = np.zeros([Loop, Size])
    all_misad = np.zeros([Loop, Size])
    all_error_noise = np.zeros([Loop, Size])
    all_misad_noise = np.zeros([Loop, Size])

    for q in range(Loop):
        x = np.random.randn(Size)  # generate the input signal with unit power and zero mean
        noise = 1 / 10 * np.random.randn(Size + L_f_un - 1)

        # This is the theoretical optimized w, use it to generate desired response signal
        opt_w = np.zeros(L_f_un)
        for i in range(L_f_un):
            opt_w[i] = 1 / (i + 1) * math.exp(-((i - 4) ** 2) / 4)
        dn = np.convolve(x, opt_w, mode="full")  # desired response signal
        dn_noise = np.add(np.convolve(x, opt_w, mode="full"), noise)  # desired response signal

        all_error[q, :], w_RLS = RLS(x, dn, forget_factor, L_f)
        all_error_noise[q, :], w_RLS_noise = RLS(x, dn_noise, forget_factor, L_f)

        # Compute the misadjustment (normalised weight error vector norm)
        for i in range(Size):
            all_misad[q, i] = compute_sse(w_RLS[i], opt_w)
            all_misad_noise[q, i] = compute_sse(w_RLS_noise[i], opt_w)

    mean_error = np.mean(all_error**2, 0)
    mean_error_noise = np.mean(all_error_noise**2, 0)
    mean_misad = np.mean(all_misad, 0)
    mean_misad_noise = np.mean(all_misad_noise, 0)

    plt.figure(1)
    plt.semilogy(mean_error, label="Without noise")
    plt.semilogy(mean_error_noise, label="With noise")
    plt.xlabel("Sample Number")
    plt.ylabel("Power of error")
    plt.title("Error between response signal and desired response signal")
    plt.legend()

    plt.figure(2)
    plt.semilogy(mean_misad, label="Without noise")
    plt.semilogy(mean_misad_noise, label="With noise")
    plt.xlabel("Sample Number")
    plt.ylabel("Error Rate")
    plt.title("Normalised weight error vector norm")
    plt.legend()

    plt.show()


