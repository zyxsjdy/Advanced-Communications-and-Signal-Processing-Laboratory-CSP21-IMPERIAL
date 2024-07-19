# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io

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
        temp = sum(x_k**2)
        if temp != 0:
            x_normalized = x_k / temp
        else:
            x_normalized = x_k
        e[k] = d_in[k] - np.dot(w[k], x_k)  # calculate the error between desired response signal and response signal
        w[k + 1] = np.add(w[k], 2 * m * x_normalized * e[k])  # update the weight
    return e, w[1:S + 1]  # return N*L weights matrix


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
    mu = 1 / (0.5592 * L_f)  # adaptation gain, input signal has a unit power
    forget_factor = 1

    s1 = scipy.io.loadmat('s1.mat')
    s1 = s1['s1']
    x = np.array(s1).flatten()
    Size = len(x)

    # This is the theoretical optimized w, use it to generate desired response signal
    opt_w = np.zeros(L_f_un)
    for i in range(L_f_un):
        opt_w[i] = 1 / (i + 1) * math.exp(-((i - 4) ** 2) / 4)
    dn = np.convolve(x, opt_w, mode="full")  # desired response signal

    Loop = 1
    all_error_LMS = np.zeros([Loop, Size])
    all_error_NLMS = np.zeros([Loop, Size])
    all_error_RLS = np.zeros([Loop, Size])

    all_misad_LMS = np.zeros([Loop, Size])
    all_misad_NLMS = np.zeros([Loop, Size])
    all_misad_RLS = np.zeros([Loop, Size])

    for q in range(Loop):
        # Calculate
        all_error_LMS[q, :], w_LMS = LMS(x, dn, mu, L_f)
        all_error_NLMS[q, :], w_NLMS = NLMS(x, dn, mu, L_f)
        all_error_RLS[q, :], w_RLS = RLS(x, dn, forget_factor, L_f)

        # Compute the misadjustment (normalised weight error vector norm)
        for i in range(Size):
            all_misad_LMS[q, i] = compute_sse(w_LMS[i], opt_w)
            all_misad_NLMS[q, i] = compute_sse(w_NLMS[i], opt_w)
            all_misad_RLS[q, i] = compute_sse(w_RLS[i], opt_w)

    mean_error_LMS = np.mean(all_error_LMS**2, 0)
    mean_error_NLMS = np.mean(all_error_NLMS**2, 0)
    mean_error_RLS = np.mean(all_error_RLS**2, 0)

    mean_misad_LMS = np.mean(all_misad_LMS, 0)
    mean_misad_NLMS = np.mean(all_misad_NLMS, 0)
    mean_misad_RLS = np.mean(all_misad_RLS, 0)

    plt.figure(1)
    plt.semilogy(mean_error_LMS, label="LMS")
    plt.semilogy(mean_error_NLMS, label="NLMS")
    plt.semilogy(mean_error_RLS, label="RLS")
    plt.xlabel("Sample Number")
    plt.ylabel("Power of error")
    plt.title("Error between response signal and desired response signal")
    plt.legend()

    plt.figure(2)
    plt.semilogy(mean_misad_LMS, label="LMS")
    plt.semilogy(mean_misad_NLMS, label="NLMS")
    plt.semilogy(mean_misad_RLS, label="RLS")
    plt.xlabel("Sample Number")
    plt.ylabel("Error Rate")
    plt.title("Normalised weight error vector norm")
    plt.legend()

    plt.show()
