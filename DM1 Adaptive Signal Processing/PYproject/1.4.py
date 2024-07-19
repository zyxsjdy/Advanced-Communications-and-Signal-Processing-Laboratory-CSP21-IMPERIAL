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
        e[k] = d_in[k] - np.dot(w[k], x_k)  # calculate the error between response signal and desired response signal
        w[k+1] = np.add(w[k], 2 * m * x_k * e[k])  # update the weight

    return e, w[1:S+1]  # return N*L weights matrix


"____________________________________________________Main function____________________________________________________"


def main(size, L_f, L_f_un, N):
    mu = 1 / (3 * L_f)  # adaptation gain, input signal has a unit power
    x = np.random.randn(size)  # generate the input signal with unit power and zero mean

    # This is the theoretical optimized w, use it to generate desired response signal
    opt_w = np.zeros(L_f_un)
    for i in range(L_f_un):
        opt_w[i] = 1/(i+1) * math.exp(-((i-4)**2)/4)
    dn = np.add(np.convolve(x, opt_w, mode="full"), N)  # desired response signal

    # Calculate
    e, w = LMS(x, dn, mu, L_f)

    # Compute the learning rate
    lr = np.zeros(size)
    for i in range(size):
        lr[i] = compute_error(w[i], opt_w)

    return e, lr  # return N*L weights matrix


"____________________________________________________Plot Diagrams____________________________________________________"


if __name__ == "__main__":
    Loop = 300
    sample = 200  # sample number

    all_error = np.zeros([Loop, sample])
    all_learning_rate = np.zeros([Loop, sample])
    all_error_noise = np.zeros([Loop, sample])
    all_learning_rate_noise = np.zeros([Loop, sample])

    for q in range(Loop):
        L_filter = 9  # length of the adaptive filter
        L_filter_un = 9  # length of the unknown filter

        # generate noise that leads to a 20dB SNR, size of dn is size+L_f_un-1
        noise_0 = np.zeros(200 + L_filter_un - 1)
        noise_1 = 1 / 10 * np.random.randn(sample + L_filter_un - 1)

        all_error[q, :], all_learning_rate[q, :] = main(sample, L_filter, L_filter_un, noise_0)
        all_error_noise[q, :], all_learning_rate_noise[q, :] = main(sample, L_filter, L_filter_un, noise_1)

    mean_error = np.mean(all_error**2, 0)
    mean_learning_rate = np.mean(all_learning_rate, 0)
    mean_error_noise = np.mean(all_error_noise**2, 0)
    mean_learning_rate_noise = np.mean(all_learning_rate_noise, 0)

    plt.figure(1)
    plt.semilogy(mean_error, label="Without Noise")
    plt.semilogy(mean_error_noise, label="With Noise")
    plt.xlabel("Sample Number")
    plt.ylabel("Power of error")
    plt.title("Error between response signal and desired response signal")
    plt.legend()

    plt.figure(2)
    plt.semilogy(mean_learning_rate, label="Without Noise")
    plt.semilogy(mean_learning_rate_noise, label="With Noise")
    plt.xlabel("Sample Number")
    plt.ylabel("Error Rate")
    plt.title("Normalised weight error vector norm")
    plt.legend()

    plt.show()
