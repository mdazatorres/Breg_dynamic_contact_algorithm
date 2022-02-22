

from integrators import Breg, NAG, gradh, HTVI_adap, HTVI_Bet_d
import numpy as np
import matplotlib.pyplot as plt
import examples as exa
import scipy.integrate as si
import pylab

def tuning_process(method, seed, flag, n_times, steps = 200):
    '''
    In this function, we do the random parameter search for each method (CM, NAG, RGD, CRGD).
    '''
    min_f = 1e+16
    ex = exa.Example1(seed)
    mu_new = 0
    dt_new = 0
    alpha_new = 0
    delta_new = 0

    for i in range(n_times):
        # -------------    CM    -------------
        if flag == 1:
            dt = np.random.uniform(1e-2, 0.8)
            mu = np.random.uniform(0.8, 0.999)
            ite_x = method(ex, mu, dt, steps)
            alpha = 0
            delta = 0
        # -------------    NAG    -------------
        elif flag == 2:
            dt = np.random.uniform(1e-3, 0.5)
            mu = np.random.uniform(0.8, 0.999)
            ite_x = method(ex, mu, dt, steps)
            alpha = 0
            delta = 0
        # -------------    RGD    -------------
        elif flag == 3:
            dt = np.random.uniform(0, 0.6)
            #dt = np.random.uniform(1e-3, 0.5)
            mu = np.random.uniform(0.49, 0.95)
            delta = np.random.uniform(0, 20)  # delta
            alpha = np.random.uniform(0, 1.0001)  # alpha
            ite_x = method(ex, mu, dt, delta, alpha, steps) #(ex, mu, dt, delta, alpha, steps)
        # -------------    CRGD    -------------
        else:
            dt = np.random.uniform(0, 0.6)
            #dt = np.random.uniform(1e-3, 0.5)
            mu = np.random.uniform(0.49, 0.95)
            delta = np.random.uniform(0, 20)  # delta
            alpha = np.random.uniform(0, 1.0001)  # alpha
            ite_x = method(ex, [delta, alpha, mu], dt, steps)

        f_sim = np.apply_along_axis(ex.f, 1, ite_x)
        min_fnew = min(f_sim)
        print('minfnew', min_fnew, 'mu', mu, 'dt', dt, 'alpha', alpha, 'delta', delta, 'mu', mu)

        if min_fnew < min_f:
            min_f = min_fnew
            mu_new = mu
            dt_new = dt
            alpha_new = alpha
            delta_new = delta
    return mu_new, dt_new, alpha_new, delta_new


def find_params(n_times, num_trials, method, flag):
    '''
    In this function, we tuned the parameters for Figure 1. We run the function
    tuning_process() for each sample of A.
    '''
    Mu = []
    Dt = []
    Alpha = []
    Delta = []

    for i in range(num_trials):
        mu, dt, alpha, delta = tuning_process(method, i, flag, n_times)
        Mu.append(mu)
        Dt.append(dt)
        Alpha.append(alpha)
        Delta.append(delta)
    return Mu, Dt, Alpha, Delta


def params_Fig1(n_times, num_trials):
    '''
    In this function, we called the tuned parameters for Figure 3.
    '''
#    mu_CM, dt_CM, _, _ = find_params(n_times, num_trials, CM, 1)
#    mu_NAG, dt_NAG, _, _ = find_params(n_times, num_trials, NAG, 2)
    mu_RGD, dt_RGD, alpha_RGD, delta_RGD = find_params(n_times, num_trials, RGD, 3)
    mu_CRGD, dt_CRGD, alpha_CRGD, delta_CRGD = find_params(n_times, num_trials, CRGD, 4)

    # np.save("mu_CM_ex1", mu_CM)
    # np.save("dt_CM_ex1", dt_CM)
    # np.save("mu_NAG_ex1", mu_NAG)
    # np.save("dt_NAG_ex1", dt_NAG)

    np.save("mu_RGD_ex1_3", mu_RGD)
    np.save("dt_RGD_ex1_3", dt_RGD)
    np.save("alpha_RGD_ex1_3", alpha_RGD)
    np.save("delta_RGD_ex1_3", delta_RGD)

    np.save("mu_CRGD_ex1_3", mu_CRGD)
    np.save("dt_CRGD_ex1_3", dt_CRGD)
    np.save("alpha_CRGD_ex1_3", alpha_CRGD)
    np.save("delta_CRGD_ex1_3", delta_CRGD)


# -------- Tuning process for example 1 ---------
# To simulate  parameters for Example 1, uncommented the following line
params_Fig1(n_times=100, num_trials=50)
