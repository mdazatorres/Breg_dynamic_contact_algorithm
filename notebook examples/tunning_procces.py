

from integrators import Breg, NAG, CM, HTVI_adap, Bet_dir
import numpy as np
import examples as exa
import matplotlib.pyplot as plt
import pylab
import examples as exa
import scipy.integrate as si
from integrators import HBr

def tuning_process_random(method,n_times, steps = 200):
    n=2
    seed=0
    ex = exa.Example1(seed, n)
    np.random.seed(None)
    min_f = 1e+16
    mu_new = 0
    dt_new = 0
    c_new = 0
    C_new = 0
    m = 0.01
    v = 1000

    for i in range(n_times):
        dt = np.random.uniform(1e-5, 0.8)
        mu = np.random.uniform(0.8, 0.999)
        c = np.random.uniform(1, 4)
        C = np.random.uniform(1, 6)

        if method == 'RB':
            solX, solP = Breg(ex, [v, m, c, C], dt, steps, [ex.x0, ex.p0], adap=False)
        elif method == 'RB':
            solX, solP = Breg(ex, [v, m, c, C], dt, steps, [ex.x0, ex.p0], adap=True)
        elif method == 'HTVI_adap':
            p0_t = -HBr(ex, ex.x0, ex.p0, ex.x0_t, c)  # np.zeros(q0.shape)
            solX = HTVI_adap(ex, c, dt, steps, [ex.x0, ex.x0_t, ex.p0, p0_t])
        elif method == 'Betancourt':
            p0_t = -HBr(ex, ex.x0, ex.p0, ex.x0_t, c)
            solX = Bet_dir(ex, c, dt, steps, [ex.x0, ex.p0, p0_t])
        elif method == 'CM':
            solX = CM(ex, mu, dt, steps)
        else:
            solX = NAG(ex, mu, dt, steps)

        f_sim = np.apply_along_axis(ex.f, 1, solX)
        min_fnew = min(f_sim)
        #print('minfnew', min_fnew, 'mu', mu, 'dt', dt, 'c', c, 'C', c)

        if min_fnew < min_f:
            min_f = min_fnew
            mu_new = mu
            dt_new = dt
            c_new =c
            C_new = C

    return mu_new, dt_new, C_new, c_new

def tuning_process(method, ex, steps = 200):
    '''
    In this function, we do the random parameter search for each method (CM, NAG, RGD, CRGD).
    '''
    min_f = 1e+16
    mu_new = 0
    dt_new = 0
    c_new = 0
    C_new = 0
    dt= np.linspace(1e-5, 0.5,12)
    mu= np.linspace(0.8, 0.99,5)
    c= np.linspace(1,6,7)
    C=np.linspace(1,4,5)
    m = 0.01
    v = 1000

    if method == 'RB' or method == 'RB_adap':
        for i in range(len(c)):
            for j in range(len(C)):
                for k in range(len(dt)):
                    if method == 'RB':
                        solX, solP = Breg(ex, [v, m, c[i], C[j]], dt[k], steps, [ex.x0, ex.p0], adap=False)
                    else:
                        solX, solP = Breg(ex, [v, m, c[i], C[j]], dt[k], steps, [ex.x0, ex.p0], adap=True)
                    f_sim = np.apply_along_axis(ex.f, 1, solX)
                    min_fnew = min(f_sim)
                    if min_fnew < min_f:
                        min_f = min_fnew
                        c_new = c[i]
                        dt_new = dt[k]
                        C_new =  C[j]

    elif method == 'HTVI_adap':
        for i in range(len(c)):
            p0_t = -HBr(ex, ex.x0, ex.p0, ex.x0_t, c[i])
            init = [ex.x0, ex.x0_t, ex.p0, p0_t]
            for k in range(len(dt)):
                solX = HTVI_adap(ex, c[i], dt[k], steps, init)
                f_sim = np.apply_along_axis(ex.f, 1, solX)
                min_fnew = min(f_sim)
                if min_fnew < min_f:
                    min_f = min_fnew
                    c_new = c[i]
                    dt_new = dt[k]

    elif method == 'Betancourt':
            for i in range(len(c)):
                p0_t = -HBr(ex, ex.x0, ex.p0, ex.x0_t, c[i])  # np.zeros(q0.shape)
                for k in range(len(dt)):
                    solX = Bet_dir(ex, c[i], dt[k], steps, [ex.x0, ex.p0, p0_t])
                    f_sim = np.apply_along_axis(ex.f, 1, solX)
                    min_fnew = min(f_sim)
                    if min_fnew < min_f:
                        min_f = min_fnew
                        c_new = c[i]
                        dt_new = dt[k]

    if method == 'CM' or 'NAG':
        for i in range(len(mu)):
                for k in range(len(dt)):
                    if method =='NAG':
                        solX = NAG(ex, mu[i], dt[k], steps)
                    else:
                        solX = CM(ex,  mu[i], dt[k], steps)
                    f_sim = np.apply_along_axis(ex.f, 1, solX)
                    min_fnew = min(f_sim)
                    if min_fnew < min_f:
                        min_f = min_fnew
                        c_new = c[i]
                        dt_new = dt[k]

    return mu_new, dt_new, c_new, C_new

ex = exa.Example1(seed=0, n=10)
#MU, DT, c_, C_ =tuning_process(method='RB', ex=ex, steps = 200)

aa=tuning_process_random(method='RB',n_times=1000, steps = 200)
#aa=tuning_process(method='RB', steps = 200)

# def find_params(n_times, num_trials, method, flag):
#     '''
#     In this function, we tuned the parameters for Figure 1. We run the function
#     tuning_process() for each sample of A.
#     '''
#     Mu = []
#     Dt = []
#     Alpha = []
#     Delta = []
#
#     for i in range(num_trials):
#         mu, dt, alpha, delta = tuning_process(method, i, flag, n_times)
#         Mu.append(mu)
#         Dt.append(dt)
#         Alpha.append(alpha)
#         Delta.append(delta)
#     return Mu, Dt, Alpha, Delta
#minfnew 0.008937552308949088 mu 0.8428895814162685 dt 0.15780650319624523 c 3.7262301365820765 C 3.7262301365820765


# def params_Fig1(n_times, num_trials):
#     '''
#     In this function, we called the tuned parameters for Figure 3.
#     '''
# #    mu_CM, dt_CM, _, _ = find_params(n_times, num_trials, CM, 1)
# #    mu_NAG, dt_NAG, _, _ = find_params(n_times, num_trials, NAG, 2)
#     mu_RGD, dt_RGD, alpha_RGD, delta_RGD = find_params(n_times, num_trials, RGD, 3)
#     mu_CRGD, dt_CRGD, alpha_CRGD, delta_CRGD = find_params(n_times, num_trials, CRGD, 4)
#
#     # np.save("mu_CM_ex1", mu_CM)
#     # np.save("dt_CM_ex1", dt_CM)
#     # np.save("mu_NAG_ex1", mu_NAG)
#     # np.save("dt_NAG_ex1", dt_NAG)
#
#     np.save("mu_RGD_ex1_3", mu_RGD)
#     np.save("dt_RGD_ex1_3", dt_RGD)
#     np.save("alpha_RGD_ex1_3", alpha_RGD)
#     np.save("delta_RGD_ex1_3", delta_RGD)
#
#     np.save("mu_CRGD_ex1_3", mu_CRGD)
#     np.save("dt_CRGD_ex1_3", dt_CRGD)
#     np.save("alpha_CRGD_ex1_3", alpha_CRGD)
#     np.save("delta_CRGD_ex1_3", delta_CRGD)
#
#
# # -------- Tuning process for example 1 ---------
# # To simulate  parameters for Example 1, uncommented the following line
# params_Fig1(n_times=100, num_trials=50)
