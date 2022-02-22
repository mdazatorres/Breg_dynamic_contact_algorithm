import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from integrators import CM, NAG
from example_breast_cancer_LR import gradLp, Lp, loss_func_test, n, Xtrain, Ytrain
from integrators_Breg import Breg
import pandas as pd
from sklearn.preprocessing import scale


plt.rcParams['font.size'] = 25

# dt = 0.01
max_ite = 200
# vc = 100000
# mass = 0.0001
# mu = 0.9
# params = [vc, mass, mu]

steps = 200
num_trials = 100

def init_sample(num_trials):
    np.random.seed(18)
    #init = np.random.rand(num_trials, n + 1)
    init=np.zeros((num_trials, n + 1))+1e-8
    return init


def results(num_trials, steps, method):
    Loss = np.empty([num_trials, steps], dtype=np.float64)
    Losst = np.empty([num_trials, steps], dtype=np.float64)
    E = np.empty([num_trials, steps], dtype=np.float64)
    Et = np.empty([num_trials, steps], dtype=np.float64)

    if method == CM:
        dt = 0.00001
        mu = 0.8925
        vc = 0
        m = 0
    elif method == NAG:
        dt = 0.00001
        mu = 0.8925
        vc = 0
        m = 0
    else:
        dt = 0.01 # no adap 0.1 # 0.000001 adap 2
        m = 0.01  # m
        vc = 1000 # v
        mu=0

    params = [dt, mu, m, vc]
    init0 = init_sample(num_trials)
    L = lambda w: Lp(w, Xtrain, Ytrain)
    gradL = lambda w: gradLp(w, Xtrain, Ytrain)
   # kinetic = 'Relativistic' #'Quartic', 'Logaritmic', 'Quadratic'
    kinetic ='Relativistic'
    for i in range(num_trials):
        init = init0[i]
        if method==Breg:
            w, loss = method(params, init, gradL, L, max_ite, kinetic, adap=False, new=True, tol=1e-10)
        else:
            w, loss = method(params, init, gradL, L, max_ite, tol=1e-10)
        Loss[i] = loss[0: steps]
        Et[i], E[i], Losst[i] = loss_func_test(w)[:, 0:steps]
    return Loss, Losst, E, Et


def plot_r(a, color, marker, fillstyle, markevery, method):
    a_mean = np.mean(a, axis=0)
    a_q25 = np.quantile(a, .025, axis=0)
    a_q90 = np.quantile(a, .975, axis=0)

    plt.plot(a_mean, color=color, marker=marker, fillstyle=fillstyle, markevery=markevery, label=method)
    plt.fill_between(np.arange(len(a_mean)), a_q25, a_q90, color=color, alpha=0.1)


def plot_results(L1, L3, L5, ylabel):
    plt.figure(figsize=(10, 7))
    plot_r(L1, color='b', marker='D', fillstyle='none', markevery=5, method='CM')
    plot_r(L3, color='r', marker='o', fillstyle='none', markevery=5, method='NAG')
    #plot_r(L4, color='g', marker='s', fillstyle='none', markevery=5, method='RGD')
    plot_r(L5, color='y', marker='<', fillstyle='none', markevery=5, method='Breg')
    #plt.xticks([0, 4, 8, 12, 16, 20])
    plt.xlabel('iterations')
    plt.ylabel(ylabel)
    plt.legend()


L1, L1t, E1, E1t = results(num_trials, steps,  CM)
L3, L3t, E3, E3t = results(num_trials, steps,  NAG)
#L4, L4t, E4, E4t = results(num_trials, steps, RGD)
L5, L5t, E5, E5t = results(num_trials, steps,  Breg)


plot_results( L1, L3, L5, ylabel='loss')
plot_results(L1t, L3t, L5t, ylabel='loss')
plot_results(E1, E3, E5, ylabel='classification error')
plot_results(E1t, E3t,E5t, ylabel='classification error')


# plot_resutls(L_SGD, L_CRGD, L_RGD, L_NAG, 'Loss')
# plot_resutls(L_SGDt, L_CRGDt, L_RGDt, L_NAGt, 'Loss')
#
# plot_resutls(e_SGD, e_CRGD, e_RGD, e_NAG, 'error')
# plot_resutls(e_SGDt, e_CRGDt, e_RGDt, e_NAGt, 'error')