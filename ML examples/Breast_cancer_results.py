import matplotlib.pyplot as plt
import numpy as np
from integrators import CM, NAG, Breg
from Breast_cancer_LR_ex import gradLp, Lp, loss_func_test, n, Xtrain, Ytrain


plt.rcParams['font.size'] = 30
max_ite = 200
steps = max_ite
num_trials = 10

def init_sample(num_trials):
    np.random.seed(18)
    #init = np.random.rand(num_trials, n + 1)
    init=np.zeros((num_trials, n + 1))#+1e-8
    return init


def results(num_trials, steps, method, kinetic):
    Loss = np.empty([num_trials, steps], dtype=np.float64)
    Losst = np.empty([num_trials, steps], dtype=np.float64)
    E = np.empty([num_trials, steps], dtype=np.float64)
    Et = np.empty([num_trials, steps], dtype=np.float64)

    if method == CM:
        dt = 0.1
        mu = 0.8925
        vc = 0
        m = 0
    elif method == NAG:
        dt = 0.1
        mu = 0.8925
        vc = 0
        m = 0

    else:
        dt = 0.1# no adap 0.1
        m = 0.01  # m
        vc = 1000 # v
        mu=0

    params = [dt, mu, m, vc]
    init0 = init_sample(num_trials)
    L = lambda w: Lp(w, Xtrain, Ytrain)
    gradL = lambda w: gradLp(w, Xtrain, Ytrain)

    for i in range(num_trials):
        init = init0[i]
        if method==Breg:
            w, loss = method(params, init, gradL, L, max_ite, kinetic, adap=False, tol=1e-10)
        else:
            w, loss = method(params, init, gradL, L, max_ite, tol=1e-10)
        Loss[i] = loss[0: steps]
        Et[i], E[i], Losst[i] = loss_func_test(w)[:, 0:steps] # Error test, Error training, Loss_test
    return Loss, Losst, E, Et


def plot_r(a, color, marker, fillstyle, markevery, method):
    a_mean = np.mean(a, axis=0)
    a_q25 = np.quantile(a, .025, axis=0)
    a_q90 = np.quantile(a, .975, axis=0)

    plt.plot(a_mean, lw=2, color=color, marker=marker,markeredgewidth=2, fillstyle=fillstyle, markevery=markevery, markersize=14,label=method,)
    #plt.fill_between(np.arange(len(a_mean)), a_q25, a_q90, color=color, alpha=0.1)


def plot_results(L_CM, L_NAG, L_EB, L_RB, ylabel, save,title):
#def plot_results(L_EB, L_RB, ylabel, save, title):
    markevery=30
    plt.figure(figsize=(10, 7))
    plot_r(L_CM, color='b', marker='D', fillstyle='none', markevery=markevery, method='CM')
    plot_r(L_NAG, color='r', marker='o', fillstyle='none', markevery=markevery, method='NAG')
    plot_r(L_EB, color='g', marker='s', fillstyle='none', markevery=markevery, method='EB')
    plot_r(L_RB, color='black', marker='<', fillstyle='none', markevery=markevery, method='RB')
    #plt.xticks([0, 4, 8, 12, 16, 20])
    plt.xlabel('Iterations')
    plt.ylabel(ylabel)
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig('figures/ex_cancer_' + title + '.png')



L_CM, Lt_CM, E_CM, Et_CM = results(num_trials, steps,  CM, kinetic='Relativistic')
L_NAG, Lt_NAG, E_NAG, Et_NAG = results(num_trials, steps,  NAG, kinetic='Relativistic')
L_EB, Lt_EB, E_EB, Et_EB = results(num_trials, steps, Breg, kinetic='Quadratic')
L_RB, Lt_RB, E_RB, Et_RB = results(num_trials, steps,  Breg, kinetic='Relativistic')

plot_results(L_CM, L_NAG, L_EB, L_RB, ylabel='Loss function', save=True, title='lossf_training')
plot_results(Lt_CM, Lt_NAG, Lt_EB, Lt_RB, ylabel='Loss function', save=True, title='lossf_test')
plot_results(E_CM, E_NAG, E_EB, E_RB, ylabel='Classification error', save=True, title='error_training')
plot_results(Et_CM, Et_NAG, Et_EB, Et_RB, ylabel='Classification error', save=True, title='error_test')



