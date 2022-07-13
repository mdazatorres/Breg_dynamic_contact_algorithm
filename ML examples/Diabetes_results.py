import matplotlib.pyplot as plt
import numpy as np
from integrators import CM, NAG, Breg
from Diabetes_LR_ex import gradLp, Lp, loss_func_test, n, Xtrain, Ytrain


plt.rcParams['font.size'] = 20
# dt = 0.01
max_ite = 200
steps = max_ite
num_trials = 10

def init_sample(num_trials):
    np.random.seed(18)
    init = np.random.rand(num_trials, n + 1)
    #init = np.zeros((num_trials, n + 1))   #+1e-3
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
        dt = 0.07502500000000001
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


def plot_r(a, color, marker, fillstyle, markevery, method, ax):
    a_mean = np.mean(a, axis=0)
    a_q25 = np.quantile(a, .025, axis=0)
    a_q90 = np.quantile(a, .975, axis=0)
    xx=np.arange(len(a_mean))+1
    #xx[0]=1
    ax.loglog(xx, a_mean, color=color,lw=2, marker=marker,markeredgewidth=2, fillstyle=fillstyle, markevery=markevery, markersize=16,label=method,)
    ax.fill_between(np.arange(len(a_mean))+1, a_q25, a_q90, color=color, alpha=0.1)
    #ax.set_yscale('log')
    #ax.set_xscale('log')


def plot_results(L_CM, L_NAG, L_EB, L_RB, ylabel, save, title):
    fig, ax = plt.subplots(figsize=(10, 7))
    #plt.figure(figsize=(10, 7))
    markevery=30
    plot_r(L_CM, color='b', marker='D', fillstyle='none', markevery=markevery, method='CM',ax=ax)
    plot_r(L_NAG, color='r', marker='o', fillstyle='none', markevery=markevery, method='NAG',ax=ax)
    plot_r(L_EB, color='g', marker='s', fillstyle='none', markevery=markevery, method='EB',ax=ax)
    plot_r(L_RB, color='black', marker='<', fillstyle='none', markevery=markevery, method='RB',ax=ax)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(ylabel)
    ax.legend()

    fig.tight_layout()
    if save:
        plt.savefig('figures/ex_glucosa_' + title + '.png')


L_CM, Lt_CM, E_CM, Et_CM = results(num_trials, steps,  CM, kinetic='Relativistic')
L_NAG, Lt_NAG, E_NAG, Et_NAG = results(num_trials, steps,  NAG, kinetic='Relativistic')
L_EB, Lt_EB, E_EB, Et_EB = results(num_trials, steps, Breg, kinetic='Quadratic')
L_RB, Lt_RB, E_RB, Et_RB = results(num_trials, steps,  Breg, kinetic='Relativistic')


#plot_results(L_CM, L_NAG, L_EB, L_RB, ylabel='Loss function', save=True,title='lossf_training')
plot_results(Lt_CM, Lt_NAG, Lt_EB, Lt_RB, ylabel='Loss function', save=True,title='lossf_test')
#plot_results(E_CM, E_NAG, E_EB, E_RB, ylabel='Classification error', save=True,title='error_training')
plot_results(Et_CM, Et_NAG, Et_EB, Et_RB, ylabel='Classification error', save=True,title='error_test')


# x=[1e-15,1,2,3,4,5]
# y1=[2,3,7,9,10,12]
# y2=[2,8,9,14,16,19]
# y3=[2,5,8,13,20,25]
# plt.loglog(x,y1)
# plt.loglog(x,y2)
# plt.loglog(x,y3)
