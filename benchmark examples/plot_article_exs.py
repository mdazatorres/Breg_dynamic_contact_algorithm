import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import examples as exa
import scipy.integrate as si
from plot_ex import plot_order, plot_contourn, eval_fun
from integrators import HBr
plt.rcParams['font.size'] = 20

methods = ['RB', 'RB_adap', 'EB', 'EB_adap']



def fig_comp(ex, dt, c, steps, save):
    C = 0.1  # C = 2.302585093
    m = 0.01
    v = 1000
    mu = 0.8
    steps = 20000

    p0_t = -HBr(ex, ex.x0, ex.p0, ex.x0_t, c)
    init = [ex.x0, ex.x0_t, ex.p0, p0_t]
    plt.figure(figsize=(8, 6))

    fsol_RB = eval_fun('RB',  params=[v, m, c, C, mu], dt=dt, steps=steps, ex=ex, init=init)
    fsol_EB = eval_fun('EB',  params=[v, m, c, C, mu], dt=dt, steps=steps, ex=ex, init=init)

    plt.semilogy(fsol_RB, label='RB', linewidth=2, color='k', marker='h', fillstyle='none', markevery=int(steps / 15),
                 markersize=12)
    plt.semilogy(fsol_EB, label='EB', linewidth=2, color='blue',ls='dashed', marker='h', fillstyle='none', markevery=int(steps / 15),
                 markersize=12)

    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    plt.legend(loc=3)
    plt.tight_layout()
    if save:
        plt.savefig('ex_Quaartic' + '.png')

def orderf(fsol, ite):
    return np.log10(fsol[ite]) / np.log10(ite -1)



def plot_comp_order(ex, dt,steps, func, save):
    C =1
    m=0.01; v=1000; mu=0.8

    c=2
    p0_t = -HBr(ex, ex.x0, ex.p0, ex.x0_t, c)
    init =[ex.x0, ex.x0_t, ex.p0, p0_t]

    colors = ['b', 'r']
    styles = ['-', '--', '-.']
    marker=['<','H','d']
    order=[2,4,8]
    methods=['RB','EB']
    fig, ax = plt.subplots(figsize=(8,6))
    ite=1500
    tt=np.linspace(ite,steps,steps-ite+1)


    for cc, col in enumerate(colors):
        for ss, sty in enumerate(styles):
            fsol_RB = eval_fun(methods[cc],  params=[v, m, order[ss], C, mu], dt=dt, steps=steps, ex=ex, init=init)
            ax.semilogy(fsol_RB,  linewidth=2, color=colors[cc], ls=styles[ss])
            #order_num1 = np.log10(fsol_RB[-5000]) / np.log10(steps -1-5000)
            #order_num = [orderf(fsol_RB, 4000), orderf(fsol_RB, 8000),orderf(fsol_RB, steps-1)]  #quadratic
            order_num = [orderf(fsol_RB, 1000), orderf(fsol_RB, 2000), orderf(fsol_RB, steps - 1)] #quartic
            print(methods[cc],order[ss], order_num)
            #ax.loglog(tt, tt**(order_num), '-m')
            #ax.semilogy(fsol_RB, linewidth=2, color=colors[cc], marker=marker[ss], fillstyle='none', markevery=int(steps / 15), markersize=12)  # ls=styles[ss]

    for cc, col in enumerate(colors):
        ax.plot(np.NaN, np.NaN, c=colors[cc],  linewidth=2, label=methods[cc])

    ax2 = ax.twinx()
    for ss, sty in enumerate(styles):
        #ax2.plot(np.NaN, np.NaN, marker=marker[ss],label=str(order[ss]), fillstyle='none', markevery=int(steps / 15), markersize=12, c='black')
        ax2.plot(np.NaN, np.NaN, ls=styles[ss],  linewidth=2, label=str(order[ss]), c='black')
    ax2.get_yaxis().set_visible(False)

    ax.legend(loc=1)
    ax2.legend(title='Order',loc=3)

    plt.show()
    plt.tight_layout()
    if save:
        plt.savefig('figures/ex_comp_'+func + '.png')

#ex=exa.Ex_Quartic_1(n=10)
#plot_comp_order(ex, dt=1e-3,steps=4000, func='Quartic', save=True)
#plot_contourn(ex, xmin=-10, xmax=10, ymin=-10, ymax=10, num_ex='Quartic',save=True)

ex = exa.Ex_Quadratic(n=100)
plot_comp_order(ex, dt=1e-4, steps=4000, func='Quadratic', save=True)
plot_contourn(ex, xmin=-10, xmax=10, ymin=-10, ymax=10, num_ex='Quadratic',save=True)

"""
ex = exa.Ex_Quadratic(n=100)
plot_comp_order(ex, dt=1e-4, steps=10000, func='Quadratic', save=True)
plot_contourn(ex, xmin=-10, xmax=10, ymin=-10, ymax=10, num_ex='Quadratic',save=True)


ex = exa.Ex_Corr_Quadratic(n=50)
plot_comp_order(ex, dt=1e-4, steps=10000, func='Corr_Quadratic', save=True)
plot_contourn(ex, xmin=-10, xmax=10, ymin=-10, ymax=10, num_ex='Corr_Quadratic',save=True)

ex=exa.Ex_Quartic_1(n=10)
plot_comp_order(ex, dt=1e-3,steps=4000, func='Quartic', save=True)
plot_contourn(ex, xmin=-10, xmax=10, ymin=-10, ymax=10, num_ex='Quartic',save=True)
"""




#ex = exa.Ex_Schwefel(n=20)
#plot_comp_order(ex, dt=1e-3, steps=50000, func='Quartic', save=True)
#ex = exa.Ex_Three_hump()
#plot_comp_order(ex, dt=1e-5, steps=10000, func='Three hump', save=True)

