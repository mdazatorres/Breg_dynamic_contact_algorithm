from integrators import Breg, NAG, gradh, HTVI_d, HTVI_adap, HTVI_Bet_d
import numpy as np
import matplotlib.pyplot as plt
import examples as exa
import scipy.integrate as si
import pylab

#v = 1000
#m = 0.01
def plot_order(method, color, marker, params, dt, steps, ex, init):
    x0, x0_t, p0, p0_t = init
    v, m, c, C = params
    g0 = 2
    if method == 'RB':  # Bregman Relativistic
        solX, solP = Breg(ex, [v, m, c, C], dt, steps, [x0, p0], adap=False, which='None')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'RB_adap':  # Bregman Relativistic adaptative
        solX, solP = Breg(ex, [v, m, c, C], dt, steps,  [x0, p0],  adap=True, which='old')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        order = np.log10(fsol[-1]) / np.log10(steps-1)
        label = method

    elif method == 'HTVI_adap':
        solX = HTVI_adap(ex, c, dt, steps, init)
        fsol = np.apply_along_axis(ex.f, 1, solX)
        order = np.log10(fsol[-1]) / np.log10(steps - 1)
        label = method

    elif method == 'Betancourt':
        solX = HTVI_Bet_d(ex, c, dt, steps, [x0, p0, p0_t])
        fsol = np.apply_along_axis(ex.f, 1, solX)
        order = np.log10(fsol[-1]) / np.log10(steps - 1)
        label = method

    else:
        print('error to introduce methods name')

    plt.semilogy(fsol, label=label, linewidth=2, color=color, marker=marker, fillstyle='none', markevery=int(steps/15),
                 markersize=12)

    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    plt.legend(loc=3)


def plot_contourn(ex, xmin, xmax, ymin, ymax):

    plt.rcParams['contour.negative_linestyle'] = 'solid'

    x_ = np.linspace(xmin, xmax, 300)
    y_ = np.linspace(ymin, ymax, 300)
    X, Y = np.meshgrid(x_, y_)
    plt.figure(figsize=(6, 4))
    cs = pylab.contourf(X, Y, ex.f2D(X, Y), levels=23, cmap='PuBu')
    plt.contour(X, Y, ex.f2D(X, Y), levels=23, linewidths=1.0, colors='k')
    cbar = pylab.colorbar(cs)
    #cbar_ticks = [0, 3, 6, 9]
    cbar.ax.set_autoscale_on(True)
    #cbar.set_ticks(cbar_ticks)

    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    #plt.xticks([-2, -1, 0, 1, 2])
    #plt.yticks([-2, -1, 0, 1, 2])
    plt.tight_layout()
