from integrators import Breg, NAG, CM, HTVI_adap, HTVI_d,Bet_dir
from integratorFJ import FJ
import numpy as np
import matplotlib.pyplot as plt
import pylab

plt.rcParams['font.size'] = 18

def eval_fun(method, params, dt, steps, ex, init):
    x0, x0_t, x0_, p0, p0_t, p0_, t0, t0_ = init
    v, m, c, C, mu, e = params
    if method == 'RB':  # Bregman Relativistic
        solX, solP = Breg(ex, [v, m, c, C], dt, steps, [x0, p0], adap=False,new=False, kinetic='Relativistic')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'EB':  # Bregman Euclidean
        solX, solP = Breg(ex, [v, m, c, C], dt, steps, [x0, p0], adap=False,new=False, kinetic='Quadratic')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'RB_adap':  # Bregman Relativistic adaptative
        solX, solP = Breg(ex, [v, m, c, C], dt, steps,  [x0, p0],  adap=True,new=False, kinetic='Relativistic')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        #order = np.log10(fsol[-1]) / np.log10(steps-1)
        label = method

    elif method == 'EB_adap':  # Bregman Euclidean adaptative
        solX, solP = Breg(ex, [v, m, c, C], dt, steps,  [x0, p0],  adap=True,new=False, kinetic='Quadratic')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        #order = np.log10(fsol[-1]) / np.log10(steps-1)
        label = method

    elif method == 'HTVI_adap':
        solX = HTVI_adap(ex, c, dt, steps, [x0, x0_t, p0, p0_t])
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'FJ_R':
        solX = FJ(ex, dt, [v, m, c, C, e], steps, [x0, x0_, p0, p0_, t0, t0_])
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'Betancourt':
        solX = Bet_dir(ex, c, dt, steps, [x0, p0, p0_t])
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'NAG':
        solX = NAG(ex, mu, dt, steps)
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'CM':
        solX = CM(ex, mu, dt, steps)
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    else:
        fsol=0

    return fsol




def plot_order(method, color, marker, params, dt, steps, ex, init):
    #x0, x0_t, p0, p0_t = init
    x0, x0_t, x0_, p0, p0_t, p0_, t0, t0_ = init

    #initFJ = [x0, x0_, p0, p0_, t0, t0_]
    #v, m, c, C, mu = params
    v, m, c, C, mu, e = params
    if method == 'RB':  # Bregman Relativistic
        solX, solP = Breg(ex, [v, m, c, C], dt, steps, [x0, p0], adap=False, new=False, kinetic='Relativistic')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'EB':  # Bregman Euclidean
        solX, solP = Breg(ex, [v, m, c, C], dt, steps, [x0, p0], adap=False, new=False, kinetic='Quadratic')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'RB_adap':  # Bregman Relativistic adaptative
        solX, solP = Breg(ex, [v, m, c, C], dt, steps,  [x0, p0],  adap=True, new=False, kinetic='Relativistic')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        #order = np.log10(fsol[-1]) / np.log10(steps-1)
        label = method

    elif method == 'EB_adap':  # Bregman Euclidean adaptative
        solX, solP = Breg(ex, [v, m, c, C], dt, steps,  [x0, p0],  adap=True, new=False, kinetic='Quadratic')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        #order = np.log10(fsol[-1]) / np.log10(steps-1)
        label = method

    elif method == 'RB_adap_new':  # Bregman Relativistic adaptative
        solX, solP = Breg(ex, [v, m, c, C], dt, steps, [x0, p0], adap=True, new=True, kinetic='Relativistic')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        # order = np.log10(fsol[-1]) / np.log10(steps-1)
        label = method

    elif method == 'EB_adap_new':  # Bregman Euclidean adaptative
        solX, solP = Breg(ex, [v, m, c, C], dt, steps, [x0, p0], adap=True, new=True, kinetic='Quadratic')
        fsol = np.apply_along_axis(ex.f, 1, solX)
        # order = np.log10(fsol[-1]) / np.log10(steps-1)
        label = method

    elif method == 'HTVI_adap':
        solX = HTVI_adap(ex, c, dt, steps, [x0, x0_t, p0, p0_t])
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'HTVI_d':
        solX = HTVI_d(ex, c, dt, steps, [x0, x0_t, p0, p0_t])
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method =='FJ':
        solX = FJ(ex, dt, [v, m, c, C, e], steps, [x0, x0_, p0, p0_, t0, t0_])
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'Betancourt':
        solX = Bet_dir(ex, c, dt, steps, [x0, p0, p0_t])
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'NAG':
        solX = NAG(ex, mu, dt, steps)
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'CM':
        solX = CM(ex, mu, dt, steps)
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    else:
        print('error to introduce methods name')

    plt.semilogy(fsol, label=label, linewidth=2, color=color, marker=marker, fillstyle='none', markevery=int(steps/15),
                 markersize=12)
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    plt.legend(loc=3)


def plot_contourn(ex, xmin, xmax, ymin, ymax,num_ex, save):
    plt.figure(figsize=(8, 6))
    #plt.figure(figsize=(5, 3))
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    x_ = np.linspace(xmin, xmax, 300)
    y_ = np.linspace(ymin, ymax, 300)
    X, Y = np.meshgrid(x_, y_)
    cs = pylab.contourf(X, Y, ex.f2D(X, Y), levels=23, cmap='hot')
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
    if save:
        plt.savefig('cotourn_ex_'+str(num_ex)+'.png')


