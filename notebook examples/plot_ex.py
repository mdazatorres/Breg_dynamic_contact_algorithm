from integrators import Breg, NAG, CM, HTVI_adap, Bet_dir
import numpy as np
import matplotlib.pyplot as plt
import pylab


def plot_order(method, color, marker, params, dt, steps, ex, init):
    x0, x0_t, p0, p0_t = init
    v, m, c, C, mu = params
    if method == 'RB':  # Bregman Relativistic
        solX, solP = Breg(ex, [v, m, c, C], dt, steps, [x0, p0], adap=False)
        fsol = np.apply_along_axis(ex.f, 1, solX)
        label = method

    elif method == 'RB_adap':  # Bregman Relativistic adaptative
        solX, solP = Breg(ex, [v, m, c, C], dt, steps,  [x0, p0],  adap=True)
        fsol = np.apply_along_axis(ex.f, 1, solX)
        #order = np.log10(fsol[-1]) / np.log10(steps-1)
        label = method

    elif method == 'HTVI_adap':
        solX = HTVI_adap(ex, c, dt, steps, init)
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


def tuning_process(method, ex, steps = 200):
    '''
    In this function, we do the random parameter search for each method (CM, NAG, RGD, CRGD).
    '''
    min_f = 1e+16
    mu_new = 0
    dt_new = 0
    c_new = 0
    C_new = 0
    dt= np.linspace(1e-5, 0.1,12)
    mu= np.linspace(0.8, 0.99,5)
    c= np.linspace(1,6,7)
    C=np.linspace(1,4,5)
    m = 0.01
    v = 1000

    p0_t = -HBr(ex, ex.x0, ex.p0, ex.x0_t, c)  # np.zeros(q0.shape)
    init = [ex.x0, ex.x0_t, ex.p0, p0_t]
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

    if method == 'HTVI_adap':
        for i in range(len(c)):
                for k in range(len(dt)):
                    solX = HTVI_adap(ex, c[i], dt[k], steps, init)
                    f_sim = np.apply_along_axis(ex.f, 1, solX)
                    min_fnew = min(f_sim)
                    if min_fnew < min_f:
                        min_f = min_fnew
                        c_new = c[i]
                        dt_new = dt[k]
    if method == 'Betancourt':
        for i in range(len(c)):
                for k in range(len(dt)):
                    solX = Bet_dir(ex, c[i], dt[k], steps, [ex.x0, ex.p0, p0_t])
                    f_sim = np.apply_along_axis(ex.f, 1, solX)
                    min_fnew = min(f_sim)
                    if min_fnew < min_f:
                        min_f = min_fnew
                        c_new = c[i]
                        dt_new = dt[k]

    if method == 'CM' or 'NAG':
        for i in range(len(c)):
                for k in range(len(dt)):
                    solX = Bet_dir(ex, c[i], dt[k], steps, [ex.x0, ex.p0, p0_t])
                    f_sim = np.apply_along_axis(ex.f, 1, solX)
                    min_fnew = min(f_sim)
                    if min_fnew < min_f:
                        min_f = min_fnew
                        c_new = c[i]
                        dt_new = dt[k]

    return mu_new, dt_new, c_new, C_new
