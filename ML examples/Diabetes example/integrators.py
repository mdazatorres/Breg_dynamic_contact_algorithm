import numpy as np
from scipy import linalg as lg
import warnings


#def CM(mu, dt, steps, xinit, gradf, f, tol):
def CM(params, xinit, gradf, f, steps, tol):
    dt, mu, _ , _ = params
    x0 = np.copy(xinit)
    p0 = np.zeros(x0.shape)
    X = []
    V = []
    X.append(x0)
    V.append(f(x0))
    cond = 10
    max_ite = 0
    while cond > tol:
        g = gradf(x0)
        p = mu * p0 - dt*g
        x = x0 + p
        p0 = p
        x0 = x
        V.append(f(x0))
        cond = lg.norm(gradf(x0))
        X.append(x0)
        if max_ite == steps:
            print('CM does not converge')
            break
        max_ite += 1
    return np.array(X), V


def NAG(params, xinit, gradf, f, steps, tol):
    dt, mu, _, _ = params
    x0 = np.copy(xinit)
    v0 = np.zeros(x0.shape)
    X = []
    V = []
    X.append(x0)
    V.append(f(x0))
    cond = tol+1
    max_ite = 0
    while cond > tol:
        g = gradf(x0 + mu*v0)
        v = mu * v0 - dt*g
        x = x0 + v
        X.append(x)
        v0 = v
        x0 = x
        V.append(f(x0))
        cond = lg.norm(gradf(x0))
        if max_ite == steps:
            print('NAG does not converge')
            break
        max_ite += 1
    return np.array(X), V


def RGD(params, xinit, gradf, f, steps, tol):
    dt, mu, m, vc = params
    x0 = np.copy(xinit)
    p0 = np.zeros(x0.shape)
    X = []
    V = []
    V.append(f(x0))
    X.append(x0)
    cond = tol+1
    max_ite = 0

    while cond > tol:
        g = gradf(x0)
        p = mu * p0 - dt * g
        x = x0 + (dt * vc * p)/(np.sqrt(lg.norm(p)**2 + (m * vc)**2))
        p0 = p
        x0 = x
        X.append(x)
        V.append(f(x0))

        cond = lg.norm(gradf(x0))

        if max_ite == steps:
            print('RGD does not converge')
            break
        max_ite += 1
    return np.array(X), V


def step_CRGD(p, x, t, params, gradf):

    # dt/2 D
    dt, mu, m, vc = params
    t = t + dt / 2

    # dt/2 A
    etf = mu ** (1 / (2 * t) + 1 / 2)
    p = p * etf

    # dt/2 B
    p = p - gradf(x) * dt / 2

    # dt C
    x = x +  (dt * vc * p) / (np.sqrt(lg.norm(p) ** 2 + (m * vc) ** 2))

    # dt/2 B
    p = p - gradf(x) * dt / 2

    # dt/2 A
    etf = mu ** (1 / (2 * t) + 1 / 2)
    p = p * etf

    # dt/2 D
    t = t + dt / 2

    return (p, x, t)



def CRGD(params, xinit, gradf, f, steps, tol):
    dt, mu, m, vc = params
    x0 = np.copy(xinit)
    ttol = 1e-13
    p0 = np.zeros(xinit.shape)
    cond = 1
    i = 0
    max_ite = 0
    X = []
    P = []
    P.append(p0)
    X.append(x0)
    V = []
    V.append(f(x0))
    while cond > tol:
        #print('valu3', x0)
        t = dt * i
        pnew, xnew, tnew = step_CRGD(p0, x0, t, params, gradf)

        if abs(tnew - t - dt) > ttol:
            warnings.warn(f"tnew-t-dt, dt inconsistency: {tnew - t - dt}, {dt}")
        x0 = np.copy(xnew)
        p0 = np.copy(pnew)
        X.append(x0)
        V.append(f(x0))

        cond = lg.norm(gradf(X[-1]))
        if max_ite == steps:
            print('CRGD does not converge')
            break
        max_ite += 1

    return X, V


def step_CNAG(p, x, s, t, params, gradf, f):  # CNAG
    dt, mu, _, _ = params
    # dt/2 D
    t = t + dt/2.

    # dt/2 A
    # etf = mu ** (np.exp(s) / 2.0)
    # etf = mu ** (s ** 2 / 2.0)
    etf = mu ** (s ** 3 / 2.0)
    p = p * etf
    sp = 1 - s ** 2 * np.log(mu) / 3.0
    s = np.sqrt(s ** 2 / sp)

    # dt/2 B
    p = p - gradf(x) * dt/2
    s = s - f(x) * dt/2

    # dt C
    x = x + p * dt
    s = s + np.linalg.norm(p) ** 2 * dt/2

    # dt/2 B
    p = p - gradf(x) * dt/2
    s = s - f(x) * dt/2

    # dt/2 A
    etf = mu ** (s ** 3 / 2.0)
    # etf = mu ** (np.exp(s) / 2.0)
    # etf = mu ** (s ** 2 / 2.0)
    p = p * etf
    sp = 1 - s ** 2 * np.log(mu)/3.0
    s = np.sqrt(s ** 2 / sp)
    # dt/2 D
    t = t + dt/2

    return (p, x, s, t)


def CNAG(params, xinit, gradf, f, steps, tol):
    dt, mu, m, vc = params
    x0 = np.copy(xinit)
    s0 = 0
    ttol = 1e-13
    p0 = np.zeros(xinit.shape)
    cond = 1
    i = 0
    max_ite = 0
    X = []
    X.append(x0)
    V = []
    V.append(f(x0))

    while cond > tol:
        #print('valu3', x0)
        t = dt * i
        pnew, xnew, snew, tnew = step_CNAG(p0, x0, s0, t, params, gradf, f)

        if abs(tnew - t - dt) > ttol:
            warnings.warn(f"tnew-t-dt, dt inconsistency: {tnew - t - dt}, {dt}")
        x0 = np.copy(xnew)
        p0 = np.copy(pnew)
        X.append(x0)
        V.append(f(x0))

        cond = lg.norm(gradf(X[-1]))
        if max_ite == steps:
            print('CNAG does not converge')
            break
        max_ite += 1

    return X, V







