import numpy as np
from scipy import linalg as lg
import warnings



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




def HBr(example, q,p,q0_t,c):
    C=1
    return c/(2*q0_t**(c+1)) * np.dot(p, p) + C * c * q0_t ** (2*c-1) * example.f(q)

def ea(c, t):
    return c/t

def eb(t, c, C):
    return np.exp(C) * t ** c


def eg(t, c, C):
    return t ** c

def step_Br(p, x, t, params, gradf, kinetic):
    #[dt, mu, m, vc]
    dt, _, m, v = params
    C = 1
    c = 2
    # dt/2 t
    t += dt / 2

    # dt/2 D
    p = p * np.exp(-ea(c, t) * dt / 2)
    # dt/2 B
    p = p * np.exp(ea(c, t) * dt/2)
    x = x * np.exp(-ea(c, t) * dt/2)
    # dt/2 C
    p = - ea(c, t) * eb(t, c, C) * gradf(x) * dt/2 + p

    # dt A
    if kinetic == 'Quartic':
        x = ea(c, t) * p ** (1 / 3) * dt + x

    elif kinetic == 'Logaritmic':
        x = -ea(c, t) * dt/p + x

    elif kinetic == 'Quadratic':
        x = ea(c, t) * p * dt + x

    else: # Relativistic
        sq = np.sqrt(v ** 2 * m ** 2 + np.sum(p**2))
        x = (ea(c, t) * v * p / sq) * dt + x

    # dt/2 C
    p = - ea(c, t) * eb(t, c, C) * gradf(x) * dt/2 + p
    # dt/2 B
    p = p * np.exp(ea(c, t) * dt/2)
    x = x * np.exp(-ea(c, t) * dt/2)
    # dt/2 D
    p = p * np.exp(-ea(c, t) * dt / 2)
    # dt/2 t
    t += dt / 2

    return (p, x, t)




def gradh(x, params):
    dt, _, m, v = params
    return m * x / np.sqrt(1 - np.dot(x, x) / v ** 2)


def Breg(params, xinit, gradf, f,  steps, kinetic, adap, tol):
    dt, _, m, v = params
    x0 = np.copy(xinit)
    ttol = 1e-13
    #t0 = 0.01

    if kinetic == 'Relativistic':
        p0 = gradh(xinit, params)
    elif kinetic == 'Quartic':
        p0 = xinit ** 3
    else:
        p0 = xinit
    p0 = np.zeros(xinit.shape)
    cond = 1
    i = 1
    max_ite = 0
    X = []
    P = []
    P.append(p0)
    X.append(x0)
    V = []
    V.append(f(x0))
    t0=1e-5
    while cond > tol:
        pnew, xnew, tnew = step_Br(p0, x0, t0, params, gradf, kinetic)
        x0 = np.copy(xnew)
        p0 = np.copy(pnew)
        X.append(x0)
        V.append(f(x0))
        t0=tnew

        cond = lg.norm(gradf(X[-1]))
        if max_ite == steps:
            print('CRGD does not converge')
            break
        max_ite += 1
    return X, V









