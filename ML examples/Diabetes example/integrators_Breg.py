import numpy as np
from scipy import linalg as lg
import warnings

def HBr(example, q,p,q0_t,c):
    C=1
    return c/(2*q0_t**(c+1)) * np.dot(p, p) + C * c * q0_t ** (2*c-1) * example.f(q)

def ea(c, t):
    return c/t

def eb(t, c, C):
    return C * t ** c


def eg(t, c, C):
    return t ** c

def step_Br(p, x, t, params, gradf):
    #[dt, mu, m, vc]
    dt, _, m, v = params
    C = 2
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


def step_Br_adap_new(p, x, t, params, gradf, kinetic): #  va a quedar
    C = 1
    c = 2
    dt, _, m, v = params
    # dt/2 t
    t += dt / 2   # T+= dT
    # dt/2 D
    p = p * np.exp(-1/t * dt / 2)

    # dt/2 B
    p = p * np.exp(1/t * dt/2)
    x = x * np.exp(-1/t * dt/2)

    # dt/2 C
    p = - gradf(x) * dt/2 + p

    # dt A
    if kinetic == 'Quartic':
        x = 1/t * p ** (1 / 3) * dt + x
    elif kinetic == 'Logaritmic':
        x = -1/t * dt/p + x
    elif kinetic == 'Quadratic':
        x = 1/t * p * dt + x
    else: # Relativistic
        sq = np.sqrt(v ** 2 * m ** 2 + np.sum(p**2))
        x = (1/t * v * p / sq) * dt + x

    # dt/2 C
    p = -gradf(x) * dt/2 + p
    # dt/2 B
    p = p * np.exp(1/t * dt/2)
    x = x * np.exp(-1/t * dt/2)

    # dt/2 D
    p = p * np.exp(- 1/t * dt / 2)

    # dt/2 t
    t += dt / 2

    return (p, x, t)


def step_Br_adap(p, x, t, params, gradf, kinetic):
    C = 1
    c = 2

    dt, _, m, v = params
    # dt/2 t
    T = eb(t, c, C)
    arg = T + dt / 2
    t = (C * arg) ** (1 / c)

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
    T = eb(t, c, C)
    arg = T + dt / 2
    t = (C * arg) ** (1 / c)

    return (p, x, t)

def g(t):
    g0=2
    return g0 / (g0 - (g0-1) * np.exp(-t))

def step_Br_new(p, x, t, params, gradf, kinetic):
    C = 1
    c = 2

    dt, _, m, v = params

    t += dt / 2

    # dt/2 D
    p = p * np.exp(- g(t) * dt / 2)


    # dt/2 B
    p = p * np.exp(g(t) * dt/2)
    x = x * np.exp(-g(t) * dt/2)

    # dt/2 C
    p = - gradf(x) * dt/2 + p
    # dt A
    if kinetic == 'Quartic':
        x = g(t) * p ** (1 / 3) * dt + x
    elif kinetic == 'Logaritmic':
        x = -g(t) * dt/p + x
    elif kinetic == 'Quadratic':
        x = g(t) * p * dt + x
    else: # Relativistic
        sq = np.sqrt(v ** 2 * m ** 2 + np.sum(p**2))
        x = (g(t) * v * p / sq) * dt + x

    # dt/2 C
    p = - gradf(x) * dt/2 + p


    # dt/2 B
    p = p * np.exp(g(t) * dt/2)
    x = x * np.exp(-g(t) * dt/2)

    # dt/2 D
    p = p * np.exp(- g(t) * dt / 2)
    # dt/2 t
    t += dt / 2

    return (p, x, t)

def gradh(x, params):
    dt, _, m, v = params
    return m * x / np.sqrt(1 - np.dot(x, x) / v ** 2)


def Breg(params, xinit, gradf, f,  steps, kinetic, adap, new, tol):
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
        #t = dt * i
        #tnew = dt * i
        #t=dt * i
        if adap:
            if new:
                pnew, xnew, tnew = step_Br_adap_new(p0, x0, t0, params, gradf, kinetic)
            else:
                pnew, xnew, tnew = step_Br_adap(p0, x0, t0, params, gradf, kinetic)
        else:
            if new:
                pnew, xnew, tnew = step_Br_new(p0, x0, t0, params, gradf, kinetic)
            else:
                pnew, xnew, tnew = step_Br(p0, x0, t0, params, gradf, kinetic)

        #if abs(tnew - t - dt) > ttol:
        #    warnings.warn(f"tnew-t-dt, dt inconsistency: {tnew - t - dt}, {dt}")
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









