import numpy as np
from scipy import linalg as lg
import warnings

def HBr(example, q,p,q0_t,c):
    C = 1
    return c/(2*q0_t**(c+1)) * np.dot(p, p) + C * c * q0_t ** (2*c-1) * example.f(q)


def CM(example, mu, dt, steps):
    """
    Compute the minimum for the function f in class [example] using Classical momentum (CM) algorithm.
    mu is the momentum factor; dt is the step; steps is the number of step.
    class for [example]; initial conditions [p0], [q0]; gradient for f [gradf].
     """
    x0 = example.x0
    p0 = np.zeros(x0.shape)
    X = []
    X.append(x0)
    for t in range(steps-1):
        g = example.gradf(x0)
        p = mu * p0 - dt*g
        x = x0 + p
        X.append(x)
        p0 = p
        x0 = x
    return np.array(X)


def NAG(example, mu, dt, steps):
    """
    Compute the minimum for the function f in class [example] using Nesterov's accelerated gradient (NAG) algorithm.
    mu is the momentum factor; dt is the step; steps is the number of step.
    class for [example]; initial conditions [p0], [q0]; gradient for f [gradf].
    """
    x0 = example.x0
    v0 = np.zeros(x0.shape)
    X = []
    X.append(x0)
    for t in range(steps-1):
        g = example.gradf(x0 + mu*v0)
        v = mu * v0 - dt*g
        x = x0 + v
        X.append(x)
        v0 = v
        x0 = x
    return np.array(X)


def HTVI_d(example, order, dt, steps, init):
    x0, x0_t, p0, p0_t = init
    C = 1
    c = order
    q0 = x0
    q0_t = x0_t
    X = []
    X.append(q0)
    for t in range(steps-1):
        p1 = p0 - c * dt * C * (q0_t) ** (2*c-1) * example.gradf(q0)
        p1_t = p0_t + dt * c * (c+1)/(2 * (q0_t) ** (c+2)) * np.dot(p1, p1) \
               - dt * C * c * (2*c-1) * (q0_t) ** (2*c-2) * example.f(q0)
        q1 = q0 + dt * c / (q0_t) ** (c+1) * p1
        q1_t = q0_t + dt

        p0 = np.copy(p1)
        p0_t = np.copy(p1_t)
        q0 = np.copy(q1)
        q0_t = np.copy(q1_t)
        X.append(q0)
    return np.array(X)


def HTVI_adap(example, order, dt, steps, init):
    C = 1
    c = order
    cdot = 0.5
    x0, x0_t, p0, p0_t = init
    q0 = x0
    q0_t = x0_t
    X = []
    X.append(q0)
    for t in range(steps-1):
        p1 = p0 - (c**2/cdot) * dt * C * (q0_t) ** (2*c - cdot/c) * example.gradf(q0)
        p1_t2 = (c**3 + c * cdot) / (2 * cdot * (q0_t)**(c + cdot / c + 1)) * dt * np.dot(p1, p1) \
                + (-2*c**3 + c * cdot) / (cdot * (q0_t)**(cdot / c + 1 - 2 * c)) * dt * C * example.f(q0)
        p1_t = (1 - dt * (q0_t) ** (-cdot/c) * (1 - c / cdot))**(-1) * (p0_t + p1_t2)
        q1 = q0 + dt * c**2/cdot * (q0_t) ** (-c - cdot / c) * p1

        q1_t = q0_t + (c / cdot) * dt * (q0_t) ** (1-cdot/c)

        p0 = np.copy(p1)
        p0_t = np.copy(p1_t)
        q0 = np.copy(q1)
        q0_t = np.copy(q1_t)
        X.append(q0)
    return np.array(X)


def HTVI_Bet_d(example, order, dt, steps, init):
    x0, p0, p0_t = init
    C = 1
    c = order
    q0 = x0
    X = []
    X.append(q0)
    tt = 0.01
    for t in range(steps-1):
        tt = tt + dt / 2
        p1_t = p0_t + dt/2 * c * (c + 1) / (2 * tt ** (c+2)) * np.dot(p0, p0) \
               - dt/2 * C * c * (2 * c - 1) * tt ** (2*c-2) * example.f(q0)
        p = p0 - dt/2 * C * c * tt ** (2 * c - 1) * example.gradf(q0)

        q = q0 + dt * c / tt ** (c + 1) * p
        p = p - dt / 2 * C * c * tt ** (2 * c - 1) * example.gradf(q)

        p1_t = p1_t + dt / 2 * c * (c + 1) / (2 * tt ** (c + 2)) * np.dot(p, p) \
               - dt / 2 * C * c * (2 * c - 1) * tt ** (2 * c - 2) * example.f(q)
        tt = tt + dt / 2

        p0 = np.copy(p)
        p0_t = np.copy(p1_t)
        q0 = np.copy(q)

        X.append(q0)
    return np.array(X)


def ea(c, t):
    return c/t


def eb(t, c, C):
    return np.exp(C) * t ** c


def step_Br(example, dt, p, x, t, params):

    v, m, c, C = params
    # dt/2 t
    t += dt / 2

    # dt/2 D
    p = p * np.exp(-ea(c, t) * dt / 2)

    # dt/2 B
    p = p * np.exp(ea(c, t) * dt/2)
    x = x * np.exp(-ea(c, t) * dt/2)

    # dt/2 C
    p = - ea(c, t) * eb(t, c, C) * example.gradf(x) * dt/2 + p

    # dt A # Relativistic
    sq = np.sqrt(v ** 2 * m ** 2 + np.sum(p**2))
    x = (ea(c, t) * v * p / sq) * dt + x

    # dt/2 C
    p = - ea(c, t) * eb(t, c, C) * example.gradf(x) * dt/2 + p

    # dt/2 B
    p = p * np.exp(ea(c, t) * dt/2)
    x = x * np.exp(-ea(c, t) * dt/2)

    # dt/2 D
    p = p * np.exp(-ea(c, t) * dt / 2)

    # dt/2 t
    t += dt / 2
    return (p, x, t)


def step_Br_adap(example, dt, p, x, t, params):
    v, m, c, C = params

    t += dt / 2

    # dt/2 D
    p = p * np.exp(- 1/t * dt / 2)

    # dt/2 B
    p = p * np.exp(1/t * dt/2)
    x = x * np.exp(-1/t * dt/2)

    # dt/2 C
    p = - example.gradf(x) * dt/2 + p

    # dt A # Relativistic
    sq = np.sqrt(v ** 2 * m ** 2 + np.sum(p**2))
    x = (1/t * v * p / sq) * dt + x

    # dt/2 C
    p = - example.gradf(x) * dt/2 + p

    # dt/2 B
    p = p * np.exp(1/t * dt/2)
    x = x * np.exp(-1/t * dt/2)

    # dt/2 D
    p = p * np.exp(- 1/t * dt / 2)

    # dt/2 t
    t += dt / 2

    return (p, x, t)

def step_Br_adap_old(example, dt, p, x, t, params):
    v, m, c, C = params
    # dt/2 t
    T = eb(t, c, C)
    arg = T + dt/2
    t = (np.exp(-C)*arg)**(1/c)
    #t = (C*arg)**(1/c)
    # dt/2 D
    p = p * np.exp(-ea(c, t) * dt / 2)
    # dt/2 B
    p = p * np.exp(ea(c, t) * dt/2)
    x = x * np.exp(-ea(c, t) * dt/2)

    # dt/2 C
    p = - ea(c, t) * eb(t, c, C) * example.gradf(x) * dt/2 + p


    # dt A # Relativistic
    sq = np.sqrt(v ** 2 * m ** 2 + np.sum(p**2))
    x = (ea(c, t) * v * p / sq) * dt + x

    # dt/2 C
    p = - ea(c, t) * eb(t, c, C) * example.gradf(x) * dt/2 + p

    # dt/2 B
    p = p * np.exp(ea(c, t) * dt/2)
    x = x * np.exp(-ea(c, t) * dt/2)

    # dt/2 D
    p = p * np.exp(-ea(c, t) * dt / 2)

    # dt/2 t
    T = eb(t, c, C)
    arg = T + dt/2
    #t = (C*arg)**(1/c)
    t = (np.exp(-C) * arg) ** (1 / c)

    return (p, x,t)


def g(t, g0):
    return g0 / (g0 - (g0-1) * np.exp(-t))

def step_Br_g(example, dt, p, x, t, params):
    v, m, g0 = params
    t += dt / 2

    # dt/2 D
    p = p * np.exp(- g(t, g0) * dt / 2)

    # dt/2 B
    p = p * np.exp(g(t, g0) * dt/2)
    x = x * np.exp(-g(t, g0) * dt/2)

    # dt/2 C
    p = - example.gradf(x) * dt/2 + p

    # dt A
    # Relativistic
    sq = np.sqrt(v ** 2 * m ** 2 + np.sum(p**2))
    x = (g(t, g0) * v * p / sq) * dt + x

    # dt/2 C
    p = - example.gradf(x) * dt/2 + p

    # dt/2 B
    p = p * np.exp(g(t, g0) * dt/2)
    x = x * np.exp(-g(t, g0) * dt/2)

    # dt/2 D
    p = p * np.exp(- g(t, g0) * dt / 2)

    # dt/2 t
    t += dt / 2

    return (p, x, t)


def gradh(x, params):
    params=params[:2]
    v, m = params
    return m * x / np.sqrt(1 - np.dot(x, x) / v ** 2)


def Breg(example, params, dt, steps, init, adap, which):
    #parms = v,m,...
    x0, p0 = init
    ttol = 1e-13
    tfinal = steps * dt
    #t0 = 1e-5
    t0=1e-4

    if adap and (which =='new'):
        v, m, c, C = params
        t0 = t0 ** c * np.exp(C)
    tspan = np.linspace(t0, tfinal, steps)

    p0 = gradh(x0, params)
    #p0 = example.p0
    solp = np.empty([steps, *np.shape(p0)], dtype=np.float64)
    solx = np.empty([steps, *np.shape(x0)], dtype=np.float64)

    solp[0] = p0
    solx[0] = x0

    for i in range(steps - 1):
        p = np.copy(solp[i])
        x = np.copy(solx[i])
        t = tspan[i]
        if adap:
            if which=='new':
                pnew, xnew, tnew = step_Br_adap(example, dt, p, x, t, params)
            else:
                pnew, xnew, tnew = step_Br_adap_old(example, dt, p, x, t, params)
        else:
            if which=='new':
                pnew, xnew, tnew = step_Br_g(example, dt, p, x, t, params)
            else:
                pnew, xnew, tnew = step_Br(example, dt, p, x, t, params)
        solp[i + 1] = pnew
        solx[i + 1] = xnew

    return solx, solp







