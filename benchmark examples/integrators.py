import numpy as np
from scipy import linalg as lg
import warnings


def HBr(example, q, p, q0_t, c):
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


def HTVI_d(example, c, dt, steps, init):
    x0, x0_t, p0, p0_t = init
    C = 1
    q0 = x0
    q0_t =x0_t
    #q0 = example.x0
    #q0_t = 0.01#np.random.rand(q0.shape[0])
    #p0 = np.zeros(q0.shape)
    #p0_t = -HBr(example, q0, p0, q0_t, order) #np.zeros(q0.shape)
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


def HTVI_adap(example, c, dt, steps, init):
    C = 1
    #c = 2
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


def Bet_dir(example, c, dt, steps, init):
    x0, p0, p0_t = init
    C = 1
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

def eg(t, c, C):
    return t ** c

def eg1(t, c, C):
    return 1/eg(t, c, C)


def Hx(t, x, p, gradf, params, c, C):
    v, m = params
    n = x.shape[0]
    sq = np.sqrt(1 - np.sum(x ** 2) / v ** 2)

    diagA = m / sq + m * x ** 2 / (sq ** 3 * v ** 2)  # Aii
    Aij = m * np.prod(x) / (sq ** 3 * v ** 2)           # Aij

    A = Aij * np.ones((n, n))
    np.fill_diagonal(A, diagA)

    arg1 = eg1(t, c, C) * p + m * x / sq  # vector
    arg2 = np.sqrt(np.sum((eg1(t, c, C) * p + m * x / sq) ** 2) + v ** 2 * m ** 2)
    arg3 = v ** 3 * m * x / np.sqrt(v ** 2 - np.sum(x ** 2))**3
    arg4 = - eg1(t, c, C) * p + eb(t, c, C) * gradf(x)
    return (v / arg2 * A @ arg1 - arg3 + arg4) * ea(c, t) * eg(t, c, C)


def Hp(t, x, p, params, c, C):
    v, m = params
    sq = np.sqrt(1 - np.sum(x ** 2) / v ** 2)
    arg1 = eg1(t, c, C) * p + m * x / sq
    arg2 = np.sum(arg1 ** 2) + v ** 2 * m ** 2

    return ea(c, t) * (v * arg1 / np.sqrt(arg2) - x)


def step_FJ(example, dt, p, p_, x, x_, t, t_, params):
    v, m, c, C, e = params
    #C = 1
    # dt/2 A
    p = p - dt/2 * Hx(t, x, p_, example.gradf, [v, m], c, C)
    t_ = t_ + dt/2
    x_ = x_ + dt/2 * Hp(t, x, p_, [v, m], c, C)

    # dt/2 B
    t = t + dt/2
    x = x + dt/2 * Hp(t_, x_, p, [v, m], c, C)
    p_ = p_ - dt/2 * Hx(t_, x_, p, example.gradf, [v, m], c, C)

    # dt C
    x = 0.5 * (x + x_ + np.cos(2 * e * dt) * (x - x_) + np.sin(2 * e * dt) * (p - p_))
    p = 0.5 * (p + p_ - np.sin(2 * e * dt) * (x - x_) + np.cos(2 * e * dt) * (p - p_))
    x_ = 0.5 * (x + x_ - np.cos(2 * e * dt) * (x - x_) - np.sin(2 * e * dt) * (p - p_))
    p_ = 0.5 * (p + p_ + np.sin(2 * e * dt) * (x - x_) - np.cos(2 * e * dt) * (p - p_))

    # dt/2 B
    t = t + dt / 2
    x = x + dt / 2 * Hp(t_, x_, p, [v,m], c, C)
    p_ = p_ - dt / 2 * Hx(t_, x_, p, example.gradf, [v,m], c, C)

    # dt/2 A
    p = p - dt / 2 * Hx(t, x, p_, example.gradf, [v,m], c, C)
    t_ = t_ + dt / 2
    x_ = x_ + dt/2 * Hp(t, x, p_, [v,m], c, C)
    return x, x_, p, p_, t, t_


def FJ(example, dt, params, steps, init):
    x0, x0_, p0, p0_, t0, t0_ = init
    ttol = 1e-13
    tfinal = steps * dt

    tspan = np.linspace(t0, tfinal, steps)
    tspan_ = np.linspace(t0_, tfinal, steps)

    solp = np.empty([steps, *np.shape(p0)], dtype=np.float64)
    solx = np.empty([steps, *np.shape(x0)], dtype=np.float64)

    solp_ = np.empty([steps, *np.shape(p0_)], dtype=np.float64)
    solx_ = np.empty([steps, *np.shape(x0_)], dtype=np.float64)
    solp[0] = p0
    solx[0] = x0
    solp_[0] = p0_
    solx_[0] = x0_

    for i in range(steps - 1):
        p = np.copy(solp[i])
        x = np.copy(solx[i])
        p_ = np.copy(solp[i])
        x_ = np.copy(solx[i])
        t = tspan[i]
        t_ = tspan_[i]

        xnew, xnew_, pnew, pnew_, tnew, tnew_ = step_FJ(example, dt, p, p_, x, x_, t, t_, params)
        if abs(tnew - t - dt) > ttol:
            warnings.warn(f"tnew-t-dt, dt inconsistency: {tnew - t - dt}, {dt}")
        solp[i + 1] = pnew
        solx[i + 1] = xnew
        solp_[i + 1] = pnew_
        solx_[i + 1] = xnew_
    return solx



def step_Br(example, dt, p, x, t, params, kinetic):
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

    # dt A
    if kinetic == 'Quadratic':
        x = ea(c, t) * p * dt + x

    else: # Relativistic
        sq = np.sqrt(v ** 2 * m ** 2 + np.sum(p ** 2))
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


def step_Br_adap(example, dt, p, x, t, params, kinetic):
    v, m, c, C = params
    # dt/2 t
    # T = eb(t, 2, C)  # np.exp(C) * t ** c
    # arg = T + dt/2
    # t = (np.exp(-C)*arg)**(1/2)

    T = eb(t, c, C)  # np.exp(C) * t ** c
    arg = T + dt/2
    t = (np.exp(-C)*arg)**(1/c)

    # dt/2 D
    p = p * np.exp(-ea(c, t) * dt / 2) #c/t

    # dt/2 B
    p = p * np.exp(ea(c, t) * dt/2)
    x = x * np.exp(-ea(c, t) * dt/2)

    # dt/2 C
    p = - ea(c, t) * eb(t, c, C) * example.gradf(x) * dt/2 + p

    if kinetic == 'Quadratic':
        x = ea(c, t) * p * dt + x

    else:  # Relativistic
        sq = np.sqrt(v ** 2 * m ** 2 + np.sum(p ** 2))
        x = (ea(c, t) * v * p / sq) * dt + x


    # dt/2 C
    p = - ea(c, t) * eb(t, c, C) * example.gradf(x) * dt/2 + p

    # dt/2 B
    p = p * np.exp(ea(c, t) * dt/2)
    x = x * np.exp(-ea(c, t) * dt/2)

    # dt/2 D
    p = p * np.exp(-ea(c, t) * dt / 2)

    # dt/2 t
    # T = eb(t, 2, C)
    # arg = T + dt/2
    # t = (np.exp(-C) * arg) ** (1 / 2)

    T = eb(t, c, C)
    arg = T + dt/2
    t = (np.exp(-C) * arg) ** (1 / c)

    return (p, x,t)


def step_Br_adap_new(example, dt, p, x, t, params, kinetic):
    v, m, c, C = params
    t += dt / 2

    # dt/2 D
    p = p * np.exp(- 1/t * dt / 2)

    # dt/2 B
    p = p * np.exp(1/t * dt/2)
    x = x * np.exp(-1/t * dt/2)

    # dt/2 C
    p = - example.gradf(x) * dt/2 + p

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
    p = - example.gradf(x) * dt/2 + p

    # dt/2 B
    p = p * np.exp(1 / t * dt / 2)
    x = x * np.exp(-1 / t * dt / 2)

    # dt/2 D
    p = p * np.exp(- 1/t * dt / 2)

    # dt/2 t
    t += dt / 2

    return (p, x, t)



def gradh(x, params):
    params = params[:2]
    v, m = params
    return m * x / np.sqrt(1 - np.dot(x, x) / v ** 2)


def Breg(example, params, dt, steps, init, adap, new, kinetic):
    #parms = v,m,...
    x0, p0 = init
    tfinal = steps * dt
    t0 = 1e-3
    tspan = np.linspace(t0, tfinal, steps)

    #p0 = gradh(x0, params)
    p0 = example.p0
    solp = np.empty([steps, *np.shape(p0)], dtype=np.float64)
    solx = np.empty([steps, *np.shape(x0)], dtype=np.float64)

    solp[0] = p0
    solx[0] = x0

    for i in range(steps - 1):
        p = np.copy(solp[i])
        x = np.copy(solx[i])
        t = tspan[i]
        if adap:
            if new:
                pnew, xnew, tnew = step_Br_adap_new(example, dt, p, x, t, params, kinetic)
            else:
                pnew, xnew, tnew = step_Br_adap(example, dt, p, x, t, params, kinetic)
        else:
            pnew, xnew, tnew = step_Br(example, dt, p, x, t, params, kinetic)
        solp[i + 1] = pnew
        solx[i + 1] = xnew

    return solx, solp


# C=1;c=2
# tinit=0.001
# dt=1e-5

# for i in range(2000):
#     T = np.exp(C) * tinit ** c
#     arg = T + dt/2
#     t = (np.exp(-C)*arg)**(1/c)
#
#     T = np.exp(C) * t ** c
#     arg = T + dt / 2
#     tend = (np.exp(-C) * arg) ** (1 / c)
#     print(tend-tinit)
#     tinit=tend
#     #print(t)


# for i in range(200):
#     T = np.exp(tinit*c)
#     arg = T + dt/2
#     t = np.log(arg)/c
#
#     T = np.exp(t*c)
#     arg = T + dt/2
#     tend = np.log(arg)/c
#     print(tend-tinit)
#     tinit=np.copy(tend)




