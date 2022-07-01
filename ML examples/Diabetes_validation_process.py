import numpy as np
from sklearn.model_selection import train_test_split
from integrators_Breg import Breg
from integrators import CM, NAG
from Diabetes_LR_ex import gradLp, Lp, loss_func_test, n, Xvalid, Yvalid


def init_sample(num_trials):
    np.random.seed(18)
    init = np.random.rand(num_trials, n + 1)
    return init


def tuning_process(num_trials, max_ite, N, method, kinetic):
    min_f = 1e+16
    mu_new = 0
    dt_new = 0
    m_new = 0
    vc_new = 0
    L = lambda w: Lp(w, Xvalid, Yvalid)
    gradL = lambda w: gradLp(w, Xvalid, Yvalid)

    theta = init_sample(num_trials)
    for k in range(num_trials):
        init0 = theta[k]
        if method ==CM:
            Dt = np.linspace(1e-4, 1e-1, N)
            Mu = np.linspace(0.6, 0.99, N)

            Dtt, Muu = np.meshgrid(Dt, Mu)
            Dtf = Dtt.flatten()
            Muf = Muu.flatten()
            ngrid =len(Dtf)

        elif method ==NAG:
            Dt = np.linspace(1e-4, 1e-1, N)
            Mu = np.linspace(0.6, 0.99, N)

            Dtt, Muu = np.meshgrid(Dt, Mu)
            Dtf = Dtt.flatten()
            Muf = Muu.flatten()
            ngrid = len(Dtf)

        else:
            Dt = np.linspace(1e-4, 1e-1, N)
            Dtf = Dt
            ngrid= len(Dtf)
            Muf = np.zeros(ngrid)
        m = 0.01  # m
        vc = 1000  # v

        for i in range(ngrid):
            #print(Dtf[i], Muf[i])
            params = [Dtf[i], Muf[i], m, vc]
            if method==Breg:
                _, loss = method(params, init0, gradL, L, max_ite, kinetic=kinetic, adap=False, tol=1e-10)
            else:
                _, loss = method(params, init0, gradL, L, max_ite, tol=1e-10)
            Loss = loss[-1] #[steps]

            min_fnew = Loss
            if min_fnew < min_f:
                 min_f = min_fnew
                 mu_new = Muf[i]
                 dt_new = Dtf[i]

    return mu_new, dt_new, m_new, vc_new

a =tuning_process(num_trials=5, max_ite=200, N=5, method=CM, kinetic='None' )
b =tuning_process(num_trials=5, max_ite=200, N=5, method=NAG, kinetic='None' )
d =tuning_process(num_trials=5, max_ite=200, N=5, method=Breg, kinetic='Quadratic' )
e =tuning_process(num_trials=5, max_ite=200, N=5, method=Breg, kinetic='Relativistic' )



