import numpy as np
from sklearn.model_selection import train_test_split
from integrators import CM, NAG
from example_breast_cancer_LR import gradLp, Lp, loss_func_test, n, Xtrain, Ytrain, Xvalid, Yvalid
from integrators_Breg import Breg


def init_sample(num_trials):
    np.random.seed(18)
    init = np.random.rand(num_trials, n + 1)
    return init


def tuning_process(num_trials, max_ite, N, method):
    min_f = 1e+16
    mu_new = 0
    dt_new = 0
    m_new = 0
    vc_new = 0

    # L = lambda w: Lp(w, Xvalid, Yvalid)
    # gradL = lambda w: gradLp(w, Xvalid, Yvalid)

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

            Mf = np.zeros(ngrid)
            Vcf = np.zeros(ngrid)

        elif method ==NAG:
            Dt = np.linspace(1e-4, 1e-1, N)
            Mu = np.linspace(0.6, 0.99, N)

            Dtt, Muu = np.meshgrid(Dt, Mu)
            Dtf = Dtt.flatten()
            Muf = Muu.flatten()
            ngrid = len(Dtf)

            Mf = np.zeros(ngrid)
            Vcf = np.zeros(ngrid)

        else:
            Dt = np.linspace(1e-4, 1e-2, N)
            Mu = np.linspace(0.8, 0.99, N)
            M = np.linspace(1e-6, 1e-5, N)
            Vc = np.linspace(1e+5, 1e+6, N)


            Dtt, Muu, Mm, Vcc = np.meshgrid(Dt, Mu, M, Vc)
            Dtf = Dtt.flatten()
            Muf = Muu.flatten()
            Mf = Mm.flatten()
            Vcf = Vcc.flatten()
            ngrid= len(Dtf)

        for i in range(ngrid):
            #print(Dtf[i], Muf[i])

            params = [Dtf[i], Muf[i], Mf[i], Vcf[i]]
            _, loss = method(params, init0, gradL, L, max_ite, tol=1e-10)
            Loss = loss[-1] #[steps]

            min_fnew = Loss
            if min_fnew < min_f:
                 min_f = min_fnew
                 mu_new = Muf[i]
                 dt_new = Dtf[i]
                 m_new = Mf[i]
                 vc_new = Vcf[i]

    return mu_new, dt_new, m_new, vc_new

#a =tuning_process(num_trials=5, max_ite=50, N=5, method=CM)
#b =tuning_process(num_trials=5, max_ite=50, N=5, method=NAG)
#c = tuning_process(num_trials=5, max_ite=50, N=5, method=CRGD)
#d = tuning_process(num_trials=2, max_ite=50, N=5, method=RGD)
e =tuning_process(num_trials=5, max_ite=500, N=5, method=CNAG)



