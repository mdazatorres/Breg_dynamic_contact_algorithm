import numpy as np
from scipy.stats import ortho_group

#Breg_dynamic_contact_transf

class Ex_Quadratic:
    def __init__(self, n):
        self.n = n
        np.random.seed(0)
        self.Q = ortho_group.rvs(dim=self.n)
        self.S = np.diag(np.random.uniform(10e-3, 1, self.n))
        self.A = np.dot(np.dot(self.Q, self.S), self.Q.T)

        self.p0 = np.zeros(self.n)
        self.x0 = np.ones(self.n)
        self.x0_t = 0.01
        self.s0 = 0
        self.y0 = np.random.rand(self.n)
        self.z0 = np.random.rand(self.n)

        self.x0_ = 0.5*np.ones(self.n)
        self.p0_ = np.random.rand(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02


    def f(self, x):
        return 0.5 * np.dot(np.dot(x, self.A), x)

    def gradf(self, x):
        return np.dot(x, self.A)

    def f2D(self, x1, x2):
        return 0.5*(self.A[0][0] * x1 ** 2 + self.A[1][0] * x1 * x2 + self.A[0][1] * x1 * x2 + self.A[1][1] * x2 ** 2)


class Ex_Corr_Quadratic:  # Correlated quadratic function
    def __init__(self, n):
        self.n = n
        #self.seed = seed
        np.random.seed(0)

        self.B = [[np.sqrt((i + 1) * (j + 1)) / 2 ** (abs(i - j)) for i in range(self.n)] for j in range(self.n)]
        self.p0 = np.zeros(self.n)
        self.x0 = np.random.uniform(-1, 1, self.n)
        self.s0 = 0
        self.x0_t = 0.01

        self.x0_ = np.random.uniform(-1, 1, self.n)
        self.p0_ = np.random.rand(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02

    def f(self, x):
        return 0.5 * np.dot(np.dot(x, self.B), x)

    def gradf(self, x):
        return np.dot(x, self.B)

    def Hessf(self,x):
        return self.B

    def f2D(self, x1, x2):
        return 0.5*(self.B[0][0] * x1 ** 2 + self.B[1][0] * x1 * x2 + self.B[0][1] * x1 * x2 + self.B[1][1] * x2 ** 2)




class Ex_Quartic_1: # Quartic function 1
    def __init__(self, n):
        self.n = n
        self.p0 = np.zeros(self.n)
        np.random.seed(0)
        self.x0 = np.random.rand(self.n)
        self.s0 = 0
        self.x0_t = 0.01

        self.p0_ = np.random.rand(self.n)
        self.x0_ = np.random.rand(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02

        xc = np.linspace(1, n, n)
        xv, yv = np.meshgrid(xc, xc)
        self.sigma = 0.9 ** (abs(xv-yv))
    def f(self, x):
        return (((x-1) @ self.sigma) @ (x-1))**2

    def gradf(self, x):
        prod = ((x-1) @ self.sigma) @ (x-1)
        return 4 * prod * ((x-1) @ self.sigma)

    def f2D(self, x1, x2):
        xc = np.linspace(1, 2, 2)
        xv, yv = np.meshgrid(xc, xc)
        sigma = 0.9 ** (abs(xv-yv))
        return (sigma[0][0]*(x1-1)**2 + sigma[0][1]*(x1-1)*(x2-1) + sigma[1][1]*(x2-1)**2)**2


class Ex_Quartic_2:  # Quartic function
    def __init__(self, n=50):
        self.n = n
        self.p0 = np.zeros(self.n)
        self.x0 = 2 * np.ones(self.n)
        self.x0_t = 0.01
        self.s0 = 0
        self.t0 = 0

        self.p0_ = np.random.rand(self.n)
        self.x0_ = np.ones(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02

    def f(self, x):
        seq = np.arange(1, self.n + 1)
        return np.dot(seq, x ** 4)

    def gradf(self, x):
        seq = np.arange(1, self.n + 1)
        gradV = 4 * seq * x ** 3
        return gradV

    def f2D(self, x1, x2):
        return x1 ** 4 + 2 * x2 ** 4


class Ex_Booth:
    def __init__(self,  n):
        self.n = n
        self.p0 = np.array([0, 0])
        self.x0 = np.array([10, 10])
        self.s0 = 0
        self.x0_t = 0.01

        self.p0_ = np.random.rand(self.n)
        self.x0_ = np.random.rand(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02

    def f(self, x):
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

    def gradf(self, x):
        return np.array([10 * x[0] + 8 * x[1] - 34, 8 * x[0] + 10 * x[1] - 38])

    def f2D(self, x1, x2):
        return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


class Ex_Schwefel:# Schwefel function:
    def __init__(self, n=20):
        self.n = n
        self.p0 = np.zeros(self.n)
        self.x0 = 2 * np.ones(self.n)
        self.x0_t = 0.01
        self.s0 = 0
        self.t0 = 0

        self.p0_ = np.random.rand(self.n)
        self.x0_ = np.random.rand(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02

    def f(self, x):
        return np.sum(x ** 10)

    def gradf(self, x):
        gradV = 10 * (x ** 9)
        return gradV

    def f2D(self, x1, x2):
        return x1 ** 10 + x2 ** 10

class Ex_Matyas:  # Matyas function
    def __init__(self):
        self.n = 2
        # self.seed = seed
        # x0= np.array([10, -7])
        self.p0 = np.zeros(self.n)
        self.x0 = np.array([10, -7])
        self.x0_t = 0.01
        self.s0 = 0
        self.t0 = 0

        self.p0_ = np.random.rand(self.n)
        self.x0_ = np.random.rand(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02

    def f(self, x):
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

    def gradf(self, x):
        return np.array([2*0.26*x[0] - 0.48*x[1], 2*0.26*x[1]-0.48*x[0]])

    def f2D(self, x1, x2):
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


class Ex_Beale:  # Beale function
    def __init__(self):
        self.n = 2
        # self.seed = seed
        # x0= np.array([10, -7])
        self.p0 = np.zeros(self.n)
        self.x0 = np.array([-3, -3])
        self.x0_t = 0.01
        self.s0 = 0
        self.t0 = 0

        self.p0_ = np.random.rand(self.n)
        self.x0_ = np.random.rand(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02

    def f(self, x):
        return (1.5-x[0] + x[0]*x[1])**2 + (2.25-x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2


    def gradf(self, x):
        X = np.copy(x)
        y = X[1]
        x = X[0]
        gradx= 2*x*y**6 + 2*x*y**4 + (5.250 - 4 * x) * y**3 + (-2*x + 4.50)*y**2 + (3- 4* x) * y - 12.750 + 6* x
        grady= (-2 - 2*y + 4*y**3 - 6*y**2 + 6*y**5)*x**2 + (3 + 9*y + 15.750*y**2)*x
        return np.array([gradx, grady])

    def f2D(self, x1, x2):
        return (1.5-x1 + x1*x2)**2 + (2.25-x1 + x1 * x2**2)**2 + (2.625 - x1 + x1*x2**3)**2

class Ex_Chung_Reynolds:  # Chung.Reynolds funcion
    def __init__(self, n=50):
        self.n = n
        self.p0 = np.zeros(self.n)
        self.x0 = 50 * np.ones(self.n)
        self.x0_t = 0.01
        self.s0 = 0
        self.t0 = 0

        self.p0_ = np.random.rand(self.n)
        self.x0_ = np.random.rand(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02

    def f(self, x):
        return np.sum(x ** 2)**2

    def gradf(self, x):
        gradV = 4 * np.sum(x ** 2) * x
        return gradV

    def f2D(self, x1, x2):
        return (x1 ** 2 +  x2 ** 2)**2


class Ex_Zakharov:  # Zakharov
    def __init__(self, n=5):
        self.n = n
        self.p0 = np.zeros(self.n)
        self.x0 = np.ones(self.n)
        self.x0_t = 0.01
        self.s0 = 0
        self.t0 = 0

        self.p0_ = np.random.rand(self.n)
        self.x0_ = np.random.rand(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02

    def f(self, x):
        seq = np.arange(1, self.n + 1)
        return np.sum(x ** 2) + (0.5 * np.dot(seq, x))**2 + (0.5 * np.dot(seq, x))**4

    def gradf(self, x):
        seq = np.arange(1, self.n + 1)
        gradV = 2 * x + 2 * (0.5 * np.dot(seq, x)) * seq + 4 * (0.5 * np.dot(seq, x)) ** 3 * seq
        return gradV

    def f2D(self, x1, x2):
        return (x1 ** 2 + x2 ** 2) + (0.5*(x1 ** 2 + 2*x2 ** 2))**2 + (0.5*(x1 ** 2 + 2*x2 ** 2))**4


class Ex_Three_hump:  # Three-hump
    def __init__(self):
        self.n = 2
        # self.seed = seed
        # x0= np.array([10, -7])
        self.p0 = np.zeros(self.n)
        self.x0 = np.array([5, 5])
        self.x0_t = 0.01
        self.s0 = 0
        self.t0 = 0

        self.p0_ = np.random.rand(self.n)
        self.x0_ = np.random.rand(self.n)
        self.t0 = 0.01
        self.t0_ = 0.02

    def f(self, x):
        return 2* x[0]**2 - 1.05 * x[0]**4 + x[0]**6 / 6 + x[0] * x[1] + x[1]**2


    def gradf(self, x):
        X = np.copy(x)
        y = X[1]
        x = X[0]
        gradx= 4*x - 4.20*x**3 + x**5 + y
        grady= x + 2*y
        return np.array([gradx, grady])

    def f2D(self, x1, x2):
        return 2 * x1**2 - 1.05 * x1**4 + x1**6 / 6 + x1 * x2+ x1**2

