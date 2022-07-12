import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler
from integrators import CM, NAG
import pandas as pd
from sklearn.preprocessing import scale

pima = pd.read_csv("data/datasets_228_482_diabetes.csv")
pima.head()

array = pima.values


X = array[1:, 0:-1]
#X = scale(X, with_std=0.5)
y = array[1:, -1]

def data_proccesing(test_size=0.25, valid_size=0.2, scale=True):
    xx_train, Xtest, yy_train, Ytest = train_test_split(X, y, test_size=test_size, random_state=0)
    Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(xx_train, yy_train, test_size=valid_size, random_state=0)

    sc = StandardScaler()
    if scale:
        Xtrain = sc.fit_transform(Xtrain)
        Xtest = sc.fit_transform(Xtest)
        Xvalid = sc.fit_transform(Xvalid)

    m_train, n = Xtrain.shape
    m_test, _ = Xtest.shape
    m_valid, _ = Xvalid.shape


    # n+1 variables
    # for our example, we added an ones columns to the data
    x1 = np.ones((m_train, 1))
    x2 = np.ones((m_test, 1))
    x3 = np.ones((m_valid, 1))

    X_train = np.hstack((x1, Xtrain))
    X_test = np.hstack((x2, Xtest))
    X_valid = np.hstack((x3, Xvalid))
    return X_train, X_test, X_valid, Ytrain, Yvalid, Ytest, n

Xtrain, Xtest, Xvalid, Ytrain, Yvalid, Ytest, n =data_proccesing()
alpha=1e-1
def sigma(w, x):
    z = np.dot(x, w)
    return 1/(1 + np.exp(-z))

def Lp(w, x, y):
    yhat = sigma(w, x)
    m = x.shape[0]
    return -1/m * np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) + alpha * np.sum(w[1:] ** 2)/(2*m)


def gradLp(w, x, y):
    yhat = sigma(w, x)
    what = np.copy(w)
    m = x.shape[0]
    what[0] = 0
    return 1/m * np.dot((yhat-y), x) + alpha * what/m



def loss_func_test(w):
    Etest = []
    Etrain = []
    Ltest = []
    for i in range(len(w)):
        Ltest.append(Lp(w[i], Xtest, Ytest))
        ypred_test = np.zeros(Xtest.shape[0])
        yprobt = sigma(w[i], Xtest)
        ypred_test[yprobt > 0.5] = 1

        ypred_train = np.zeros(Xtrain.shape[0])
        yprob = sigma(w[i], Xtrain)
        ypred_train[yprob > 0.5] = 1

        e_test = 1 - np.sum(Ytest == ypred_test) / len(ypred_test)
        e_train = 1 - np.sum(Ytrain == ypred_train) / len(ypred_train)

        Etest.append(e_test)
        Etrain.append(e_train)
    return np.array([Etest, Etrain, Ltest])

#L = lambda w: Lp(w, Xtrain, Ytrain)
#gradL = lambda w: gradLp(w, Xtrain, Ytrain)





