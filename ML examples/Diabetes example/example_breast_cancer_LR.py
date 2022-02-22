
from sklearn.preprocessing import scale
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split

import seaborn as sns

#  Reading and cleaning the data
cancer = pd.read_csv('data_breast_cancer.csv')
#sns.heatmap(cancer.isnull(),cmap='Blues')
cancer.drop('Unnamed: 32', axis=1, inplace=True)
cancer.diagnosis = [1 if each == "M" else 0 for each in cancer.diagnosis]
#plt.figure(figsize = (25, 10))
#sns.heatmap(cancer.corr(),cmap='magma',linewidths=2,linecolor='black',annot=True)
#sns.countplot(cancer['diagnosis'])

#   Train Test Split
Features = cancer[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]

Label = cancer['diagnosis']

X = Features
y = Label
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.33, random_state=0)




m_train, n = Xtrain.shape
m_test, _ = Xtest.shape


# n+1 variables
# for our example, we added an ones columns to the data
x1 = np.ones((m_train, 1))
x2 = np.ones((m_test, 1))

Xtrain = np.hstack((x1, Xtrain))
Xtest = np.hstack((x2, Xtest))
#alpha = 0.1


alpha = 1e-10


def sigma(w, x):
    z = np.dot(x, w)
    return 1/(1 + np.exp(-z))

def Lp(w, x, y):
    yhat = sigma(w, x)
    mm = x.shape[0]
    return -(1/mm) * np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) + alpha * np.sum(w[1:] ** 2)/(2*mm)


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
