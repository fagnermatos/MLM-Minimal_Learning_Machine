import numpy as np
import random

def aleatorioRefs(X, y, refs):
    dados = []
    for i in range(len(y)):
        dados.append((X[i], y[i]))
    random.shuffle(dados)

    Xn = []
    yn = []
    for i in range(len(y)):
        Xn.append(dados[i][0])
        yn.append(dados[i][1])
    Xn = np.array(Xn)
    yn = np.array(yn)

    return Xn[0:refs], yn[0:refs]

def binarizar(y):
    biny = np.zeros((y.shape[0],np.amax(y)+1))
    for i in range(len(y)):
        biny[i][y[i]] = 1
    return np.array(biny)

def removeDuplicatas(X, y):
    Xn = [X[0]]
    yn = [y[0]]
    for i in range(len(X)):
        if not np.any(np.equal(Xn, X[i]).all(1)):
            Xn.append(X[i])
            yn.append(y[i])
    return np.array(Xn), np.array(yn)