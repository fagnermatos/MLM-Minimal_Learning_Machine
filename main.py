from sklearn import datasets
from MLM import MLM
import numpy as np
import util

ITERACOES = 20

iris = datasets.load_iris()
X = iris.data
y = util.binarizar(iris.target)

mlm = MLM()
mlm.treina(X[0:10],y[0:10])
Dyh = mlm.classifica(X[50:150])

erros = []
for i in range(ITERACOES):
    errosI = 0
    for j in range(len(y)):
        errosI += int(np.array_equal(y,Dyh))
    erros.append(errosI)
print np.mean(erros)