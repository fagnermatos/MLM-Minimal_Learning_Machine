from sklearn.model_selection import train_test_split
from sklearn import datasets
from MLM import MLM
import numpy as np
import util

ITERACOES = 20

iris = datasets.load_iris()
X = iris.data
y = util.binarizar(iris.target)
mlm = MLM()

erros = []
for i in range(ITERACOES):

    Xl, Xt , yl, yt = train_test_split(X, y, test_size=0.2)

    mlm.treina(Xl, yl)
    Dyh = mlm.classifica(Xt)

    errosI = 0
    for j in range(len(yt)):
        errosI += int(not np.array_equal(yt[j],Dyh[j]))
        erros.append(errosI)
print np.mean(erros)