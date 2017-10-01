from sklearn.model_selection import train_test_split
from sklearn import datasets
from MLM import MLM
import numpy as np
import util

ITERACOES = 20
TEST_SIZE = 0.2

iris = datasets.load_iris()
X = iris.data
y = util.binarizar(iris.target)
X, y = util.removeDuplicatas(X, y)
mlm = MLM()

acertos = []
for i in range(ITERACOES):
    Xl, Xt , yl, yt = train_test_split(X, y, test_size=TEST_SIZE)

    mlm.treina(Xl, yl)
    Dyh = mlm.classifica(Xt)

    errosI = 0
    for j in range(len(yt)):
        errosI += int(not np.array_equal(yt[j],Dyh[j]))
    acertos.append(1 - errosI/ float(len(yt)))
print "Total de acertos do MLM: ", np.mean(acertos)*100, "%"

