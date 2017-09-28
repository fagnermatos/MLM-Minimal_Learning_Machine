from sklearn import datasets
from MLM import MLM
import util


iris = datasets.load_iris()
X = iris.data
y = util.binarizar(iris.target)
# print y


mlm = MLM()
mlm.treina(X,y)


