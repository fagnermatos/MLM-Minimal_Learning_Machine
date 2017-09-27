from sklearn import datasets
import numpy as np
from MLM import MLM
import random

iris = datasets.load_iris()
X = iris.data
y = np.where(iris.target==0, 1, -1)
