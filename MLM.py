from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import util

class MLM(object):

    #hiperparametro
    nRefs = 10

    #parametros
    refX = []
    refY = []
    B = []

    def __init__(self, nRefs=50):
        self.nRefs = nRefs

    def treina(self, X, y):

        # Pega nRefs referencias aleatorio
        self.refX, self.refY = util.aleatorioRefs(X, y, self.nRefs)

        # Calcula pares de distancia
        Dx = euclidean_distances(X, self.refX)
        Dy = euclidean_distances(y, self.refY)

        #Calcula B
        """ Evita o erro LinAlgError('Last 2 dimensions of the array must be square')"""
        self.B = np.linalg.solve(Dx.T.dot(Dx), Dx.T).dot(Dy)
        # self.B = np.linalg.solve(Dx,Dy) # Retorna erro 'Last 2 dimensions of the array must be square'

        return self

    def classifica(self, X):
        Dx = euclidean_distances(X, self.refX)
        Dy = Dx.dot(self.B)

        Dyh = []
        for i in range(len(X)):
            Dyh.append(self.refY[np.argmin(Dy[i])])
        return np.array(Dyh)