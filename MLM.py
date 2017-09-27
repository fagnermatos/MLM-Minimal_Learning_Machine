import numpy as np
import random
import util

class MLM(object):

    def __init__(self, nRefs=10):
        self.nRefs = nRefs

    def treina(self, X, y):
        # Pega refs referencias aleatorio
        refX, refY = util.aleatorioRefs(X, y, self.nRefs)

        # Calcula pares de distancia
        Dx = util.distancia(X, refX)
        Dy = util.distancia(y, refY)

        #Calcula B
        B = np.linalg.solve(Dx, Dy)

        return self

    def classifica(self, X):
        return np.where(self._calcZ(X) >= 0, 1, -1)