import numpy as np

from Numeric.DiffusionFDM2D import DiffusionReactionFDM2D

class SolvingDiffusionReactionFDM2D():
    
    def __init__(self):
        self.__max_timestep=50
        self.__N=10
        self._M=10
        self.__h=15e-9/self.__N
        self.__k=1e-9/500
        self.__κ=8e-7

        self.__N = DiffusionReactionFDM2D(self.__h,self.__k,self.__κ,5000,self.__N,self.__M)
        self.__R= DiffusionReactionFDM2D(self.__h, self.__k, self.__κ, )
        self.__Rb= DiffusionReactionFDM2D



        