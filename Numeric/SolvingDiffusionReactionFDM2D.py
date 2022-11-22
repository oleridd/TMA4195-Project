import numpy as np

from Numeric.DiffusionFDM2D import DiffusionReactionFDM2D

class SolvingDiffusionReactionFDM2D():
    
    def __init__(self):
        self.__max_timestep=500
        self.__N=10
        self.__M=10
        self.__h=15e-9/self.__N
        self.__k=1e-9/self.__max_timestep
        self.__κ=8e-7
        self.__r=220e-9
        self.__k1=(4*10**6)/(6.022*10**23)*(self.__r**2*np.pi*self.__h)/(0.001*self.__M)
        self.__k2=5

        self.__SN=5000/(self.__r**2*np.pi*self.__h)
        self.__SR=(1000e-6)/self.__h

        self.__IC_Rb=IC = np.zeros((self._N, self._M))
        self.__IC_R=IC = np.zeros((self._N, self._M))

        m_r0 = int(0.25*self._M)
        n_ε  = max(int(0.01*self._N), 1)
        self.__IC_R[n_ε:, :m_r0] = self.__SR / (n_ε + m_r0)
     

        self.__N = DiffusionReactionFDM2D(self.__h,self.__k,self.__κ,self.__SN,self.__N,self.__M,has_diffusion= True)
        self.__R= DiffusionReactionFDM2D(self.__h, self.__k, self.__κ,self.__SR,self.__N,self.__M,IC=self.__IC_R,has_diffusion= False)
        self.__Rb= DiffusionReactionFDM2D(self.__h, self.__k, self.__κ,0,0,self.__M,IC=self.__IC_Rb,has_diffusion= False)
    
    def solve(self):
        for t in range(1, self._max_timestep):
            self.__N._solution[t] = (( self.__N._A @ self.__N._solution[t-1].flatten() )-(self.__k1*np.cross(self.__N._solution[t-1].flatten(),self.__R._solution[t-1].flatten()))+(self.__k2*self.__Rb._solution[t-1].flatten())).reshape((self._N, self._M))
            self.__R._solution[t] = (-(self.__k*self.__k1*np.cross(self.__N._solution[t-1].flatten(),self.__R._solution[t-1].flatten()))+(self.__k*self.__k2*self.__Rb._solution[t-1].flatten())+self.__R._solution[t-1].flatten()).reshape((self._N, self._M))
            self.__Rb._solution[t] = ((self.__k*self.__k1*np.cross(self.__N._solution[t-1].flatten(),self.__R._solution[t-1].flatten()))-(self.__k*self.__k2*self.__Rb._solution[t-1].flatten())+self.__Rb._solution[t-1].flatten()).reshape((self._N, self._M))
        self.__N._solved = True
        self.__R._solved = True
        self.__Rb._solved = True

    def plot(self):
        self.__N.plot()
        self.__R.plot()
        self.__Rb.plot()
        






        