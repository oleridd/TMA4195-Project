import numpy as np
import matplotlib.pyplot as plt


from Numeric.DiffusionReactionFDM2D import DiffusionReactionFDM2D
from Numeric.Utility import DiffusionReaction2DStdConfig


class SolvingDiffusionReactionFDM2D():
    
    def __init__(self):
        # Constructing particle object instances:
        config_N  = (conf_params := DiffusionReaction2DStdConfig(particle_type='N' )).config # Standard parameters for N
        config_R  = DiffusionReaction2DStdConfig(particle_type='R' ).config                  # Standard parameters for R
        config_Rb = DiffusionReaction2DStdConfig(particle_type='Rb').config                  # Standard parameters for R

        self.__neuro_particle = DiffusionReactionFDM2D(**config_N,  has_diffusion=True )
        self.__recep_particle = DiffusionReactionFDM2D(**config_R,  has_diffusion=False)
        self.__rebnd_particle = DiffusionReactionFDM2D(**config_Rb, has_diffusion=False)
    

        # Constructing reaction constants:
        self.__max_timestep, self.__N, self.__M, self.__h, self.__k = conf_params.get_basic_parameters()
        self.__k1 = (4*10**6)/(6.022*10**23)*(config_N['r']**2*np.pi*config_N['h'])/(0.001*config_N['M'])
        self.__k2 = 5
    

    def solve(self):
        """
        Solves the reaction diffusion equation for all three particles.

        Args:
            None
        Returns:
            None
        """
        A = self.__neuro_particle.A
        P = len(A)
        B = np.identity(P) - self.__neuro_particle.r*A

        for t in range(1, self.__max_timestep):
            
            prev_N  = self.__neuro_particle.get_solution(t-1).flatten()
            prev_R  = self.__recep_particle.get_solution(t-1).flatten()
            prev_Rb = self.__rebnd_particle.get_solution(t-1).flatten()

            # Neurotransmitters:
            self.__neuro_particle.set_solution(
                t,
                (
                    np.linalg.solve(B + self.__k*self.__k1*prev_R*np.identity(P), prev_N + self.__k2*prev_Rb)
                ).reshape((self.__N, self.__M))
            )

            # Receptors:
            self.__recep_particle.set_solution(
                t,
                (
                          prev_R  + self.__k*((-self.__k1*prev_N*prev_R) + (self.__k2*prev_Rb))
                ).reshape((self.__N, self.__M))
            )

            # Bound receptors
            self.__rebnd_particle.set_solution(
                t,
                (
                          prev_Rb + self.__k*(( self.__k1*prev_N*prev_R) - (self.__k2*prev_Rb))
                ).reshape((self.__N, self.__M))
            )

        self.__neuro_particle.set_solved()
        self.__recep_particle.set_solved()
        self.__rebnd_particle.set_solved()


    def plot(self) -> None:
        """
        Plots the solutions for concentration of all of the different
        particles.
        """
        fig, ax = plt.subplots(1, 3, figsize=(16, 8))
        self.__sliders = []

        self.__sliders.append(self.__neuro_particle.plot(slider=True, ax=ax[0]))
        ax[0].set_title("Neurotransmitters concentration")

        self.__sliders.append(self.__recep_particle.plot(slider=True, ax=ax[1]))
        ax[1].set_title("Receptors concentration")

        self.__sliders.append(self.__rebnd_particle.plot(slider=True, ax=ax[2]))
        ax[2].set_title("Bounded receptors concentration")
    

    # def plot(self):
    #     self.__N.plot()
    #     self.__R.plot()
    #     self.__Rb.plot()