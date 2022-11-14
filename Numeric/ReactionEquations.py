import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Solving the ordinary differential equations for the reactions in the last layer
#Input: Concentration of neurotransmitters in the last layer
#Output: Solution to the system of reaction differential equations.

#Reaction constants
# k_1 = 1e3-1e4        
# k_0 = 1e-2-10         #k_{-1}
# A= 0.22*1e6 *np.pi    

# # Differential equations for the reactions

# #u = cr density of reseptors, unbounded
# #v = cn density of neurotransmitters

# def dudtfnc(u,t):
#     dudt = -k_1 * u * v + k_0* u
#     return dudt

# def dvdtfnc(w,t):
#     dvdt = k_1 * u * v - k_0* u
#     return dvdt

# # Initial conditions
# u0 = 1000 #Insert concentration in last layer
# v0 = 1000 #Initial receptor count

# # time points
# t = np.linspace(0,15,1000)

# # solve ODE
# u = odeint(dudtfnc,u0,t)
# v = odeint(dvdtfnc,v0,t)

def odes(x,t):
    #Reaction constants
    k_1 = 1e3-1e4        
    k_0 = 1e-2-10         #k_{-1}
    A= 0.22*1e6 *np.pi  

    #Assigning each ODE to a vector element
    u = x[0]
    v = x[1]

    # Defining each ODE
    dudt = -k_1 * u * v + k_0* u
    dvdt = k_1 * u * v - k_0* u

    return [dudt, dvdt]

# Initial conditions
x0= [1000,1000]

# time points
t = np.linspace(0,15,1000)
x = odeint(odes, x0,t)

u = x[:,0]
v = x[:,1]

# plot the results
plt.semilogy(t,u)
plt.semilogy(t,v)
plt.show()

# # plot results
# plt.plot(t,u,'r-',label='Output (u(t))')
# plt.plot(t,v,'r-',label='Output (u(t))')
# plt.plot([0,10,10,40],[0,0,2,2],'b-',label='Input (u(t))')
# plt.ylabel('values')
# plt.xlabel('time')
# plt.legend(loc='best')
# plt.show()
