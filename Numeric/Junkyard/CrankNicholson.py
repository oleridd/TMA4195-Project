import numpy as np
import matplotlib.pyplot as plt
numpy.set_printoptions(precision=3)

# Solving the modelling equations numerically with Crank-Nicholson

# Assumption 1: cr and crn=0 until the boundary, leaving the regular heat equation (diffusion) up until boundary
# Assumption 2: Reaction happens quickly, st we may disregard diffusion 
# Assumption 3: 

# Initial conditions:
# c(0,0)= 5000/A
# c(x,0) =0 for x\neq 0
# cr(15,0)=1000
# crn(15,0)=0 
# cr(15,t)=500 for t\neq 0

#Reaction constants
k_1 = 1e3-1e4
k_0 = 1e-2-10
A= 0.22*1e6 *np.pi 

#Grid
c = np.empty((50, 15*1e6, 15*1e6))

# Initial conditions
c_top = 5000/A
c_init = 0

##Function to construct tridiagonal matrices
def tridiag(v, d, w, N):
    # Help function 
    # Returns a tridiagonal matrix A=tridiag(v, d, w) of dimension N x N.
    e = np.ones(N)        # array [1,1,...,1] of length N
    A = v*np.diag(e[1:],-1)+d*np.diag(e)+w*np.diag(e[1:],1)
    return A
    
## Crank nicholson solver
def crank_nicholson(Q, f, T, R, N, M, K, sigma, r, c): 
    h = 1/M     # Stepsize in space
    k = T/N     # Stepsize in time
    
    U = np.zeros((N+1,M+1))
    x = np.linspace(0,R,M+1)   # Gridpoints on the x-axis
    t = np.linspace(0,T,N+1)   # Gridpoints on the t-axis
    
    U[0] = f(x,K)
    g = np.zeros(M-1)
    g2 = np.zeros(M-1)
    
    a = 1/(2*h**2) * sigma**2 * x**2
    b = 1/(2*h) * r * x
    c = c
    
    d0  = a[1:-1]*k + 1/2 * c*k
    dpp = -1/2 * a[1:-2]*k - 1/2 * b[1:-2]*k
    dmm = -1/2 * a[2:-1]*k + 1/2 * b[2:-1]*k
    
    A_LHS = np.identity(M-1) + (np.diag(d0) + np.diag(dpp,1) + np.diag(dmm, -1))
    A_RHS = np.identity(M-1) - (np.diag(d0) + np.diag(dpp,1) + np.diag(dmm, -1))
    
    A_LHS_inv = np.linalg.inv(A_LHS)
    
    print("sigma^2 >= 2r: ", (sigma**2>=2*r))
    print("CFL:", k/(h**2) <= 2./(sigma**2 * R**2), k/(h**2), 2./(sigma**2 * R**2))
    
    for n in range(1,N+1):
        U[n][0] = (1-k*c)*U[n-1][0]
        U[n][-1] = U[0][-1]#0
        
        g[0] = (1/2 * a[1] * k - 1/2 * b[1] * k) * U[n][0] + (1/2 * a[1] * k - 1/2 * b[1] * k) * U[n-1][0]
        g[-1]= (1/2 * a[-2]* k + 1/2 * b[-2]* k) * U[n][-1]+ (1/2 * a[-2]* k + 1/2 * b[-2]* k) * U[n-1][-1]
        
        #U[n][1:-1] = np.linalg.solve(A_LHS, np.dot(A_RHS,U[n-1][1:-1]) + g)
        U[n][1:-1] = np.dot(A_LHS_inv, np.dot(A_RHS,U[n-1][1:-1]) + g)
    
    return U,x,t



def plot_solution(x, t, U, txt='Solution'):
    # Plot the solution of the heat equation
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    T, X = np.meshgrid(t,x)
    # ax.plot_wireframe(T, X, U)
    ax.plot_surface(T, X, U, cmap=cm.coolwarm)
    ax.view_init(azim=30)              # Rotate the figure
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(txt)

# Plotting heatmap

# def plotheatmap(c_k, k):
#   # Clearing the plot
#   plt.clf()
#   plt.title(f"Concentration at t = {k*delta_t:.3f}")
#   plt.xlabel("x")   
#   plt.ylabel("y")
  
#   # Plotting c at timestep k
#   plt.pcolormesh(c_k, cmap=plt.cm.jet, vmin=0, vmax=100)
#   plt.colorbar()
  
#   return plt