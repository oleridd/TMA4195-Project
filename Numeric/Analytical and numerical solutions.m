% parameters
D =400; %Diffusion rate: 0.02um/s
k = 23e7; %Consumption rate
L = 0.02; %Cleft height: 0.02um
R = 0.4; %Cylindrical diameter : 0.4um
u0 = 6e-13;
tmax=5e-7; %time range: 0--> 5*10(-5)ms
drho = R/100; %rhostep
rho = 0:drho:R;
nrho = length(rho);
dt = tmax/100; %timestep
t = 0:dt:tmax;
nt = length(t);
BesselRoots = ... % first five roots of the first 5 Bessel functions of the 1st kind
[2.4048, 3.8317, 5.1356, 6.3802, 7.5883, 8.7715; ...
5.5201, 7.0156, 8.4172, 9.7610, 11.0647, 12.3386; ...
8.6537, 10.1735, 11.6198, 13.0152, 14.3725, 15.7002; ...
11.7915, 13.3237, 14.7960, 16.2235, 17.6160, 18.9801; ...
14.9309, 16.4706, 17.9598, 19.4094, 20.8269, 22.2178];
sol_ana = zeros(nt,nrho);
for n=1:500
 for i=1:5
 m = (2*n-1) * pi /(2* L);
 lam = BesselRoots(i,1) /R;
 A = 2 * u0 / (pi * L * R^2 * besselj(1, BesselRoots(i,1))^2);
 % Analytical solution 
 sol_ana = sol_ana + A *cos(m*z)* besselj(0,lam*rho)'* exp(-D *(lam^2 +m^2 + k/D) * t);
 end
end
% create 3d plot(rho-t)
surf(rho,t,sol_ana');
title('Diffusion rho-t')
ylabel('t/s');
xlabel('rho/um');
zlabel('concentration/(mol/m^3)');
15
z=0

# NUmerical solution
D = 0.2;
height = 0.02;
width = 0.400;
k = 0.1;
ic = 5000;
tstop = 10;
% domain
xmesh = linspace(0,width,10);
ymesh = linspace(0,width,10);
zmesh = linspace(0,height,10);
tmesh = linspace(0,tstop,10);
dx = max(xmesh)/length(xmesh);
dy = max(ymesh)/length(ymesh);
dz = max(zmesh)/length(zmesh);
dt = max(tmesh)/length(tmesh);
% solution using finite differences (see Week 1 class notes)
nx = length(xmesh); % number of points in x dimension
ny = length(ymesh); % number of points in x dimension
nz = length(zmesh); % number of points in x dimension
nt = length(tmesh); % number of points in t dimension
stepsizex = 1/10; % stepsize for numerical integration
stepsizey = 1/10; % stepsize for numerical integration
stepsizez = 1/10; % stepsize for numerical integration
sol_fd = zeros(nx, ny, nz, nt);
sol_fdx = zeros(nx, ny, nz, nt);
sol_fdy = zeros(nx, ny, nz, nt);
sol_fdz = zeros(nx, ny, nz, nt);
16
sol_fd(ceil(nx/2), ceil(ny/2), 1, 1) = ic; % initial conditions; delta
impulse at center
for t = 1:nt-1
% old_sol_fd = sol_fd;
 % update boundary conditions
 sol_fd(:, :, nz, t+1) = 0; % right boundary conditions; zero value
 sol_fd(:, 1, :, t+1) = 0; % top boundary conditions; zero value.
 sol_fd(:, ny, :, t+1) = 0; % bottom boundary conditions; zero value
 sol_fd(1, :, :, t+1) = 0; % forward boundary conditions; zero value
 sol_fd(nx, :, :, t+1) = 0; % back boundary conditions; zero value
 % update x coordinate for loops
 for z = 1:nz
 for y = 1:ny
 for x = 2:nx-1
 sol_fdx(x, y, z, t) = stepsizex * ...
 (sol_fd(x-1,y, z, t) - 2 * sol_fd(x, y, z, t) +
sol_fd(x+1,y, z, t));
 end
 end
 end

 % update y coordinate for loops
 for z = 1:nz
 for x = 1:nx
 for y = 2:ny-1
 sol_fdy(x, y, z, t) = stepsizey * ...
 (sol_fd(x, y-1, z, t) - 2 * sol_fd(x, y, z, t) + sol_fd(x,
y+1, z, t));
 end
 end
 end

 % update y coordinate for loops
 for x = 1:nx
 for y = 1:ny
 for z = 2:nz-1
 sol_fdz(x, y, z, t) = stepsizez * ...
 (sol_fd(x, y, z-1, t) - 2 * sol_fd(x, y, z, t) + sol_fd(x,
y, z+1, t));
 end
 end
 end
17
 sol_fd(:,:,:,t+1) = sol_fd(:,:,:,t) + sol_fdx(:,:,:,t) +
sol_fdy(:,:,:,t) + sol_fdz(:,:,:,t) - stepsizez.*k.*sol_fd(:,:,:,t);
 sol_fd(:, :, 1, t+1) = sol_fd(:, :, 2, t+1); % left boundary conditions;
zero flux
 figure(t)
 zz = 3;
 surf(xmesh,ymesh,sol_fd(:,:,zz,t))
 title(['Concentration of neurotransmitters, at time interval
',num2str(t*dt),' and at location ',num2str(zz)])
 xlabel('Dimension of the plane containing the surface of the synpase
in um')
 ylabel('Dimension of the plane containing the surface of the synpase
in um')
 zlabel('Number of molecules at position 0.0060 um from the presynaptic
site in um')
end


