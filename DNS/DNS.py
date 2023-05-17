"""
From "Exploiting self-organized criticality in strongly stratified turbulence"
by G. P. Chini, G. Michel, K. Julien, C. B. Rocha, C. P. Caulfield
Direct numerical simulation of the primitive 2D Boussinesq equations
"""

import numpy as np
from dedalus import public as de
import logging
logger = logging.getLogger(__name__)

# Parameters
Epsilon = 0.02	# Epsilon, based on which are scaled Fr and Rb
Pr=1.		# Prandtl number
Fr=Epsilon	# Froude number
Rb=1.		# Buoyancy Reynolds number
Re=Rb/Fr**2	# Reynolds number

## Simulation parameters
k=2.515/Fr	# wave number considered
Lz = 2*np.pi/3	# Height of the domain
Lx = 2*np.pi/k	# Length of the domain
tf=3.5		# Duration of the simulation
Nz=128		# Number of grid points in z
Nx=128		# Number of grid points in x
dt=1e-4	# Time step

# Bases and domain
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)	
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)	
domain = de.Domain([x_basis,z_basis], np.float64)

# Problem
problem = de.IVP(domain, variables=['u','w','p','b'])

problem.parameters['Rb'] = Rb
problem.parameters['Re'] = Re
problem.parameters['Fr'] = Fr
problem.parameters['Pr'] = Pr
problem.parameters['k'] = k
problem.parameters['Lz'] = Lz
problem.parameters['Lx'] = Lx

problem.add_equation("p = 0", condition="(nx == 0) and (nz == 0)")
problem.add_equation("dx(u) + dz(w) = 0", condition="(nx != 0) or (nz != 0)")
problem.add_equation("dt(b) - (1/(Pr*Re))*dx(dx(b)) -(1/(Pr*Rb))* dz(dz(b)) + w    = -(u*dx(b) + w*dz(b))")
problem.add_equation("dt(u) - (1/Re)*dx(dx(u)) -(1/Rb)*dz(dz(u)) + dx(p)     = -(u*dx(u) + w*dz(u))+9*cos(3*z)/Rb")
problem.add_equation("dt(w) - (1/Re)*dx(dx(w)) -(1/Rb)*dz(dz(w)) + (1/Fr**2)*dz(p) - (1/Fr**2)*b = -(u*dx(w) + w*dz(w))") 

# Build solver
solver = problem.build_solver(de.timesteppers.RK222) #second order Runge-Kutta time-stepping scheme
solver.stop_sim_time = tf

# Snapshots (store all the variables every 10 000 timesteps)
snapshots = solver.evaluator.add_file_handler('snapshots', iter=10000, max_writes=50000)
snapshots.add_system(solver.state)

# Analysis (store the energy every 10 timesteps)
analysis = solver.evaluator.add_file_handler('energy', iter=10, max_writes=50000)
analysis.add_task("0.5/(Lx*Lz)*integ(integ(u**2+Fr**2*w**2+b**2,'z'),'x')", name='energy')

# Initial conditions (random perturbation)
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

b = solver.state['b']
zb, zt = z_basis.interval
z = domain.grid(1)
pert =  1e-2 * noise * (zt - z) * (z - zb)
b['g'] = pert

# Main loop
while solver.ok:
  solver.step(dt)
  
  if solver.iteration % 1000 == 0:
     logger.info('Iteration: %i, Time: %e' %(solver.iteration, solver.sim_time))
