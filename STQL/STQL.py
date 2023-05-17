"""
From "Exploiting self-organized criticality in strongly stratified turbulence"
by G. P. Chini, G. Michel, K. Julien, C. B. Rocha, C. P. Caulfield
Single Time-scale formulation of the QL system (STQL)
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
k=2.515	# wave number considered
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
problem = de.IVP(domain, variables=['u','b','up','wp','bp','pp'])

problem.parameters['Rb'] = Rb
problem.parameters['Fr'] = Fr
problem.parameters['Pr'] = Pr
problem.parameters['Lx'] = Lx
problem.parameters['Lz'] = Lz

problem.add_equation("dt(u)/Fr -(1/Rb)*dz(dz(u))     = -dz(integ_x(up*wp)/Lx) +9*cos(3*z)/Rb")
problem.add_equation("dt(b)/Fr -(1/(Pr*Rb))* dz(dz(b))     = -dz(integ_x(wp*bp)/Lx)")

problem.add_equation("dt(up) + dx(pp) -(Fr/Rb)*dx(dx(up)) -(Fr/Rb)*dz(dz(up))  = -u*dx(up)-wp*dz(u)") 
problem.add_equation("dt(wp) + dz(pp) -bp -(Fr/Rb)*dx(dx(wp)) -(Fr/Rb)*dz(dz(wp))  = -u*dx(wp)") 
problem.add_equation("dt(bp)  +wp  - (Fr/(Pr*Rb))*dx(dx(bp)) -(Fr/(Pr*Rb))* dz(dz(bp))    = -u*dx(bp) - wp*dz(b)")

problem.add_equation("dx(up) + dz(wp) = 0", condition="(nx != 0) or (nz != 0)")
problem.add_equation("pp = 0", condition="(nx == 0) and (nz == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222) #2nd order Runge Kutta timestepper
solver.stop_sim_time = tf/Fr

# Snapshots (store all the variables every 10 000 timesteps)
snapshots = solver.evaluator.add_file_handler('snapshots', iter=10000, max_writes=500000)
snapshots.add_system(solver.state)

# Analysis (store the energy every 10 timesteps)
analysis = solver.evaluator.add_file_handler('energy', iter=10, max_writes=50000)
analysis.add_task("0.5/(Lx*Lz)*integ(integ( (u+sqrt(Fr)*up)**2+Fr*wp**2+(b+sqrt(Fr)*bp)**2,'z'),'x')", name='energy')
analysis.add_task("Fr*t", name='slow time')

# Initial conditions (random perturbation)
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

b = solver.state['b']
bp = solver.state['bp']
zb, zt = z_basis.interval
z = domain.grid(1)
pert =  1e-2 * noise * (zt - z) * (z - zb)
bp['g'] = pert


while solver.ok:
  solver.step(dt/Fr)
  
  if (solver.iteration-1) % 1000 == 0:
     logger.info('Iteration: %i, Time: %e' %(solver.iteration, solver.sim_time*Fr))
