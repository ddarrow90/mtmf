"""
From "Exploiting self-organized criticality in strongly stratified turbulence"
by G. P. Chini, G. Michel, K. Julien, C. B. Rocha, C. P. Caulfield
Multiple Time-scale formulation of the QL system (MTQL)
"""

import numpy as np
from dedalus import public as de
from compute_eigenmode_EVP import compute_eigenmode_EVP
from compute_eigenmode_adjoint_EVP import compute_eigenmode_adjoint_EVP
from compute_second_mode import compute_second_mode
from compute_second_mode import compute_2_1_feedback
import logging
logger = logging.getLogger(__name__)

# Parameters
Epsilon = np.sqrt(0.02)	# Epsilon, based on which are scaled Fr and Rb
Pr=1.		# Prandtl number
Fr=Epsilon**2	# Froude number
Rb=1.		# Buoyancy Reynolds number
Re=Rb/Fr**2	# Reynolds number

# Simulation parameters
k=2.4	# initial wavenumber considered
Lz = 2*np.pi/3	# Height of the domain
tf=1.0		# Duration of the simulation
Nz=128		# Number of grid points in z
dt=1e-4	# Time step
dk=0.001	# wavenumber step

# Bases and domain
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=1)
domain = de.Domain([z_basis], np.complex128)
z = domain.grid(0)

# Fields used to save k, sigma and A^2
k_save=domain.new_field()
sigma_save=domain.new_field()
A_save=domain.new_field()

k_save.meta['z']['constant']=True
sigma_save.meta['z']['constant']=True	
A_save.meta['z']['constant']=True	

# Fields used to compute A and the Reynolds stress given the eigenvector (Psi,b)
Psi=domain.new_field()
Psiz=domain.new_field()
Psizz=domain.new_field()
Psi_c=domain.new_field()	# 'c' denotes a complex conjugate
Psiz_c=domain.new_field()

Psi2=domain.new_field() # second mode
Psi2z=domain.new_field()
Psi2_c=domain.new_field()
Psi2z_c=domain.new_field()

Psi1_2=domain.new_field() # feedback from second to first mode
Psi1_2z=domain.new_field()
Psi1_2_c=domain.new_field()
Psi1_2z_c=domain.new_field()


b=domain.new_field()
bz=domain.new_field()
b_c=domain.new_field()

b2=domain.new_field() # second mode
b2_c=domain.new_field()

b1_2=domain.new_field() # feedback from second to first mode
b1_2_c=domain.new_field()

Psi_adj_c=domain.new_field()	# 'adj' denotes the adjoint problem
Psiz_adj_c=domain.new_field()
Psizz_adj_c=domain.new_field()

b_adj_c=domain.new_field()
bz_adj_c=domain.new_field()

RS_u_geom=domain.new_field()
RS_b_geom=domain.new_field()
RS_u_geom=1j*k*z_basis.Differentiate(Psi*Psiz_c-Psi_c*Psiz)
RS_b_geom=1j*k*z_basis.Differentiate(Psi*b_c-Psi_c*b)

RS_u2_geom=2j*k*z_basis.Differentiate(Psi2*Psi2z_c-Psi2_c*Psi2z)
RS_b2_geom=2j*k*z_basis.Differentiate(Psi2*b2_c-Psi2_c*b2)

RS_u1_6_geom=1j*k*z_basis.Differentiate(Psi*Psi1_2z_c+Psi1_2*Psiz_c - Psi_c*Psi1_2z - Psi1_2_c*Psiz)
RS_b1_6_geom=1j*k*z_basis.Differentiate(Psi*b1_2_c + Psi1_2*b_c - Psi_c*b1_2 -Psi1_2_c*b)

RS_u1_8_geom=1j*k*z_basis.Differentiate(Psi1_2*Psi1_2z_c - Psi1_2_c*Psi1_2z)
RS_b1_8_geom=1j*k*z_basis.Differentiate(Psi1_2*b1_2_c  - Psi1_2_c*b1_2)


RS_u=domain.new_field()		
RS_b=domain.new_field()

# Temporary fields used in the main loop
Psi_temp=domain.new_field()
Psiz_temp=domain.new_field()
Psizz_temp=domain.new_field()
b_temp=domain.new_field()
bz_temp=domain.new_field()

# Problem
problem = de.IVP(domain, variables=['u','uz','b','bz','F'])

problem.parameters['Rb'] = Rb
problem.parameters['Pr'] = Pr
problem.parameters['RS_u'] = RS_u
problem.parameters['RS_b'] = RS_b
problem.parameters['Psi'] = Psi
problem.parameters['bw'] = b
problem.parameters['Psi2'] = Psi2
problem.parameters['b2'] = b2
problem.parameters['Psi1_2'] = Psi1_2
problem.parameters['b1_2'] = b1_2
problem.parameters['sigma'] = sigma_save
problem.parameters['A_square'] = A_save
problem.parameters['k'] = k_save

problem.add_equation("dt(u) - (1/Rb)*dz(uz) = F + RS_u ")
problem.add_equation("dt(b) - (1/Rb)*(1/Pr)*dz(bz) = RS_b ")
problem.add_equation("dz(u)  - uz  = 0")
problem.add_equation("dz(b)  - bz  = 0")
problem.add_equation("F =9*cos(3*z)/Rb")

problem.add_bc("right(uz)-left(uz)  = 0")
problem.add_bc("right(u)-left(u)  = 0")
problem.add_bc("right(b)-left(b)  = 0")
problem.add_bc("right(bz)-left(bz)  = 0")


# Build solver
solver = problem.build_solver(de.timesteppers.RK222) #2nd order Runge Kutta timestepper
solver.stop_sim_time = tf
U_MF = solver.state['u']
B_MF = solver.state['b']
UZ_MF = solver.state['uz']
BZ_MF = solver.state['bz']
F=domain.new_field()
F = solver.state['F']
F['g']=9.*np.cos(3*z)/Rb

# Snapshots (store all the variables every 100 timesteps)
snapshots = solver.evaluator.add_file_handler('snapshots', iter=100, max_writes=1000)
snapshots.add_system(solver.state)
snapshots.add_task("Psi", layout='g', name='Psi')
snapshots.add_task("bw", layout='g', name='bw')
snapshots.add_task("Psi2", layout='g', name='Psi2')
snapshots.add_task("b2", layout='g', name='b2')
snapshots.add_task("Psi1_2", layout='g', name='Psi1_2')
snapshots.add_task("b1_2", layout='g', name='b1_2')
snapshots.add_task("sigma", layout='g', name='sigma')
snapshots.add_task("A_square", layout='g', name='A_square')
snapshots.add_task("k", layout='g', name='k')

sigma=0.
Asq=0.

# Main loop
while solver.ok:
	logger.info('  ')
	logger.info('Iteration %i, t = %.2f, sigma = %.8f + %.8f i ' %(solver.iteration,solver.sim_time, np.real(sigma), np.imag(sigma)))
	logger.info('Solving the direct EVP')
	(sigma,Psi['g'],Psiz['g'],Psizz['g'],b['g'],bz['g'])=compute_eigenmode_EVP(U_MF,UZ_MF,BZ_MF,Fr,Pr,Rb,k,domain,0)

	# check over adjacent k if sigma is close to zero
	if np.real(sigma) > -0.02:
		logger.info('Adjusting k = %.3f based on sigma = %.8f + %.8f i' %(k, np.real(sigma), np.imag(sigma)))
		(sigma_temp, Psi_temp['g'], Psiz_temp['g'], Psizz_temp['g'], b_temp['g'], bz_temp['g']) = compute_eigenmode_EVP(U_MF, UZ_MF, BZ_MF, Fr, Pr, Rb, k+dk, domain, sigma)
		if np.real(sigma_temp) > np.real(sigma):
			while np.real(sigma_temp) > np.real(sigma):
				Psi['g'] = Psi_temp['g']
				Psiz['g'] = Psiz_temp['g']
				Psizz['g'] = Psizz_temp['g']
				b['g'] = b_temp['g']
				bz['g'] = bz_temp['g']
				k = k+dk
				sigma = sigma_temp
				(sigma_temp, Psi_temp['g'], Psiz_temp['g'], Psizz_temp['g'], b_temp['g'], bz_temp['g']) = compute_eigenmode_EVP(U_MF, UZ_MF, BZ_MF, Fr, Pr, Rb, k+dk, domain, sigma)
		else:
			(sigma_temp, Psi_temp['g'], Psiz_temp['g'], Psizz_temp['g'], b_temp['g'], bz_temp['g']) = compute_eigenmode_EVP(U_MF, UZ_MF, BZ_MF, Fr, Pr, Rb, k-dk, domain, sigma)
			if np.real(sigma_temp) > np.real(sigma):
				while np.real(sigma_temp) > np.real(sigma):
					Psi['g'] = Psi_temp['g']
					Psiz['g'] = Psiz_temp['g']
					Psizz['g'] = Psizz_temp['g']
					b['g'] = b_temp['g']
					bz['g'] = bz_temp['g']
					k = k-dk
					sigma = sigma_temp
					(sigma_temp, Psi_temp['g'], Psiz_temp['g'], Psizz_temp['g'], b_temp['g'], bz_temp['g']) = compute_eigenmode_EVP(U_MF, UZ_MF, BZ_MF, Fr, Pr, Rb, k-dk, domain, sigma)
	logger.info('New k is %.3f' %(k))

	# We "normalize" Psi with Psi(0)=1 and update the c.c. fields
	norm=1./Psi['g'][0]
	Psi['g']=Psi['g']*norm
	Psiz['g']=Psiz['g']*norm
	Psizz['g']=Psizz['g']*norm
	b['g']=b['g']*norm
	bz['g']=bz['g']*norm
	Psi_c['g']=np.conjugate(Psi['g'])
	Psiz_c['g']=np.conjugate(Psiz['g'])
	b_c['g']=np.conjugate(b['g'])

	if np.real(sigma) > 0 :
		logger.info('Solving the adjoint EVP')
		(sigma_temp,Psi_adj_c['g'],b_adj_c['g'],bz_adj_c['g'],Psiz_adj_c['g'],Psizz_adj_c['g'])=compute_eigenmode_adjoint_EVP(U_MF,UZ_MF,BZ_MF,Fr,Pr,Rb,k,domain,np.conjugate(sigma))

		# For the definitions of the constants C1, C2 and C3, refer to the manuscript
		C1 = 1/(1j*k)*z_basis.Integrate((-k**2*Psi + Psizz)* Psi_adj_c + b*b_adj_c).evaluate()['g'][0]
		C2 = z_basis.Integrate( RS_u_geom.evaluate()  * (k**2*Psi*Psi_adj_c+2*Psiz*Psiz_adj_c+Psi*Psizz_adj_c-b*b_adj_c) - RS_b_geom.evaluate()*(Psiz*b_adj_c+Psi*bz_adj_c)).evaluate()['g'][0]
		C3 = z_basis.Integrate( (z_basis.Differentiate(UZ_MF).evaluate()*(1/Rb) + F  )  * (k**2*Psi*Psi_adj_c+2*Psiz*Psiz_adj_c+Psi*Psizz_adj_c-b*b_adj_c) - z_basis.Differentiate(BZ_MF).evaluate()*1/(Pr*Rb)*(Psiz*b_adj_c+Psi*bz_adj_c ) ).evaluate()['g'][0]

	# Computation of the Reynolds stresses
	if np.real(sigma) < 0 :
		logger.info('Negative growth rate, A=0: both Reynolds stresses are set to zero')
		RS_u['g']=0
		RS_b['g']=0
	else :
		if np.real(C3/C1)< 0 :
			logger.info('Positive growth rate that would decay with A=0: both Reynolds stresses are set to zero')
			RS_u['g']=0
			RS_b['g']=0
		else :
			Asq=-(np.real(C3/C1)/np.real(C2/C1))
			if np.real(sigma)>1e-7 : Asq=Asq-.001*np.real(sigma)/dt/np.real(C2/C1)
			if Asq <0:
					  logger.info('Positive growth rate that would increase with A = 0 and even more with A > 0: fast transient dynamics would have to be included')
			else : 
				logger.info('Positive growth rate that would increase with A = 0: Reynolds stresses are set to some non-zero values, Asq= %f' %(Asq))
				
				(Psi2['g'], Psi2z['g'], b2['g']) = compute_second_mode(U_MF,UZ_MF,BZ_MF,Psi,b,Epsilon,Pr,Rb,k,domain,sigma)
				(Psi1_2['g'], Psi1_2z['g'], b1_2['g']) = compute_2_1_feedback(U_MF,UZ_MF,BZ_MF,Psi_c,b_c,Psi2,b2,Epsilon,Pr,Rb,k,domain,sigma)
				
				Psi2_c['g']=np.conjugate(Psi2['g'])
				Psi2z_c['g']=np.conjugate(Psi2z['g'])
				b2_c = np.conjugate(b2['g'])
				
				Psi1_2_c['g']=np.conjugate(Psi1_2['g'])
				Psi1_2z_c['g']=np.conjugate(Psi1_2z['g'])
				b1_2_c = np.conjugate(b1_2['g'])
				
				# We disable the effect of the feedback term on the Reynolds stress, because it isn't working. More to investigate here.
				RS_u['g']=Asq*RS_u_geom.evaluate()['g'] + Asq**2*RS_u2_geom.evaluate()['g'] + 0*Asq**3*Asq*RS_u1_6_geom.evaluate()['g'] + 0*Asq**4*Asq*RS_u1_8_geom.evaluate()['g']
				RS_b['g']=Asq*RS_b_geom.evaluate()['g'] + Asq**2*RS_b2_geom.evaluate()['g'] + 0*Asq**3*RS_b1_6_geom.evaluate()['g'] + 0*Asq**4*RS_b1_8_geom.evaluate()['g']
	sigma_save['g']=sigma
	A_save['g']=Asq
	k_save['g']=k
	solver.step(dt)
