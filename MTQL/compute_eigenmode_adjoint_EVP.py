"""
Stratified shear flows: adjoint eigenvalue problem
Input: mean fields, transport parameters (Re_b, Fr, Pr) and guess for the eigenvalue
Output: eigenvalue and eigenvector  
"""

import dedalus.public as de
import numpy as np
import logging
logger = logging.getLogger(__name__)

def compute_eigenmode_adjoint_EVP(u,uz,bz,Fr,Pr,Rb,k,domain,sigma_guess):

	# Eigenvalue problem
	problem = de.EVP(domain, variables=['Psi','b','Psiz','bz','Psizz','Psizzz'],eigenvalue='sigma')

	problem.parameters['U0']=u
	problem.parameters['U0z']=uz
	problem.parameters['Rb'] = Rb
	problem.parameters['Pr'] = Pr
	problem.parameters['Fr'] = Fr
	problem.parameters['b0z']=bz
	problem.parameters['k']=k

	problem.add_equation("sigma*(Psizz - k**2*Psi)-Fr/Rb*(k**4*Psi-2*k**2*Psizz+dz(Psizzz))+1j*k*b=  2*1j*k*U0z*Psiz + 1j*k*U0*(Psizz - k**2*Psi)-b0z*1j*k*b")
	problem.add_equation("-1j*k*Psi+sigma*b-Fr/(Pr*Rb)*(dz(bz)-k**2*b)  = 1j*k*U0*b ")

	problem.add_equation("Psiz - dz(Psi) = 0")
	problem.add_equation("Psizz - dz(Psiz) = 0")
	problem.add_equation("Psizzz - dz(Psizz) = 0")
	problem.add_equation("bz - dz(b) = 0")

	problem.add_bc("left(Psi)-right(Psi)=0")
	problem.add_bc("left(Psiz)-right(Psiz)=0")
	problem.add_bc("left(Psizz) -right(Psizz)=0")
	problem.add_bc("left(Psizzz) -right(Psizzz)=0")
	problem.add_bc("left(b) -right(b)=0")
	problem.add_bc("left(bz)-right(bz)=0")

	solver = problem.build_solver()
	solver.solve_sparse(solver.pencils[0],N=10,target=sigma_guess)

	# Filter infinite/nan eigenmodes
	finite = np.isfinite(solver.eigenvalues)
	solver.eigenvalues = solver.eigenvalues[finite]
	solver.eigenvectors = solver.eigenvectors[:, finite]

	# Sort eigenmodes by eigenvalue
	order = np.argsort(-np.real(solver.eigenvalues))
	solver.eigenvalues = solver.eigenvalues[order]
	solver.eigenvectors = solver.eigenvectors[:, order]

	solver.set_state(0)

	return (solver.eigenvalues[0],np.conjugate(solver.state['Psi']['g']),np.conjugate(solver.state['b']['g']),np.conjugate(solver.state['bz']['g']),np.conjugate(solver.state['Psiz']['g']),np.conjugate(solver.state['Psizz']['g']))

