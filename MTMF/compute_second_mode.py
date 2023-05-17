"""
Stratified shear flows: eigenvalue problem
Input: mean fields, transport parameters (Re_b, Fr, Pr) and guess for the eigenvalue
Output: eigenvalue and eigenvector  
"""

import dedalus.public as de
import numpy as np
import logging
logger = logging.getLogger(__name__)

def compute_second_mode(u,uz,bz,Psi,b,Epsilon,Pr,Rb,k,domain,sigma):

	# Eigenvalue problem
	#problem = de.EVP(domain, variables=['Psi','b','Psiz','bz','Psizz','Psizzz'],eigenvalue='sigma')
	problem = de.LinearBoundaryValueProblem(domain, variables=['Psi2','b2','Psi2z','b2z','Psi2zz','Psi2zzz','Psiz','Psizz'])

	problem.parameters['sigma']=sigma
	problem.parameters['U0']=u
	problem.parameters['U0z']=uz
	problem.parameters['Rb'] = Rb
	problem.parameters['Pr'] = Pr
	problem.parameters['eps'] = Epsilon
	problem.parameters['b0z']=bz
	problem.parameters['k']=k
	
	problem.parameters['Psi']=Psi
	problem.parameters['b']=b
	
	#problem.add_equation("2*sigma*(Psi2zz-4*k**2*Psi2)+2j*k*b2-eps**2/Rb*(dz(Psi2zzz)-2*4*k**2*Psi2zz+16*k**4*Psi2)+2j*k*U0*(Psi2zz-4*k**2*Psi2)-2j*k*dz(U0z)*Psi2-eps*(1j*k*Psi*(dz(Psizz)-k**2*dz(Psi))-1j*k*dz(Psi)*(Psizz-k**2*Psi))=0")
	problem.add_equation("2*sigma*(Psi2zz-4*k**2*Psi2)+2j*k*b2-eps**2/Rb*(dz(Psi2zzz)-2*4*k**2*Psi2zz+16*k**4*Psi2)+2j*k*U0*(Psi2zz-4*k**2*Psi2)-2j*k*dz(U0z)*Psi2-eps*(1j*k*Psi*dz(Psizz)-1j*k*dz(Psi)*Psizz)=-eps*(1j*k*Psi*k**2*dz(Psi)-1j*k*dz(Psi)*k**2*Psi)")
	problem.add_equation("2*sigma*b2-2j*k*Psi2-eps**2/(Pr*Rb)*(dz(b2z)-4*k**2*b2) +2j*k*U0*b2 -2j*k*b0z*Psi2=eps*(1j*k*Psi*dz(b)-1j*k*b*dz(Psi))")
	#problem.add_equation("sigma*(Psizz-k**2*Psi)+1j*k*b-Fr/Rb*(dz(Psizzz)-2*k**2*Psizz+k**4*Psi)=-1j*k*U0*(Psizz-k**2*Psi)+1j*k*dz(U0z)*Psi")
	#problem.add_equation("sigma*b-1j*k*Psi-Fr/(Pr*Rb)*(dz(bz)-k**2*b) =-1j*k*U0*b +1j*k*b0z*Psi")


	problem.add_equation("Psiz = dz(Psi)")
	problem.add_equation("Psizz - dz(Psiz) = 0")
	
	problem.add_equation("Psi2z - dz(Psi2) = 0")
	problem.add_equation("Psi2zz - dz(Psi2z) = 0")
	problem.add_equation("Psi2zzz - dz(Psi2zz) = 0")
	problem.add_equation("b2z - dz(b2) = 0")

	#problem.add_bc("left(Psiz)-right(Psiz)=0")
	problem.add_bc("left(Psizz) -right(Psizz)=0")

	problem.add_bc("left(Psi2)-right(Psi2)=0")
	problem.add_bc("left(Psi2z)-right(Psi2z)=0")
	problem.add_bc("left(Psi2zz) -right(Psi2zz)=0")
	problem.add_bc("left(Psi2zzz) -right(Psi2zzz)=0")
	
	problem.add_bc("left(b2) -right(b2)=0")
	problem.add_bc("left(b2z)-right(b2z)=0")

	solver = problem.build_solver()
	solver.solve()#solver.solve_sparse(solver.pencils[0],N=10,target=sigma_guess)

	return (solver.state['Psi2']['g'],solver.state['Psi2z']['g'],solver.state['b2']['g'])

def compute_2_1_feedback(u,uz,bz,Psi_c,b_c,Psi2,b2,Epsilon,Pr,Rb,k,domain,sigma):

	# Eigenvalue problem
	#problem = de.EVP(domain, variables=['Psi','b','Psiz','bz','Psizz','Psizzz'],eigenvalue='sigma')
	problem = de.LinearBoundaryValueProblem(domain, variables=['Psi1_2','b1_2','Psi1_2z','b1_2z','Psi1_2zz','Psi1_2zzz','Psiz_c','Psizz_c','Psi2z','Psi2zz'])

	problem.parameters['sigma']=sigma
	problem.parameters['U0']=u
	problem.parameters['U0z']=uz
	problem.parameters['Rb'] = Rb
	problem.parameters['Pr'] = Pr
	problem.parameters['eps'] = Epsilon
	problem.parameters['b0z']=bz
	problem.parameters['k']=k
	
	problem.parameters['Psi_c']=Psi_c
	problem.parameters['b_c']=b_c
	problem.parameters['Psi2']=Psi2
	problem.parameters['b2']=b2
	
	#	problem.add_equation("2*sigma*(Psi2zz-4*k**2*Psi2)+2j*k*b2-eps**2/Rb*(dz(Psi2zzz)-2*4*k**2*Psi2zz+16*k**4*Psi2)+2j*k*U0*(Psi2zz-4*k**2*Psi2)-2j*k*dz(U0z)*Psi2-eps*(1j*k*Psi*dz(Psizz)-1j*k*dz(Psi)*Psizz)=-eps*(1j*k*Psi*k**2*dz(Psi)-1j*k*dz(Psi)*k**2*Psi)")
	problem.add_equation("sigma*(Psi1_2zz-k**2*Psi1_2)+1j*k*b1_2-eps**2/Rb*(dz(Psi1_2zzz)-2*k**2*Psi1_2zz+k**4*Psi1_2)+1j*k*U0*(Psi1_2zz-k**2*Psi1_2)-1j*k*dz(U0z)*Psi1_2-eps*(-1j*k*Psi_c*dz(Psi2zz)+2j*k*Psi2*dz(Psizz_c)-2j*k*dz(Psi_c)*Psi2zz+1j*k*dz(Psi2)*Psizz_c)=-eps*(-1j*k*Psi_c*4*k**2*dz(Psi2)+2j*k*Psi2*k**2*dz(Psi_c)-2j*k*dz(Psi_c)*4*k**2*Psi2+1j*k*dz(Psi2)*k**2*Psi_c)")
	#problem.add_equation("2*sigma*b2-2j*k*Psi2-eps**2/(Pr*Rb)*(dz(b2z)-4*k**2*b2) +2j*k*U0*b2 -2j*k*b0z*Psi2=eps*(1j*k*Psi*dz(b)-1j*k*b*dz(Psi))")
	problem.add_equation("sigma*b1_2-1j*k*Psi1_2-eps**2/(Pr*Rb)*(dz(b1_2z)-k**2*b1_2) +1j*k*U0*b1_2 -1j*k*b0z*Psi1_2=eps*(-1j*k*Psi_c*dz(b2)+2j*k*Psi2*dz(b_c)-2j*k*b2*dz(Psi_c)+1j*k*b_c*dz(Psi2))")


	problem.add_equation("Psiz_c = dz(Psi_c)")
	problem.add_equation("Psizz_c - dz(Psiz_c) = 0")
	
	problem.add_equation("Psi2z = dz(Psi2)")
	problem.add_equation("Psi2zz - dz(Psi2z) = 0")
	
	problem.add_equation("Psi1_2z - dz(Psi1_2) = 0")
	problem.add_equation("Psi1_2zz - dz(Psi1_2z) = 0")
	problem.add_equation("Psi1_2zzz - dz(Psi1_2zz) = 0")
	problem.add_equation("b1_2z - dz(b1_2) = 0")

	#problem.add_bc("left(Psiz)-right(Psiz)=0")
	problem.add_bc("left(Psizz_c) -right(Psizz_c)=0")
	problem.add_bc("left(Psi2zz) -right(Psi2zz)=0")

	problem.add_bc("left(Psi1_2)-right(Psi1_2)=0")
	problem.add_bc("left(Psi1_2z)-right(Psi1_2z)=0")
	problem.add_bc("left(Psi1_2zz) -right(Psi1_2zz)=0")
	problem.add_bc("left(Psi1_2zzz) -right(Psi1_2zzz)=0")
	
	problem.add_bc("left(b1_2) -right(b1_2)=0")
	problem.add_bc("left(b1_2z)-right(b1_2z)=0")

	solver = problem.build_solver()
	solver.solve()#solver.solve_sparse(solver.pencils[0],N=10,target=sigma_guess)

	return (solver.state['Psi1_2']['g'],solver.state['Psi1_2z']['g'],solver.state['b1_2']['g'])

