from fenics import *
from mshr import * 
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import math
import logging

import lin_solver_t, orders

class Quasi_Linear_PDE_t(lin_solver_t.Linear_PDE_t):

	"""

	Attributes
	----------
	
	mu : function
		Spatially dependent function
	csi : function
		Non-linearity function, upper and lower uniform positive bounds are required.
	csi_p : function
		Derivative of csi.
	csi_pp : function
		Second derivative of csi.
	tol : float
		Arbitrary small parameter needed for the SQP stopping criterion.
	max_it_nm : int
		Maximum number of Newton's method iterations.

	M_u_sp : scipy.sparse.csr.csr_matrix
		Mass matrix of shape (mesh_size, mesh_size),
		associated to the space at a given time.
	u_final : numpy.ndarray
		Array of shape (mesh_size * num_steps,),
		space discretization of ontrol at time T.
	b_array : numpy.ndarray
		Array of shape (mesh_size * num_steps,),
		space and time discretization of the equation RHS (contains u_0, f, g).
	u : dict[int, dolfin.function.function.Function]
		keys from 0 to num_steps, 
		values are the solution functions at key-th timestep.
	
	Methods
	-------
	set_non_linearity(mu, csi, csi_p, csi_pp)
	set_maxit_and_tol(tol, max_it_nm)
	compute_invariants()
	subproblem_solve(u_old)
	newton_solve()
	plot_increments(path)

	"""

	def set_non_linearity(self, mu, csi, csi_p, csi_pp):

		"""

		Parameters
		----------
		mu : function
			Spatially dependent function
		csi : function
			Non-linearity function, upper and lower uniform positive bounds are required.
		csi_p : function
			Derivative of csi.
		csi_pp : function
			Second derivative of csi.

		"""

		self.mu = mu
		self.csi = csi
		self.csi_p = csi_p
		self.csi_pp = csi_pp

	def set_maxit_and_tol(self, tol, max_it_nm):

		"""

		Parameters
		----------
		tol : float
			Arbitrary small parameter needed for the SQP stopping criterion.
		max_it_nm : int
			Maximum number of Newton's method iterations

		"""		

		self.tol = tol
		self.max_it_nm = max_it_nm

	
	def compute_invariants(self):

		""" Computes all the time independant variables
		needed in the sub-problem solver and stores them as class attributes:
		M_u_sp, b_array.

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		u_init = interpolate(self.u_0, self.V).vector().get_local()
		
		u = TrialFunction(self.V)
		v = TestFunction(self.V)

		start_time = time.perf_counter()

		m_u = u*v*dx

		# assembling FENICS mass matrix
		M_u = assemble(m_u)

		# apply dirichlet boundary conditions
		if not self.dirichlet_boundary is None:
			# primal
			self.bc = DirichletBC(
				self.V,
				self.u_D, 
				self.dirichlet_boundary)
			self.bc.apply(M_u)

		# state and adjoint equations
		# B_mat = as_backend_type(B).mat()

		# B_sp = sps.csr_matrix(B_mat.getValuesCSR()[::-1], shape = B_mat.size)

		# convert to sparse matirix
		M_u_mat = as_backend_type(M_u).mat()

		M_u_sp = sps.csr_matrix(M_u_mat.getValuesCSR()[::-1], shape = M_u_mat.size)

		self.M_u_sp = M_u_sp

		# interested in all controls but the one at initial time
		M_u_sp_blocks = sps.block_diag([M_u_sp for i in range(1, self.num_steps)], format='csc')
		self.M_u_sp_blocks = sps.bmat(
			[ 	[None, M_u_sp_blocks], 
				[np.zeros((self.mesh_size, self.mesh_size)), None]	],
			format='csc')

		M_u_sp_blocks = sps.block_diag([M_u_sp for i in range(self.num_steps)], format='csc')
		
		# compute the term depending on u_0 in the RHS
		b_u_0 = M_u_sp.dot(u_init)
		
		b_array = np.hstack((
			b_u_0, 
			np.zeros(self.mesh_size*(self.num_steps - 1))))

		# initialize an array to store the term containing f, g in RHS
		rhs_list = []
		
		# time stepping 
		t = 0
		for n in range(self.num_steps + 1):

			# f, g on the rhs
			self.f.t, self.g.t = t, t
			# assemble right hand side
			rhs = self.dt*self.f*v*dx + self.dt*self.g*v*ds
			b_rhs = assemble(rhs)
			# apply dirichlet boundary conditions
			if not self.dirichlet_boundary is None:
				self.bc = DirichletBC(
					self.V, 
					self.u_D, 
					self.dirichlet_boundary)
				self.bc.apply(b_rhs)

			# don't care for initial time
			if n > 0:
				rhs_list += list(b_rhs.get_local())

			# update time
			t += self.dt

		# update rhs with the term containing f, g
		b_array += np.array(rhs_list)
		self.b_array = b_array

		# quadratic and cubic polinomials make the size of nodal basis greater
		if self.degree > 1:
			self.mesh_size = np.shape(b_array.get_local())[0]

		end_time = time.perf_counter()

		logging.info(f'invariants computed in {end_time - start_time} s\n')

		return 0		

	def subproblem_solve(self, u_old):

		"""

		Parameters
		----------
		u_old : dict[int, dolfin.function.function.Function]
			keys from 0 to num_steps, 
			values are the previous iterate solution functions at key-th timestep.
		
		Returns
		-------
		dict or None
			dict[int, dolfin.function.function.Function]
			corresponding to solution to subproblem if successful,
			None otherwise.

		"""

		# time at the begininng of primal dual strategy 
		start_time = time.perf_counter()

		# Define variational problem
		u = TrialFunction(self.V)
		# p = TrialFunction(self.V)
		v = TestFunction(self.V)

		start_time = time.perf_counter()

		# initialize lists
		A_sp_list, L_1_list = [], []

		for k in range(self.num_steps + 1):

			# primal equation is , for all v in V
			a_0 = self.dt*self.mu*self.csi(u_old[k])*inner(grad(u), grad(v))*dx
			a_1 = self.dt*self.mu*self.csi_p(u_old[k])*u*inner(grad(u_old[k]), grad(v))*dx

			# assembling FENICS matices for state and adjoint equation
			A = assemble(a_0 + a_1)

			# apply dirichlet boundary conditions
			if self.dirichlet_boundary is not None:
				# primal
				self.bc = DirichletBC(self.V, self.u_D, self.dirichlet_boundary)
				self.bc.apply(A)
				
			# convert to sparse matricx
			A_mat = as_backend_type(A).mat()

			A_sp = sps.csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size)

			# load vector
			el_1 = self.dt*self.mu*self.csi_p(u_old[k])*u_old[k]*inner(grad(u_old[k]), grad(v))*dx
				
			L_1 = assemble(el_1)

			# apply dirichlet boundary conditions
			if self.dirichlet_boundary is not None:
				# primal
				self.bc.apply(L_1)

			# list appending
			A_sp_list.append(A_sp)

			# interested in every timestep but the initial one
			if k > 0:
				L_1_list += list(L_1)

		# convert to array
		L_1_array = np.array(L_1_list)

		# assemble block matrix for lhs of SE: all times but initial on diagonal
		A_sp_blocks	= sps.bmat(	
			[ [	self.M_u_sp + A_sp_list[i + 1] if i == j 
				else - self.M_u_sp if i - j == 1
				else None 
				for j in range(self.num_steps) ]
				for i in range(self.num_steps) ],
			format='csc')

		# size of block matrices
		time_size = self.mesh_size*self.num_steps

		# assemble ( (size x num_steps) x (size x num_steps) ) sparse lhs of the linear system
		Global_matrix = A_sp_blocks

		# if it == 2:
		# 	print('---------------spy------------------')
		# 	self.spy_sparse(Global_matrix)
		# 	print('----------------ok------------------')

		# assemble (size x num_steps) dense rhs of linear system 
		Global_right_term = self.b_array + L_1_array

		# compute solution of sparse linear system	
		time1 = time.perf_counter()
		sol_sp = spla.spsolve(Global_matrix, Global_right_term)
		time2 = time.perf_counter()
		logging.info(f'sparse system of size {np.shape(sol_sp)[0]} x {np.shape(sol_sp)[0]} solved in: {time2 - time1} s')

		# time1 = time.perf_counter()
		# sol_sp = gpu_solve.gpu_solve(Global_matrix, Global_right_term)
		# time2 = time.perf_counter()
		# print('gpu ',time2-time1, '\n')

		# adding initial condition
		u_vec = np.hstack( ( self.u_init, sol_sp ) )

		# we store also a dictionary of functions, will be useful for visualization and computation of errors
		# initialization
		u_t = {n : Function(self.V) for n in range(self.num_steps + 1)}			
		
		for n in range(self.num_steps + 1):

			additional = Function(self.V)

			if n < self.num_steps:
				additional.vector().set_local(u_vec[n * self.mesh_size : (n + 1) * self.mesh_size])
			else:
				additional.vector().set_local(u_vec[n * self.mesh_size :					  ])

			u_t[n].assign(additional)

		# time at end of primal dual strategy
		end_time = time.perf_counter()

		logging.info(f'SUBPROBLEM SOLVED in {end_time - start_time} s')

		return u_t

	def newton_solve(self):

		"""

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		logging.info(f'size of mesh is {self.mesh_size} with {self.num_steps} timesteps')
		logging.info(f'maximum number of iterations for NM is: {self.max_it_nm}, with tolerance: {self.tol}')

		# set initial condition vector
		self.u_init = interpolate(self.u_0, self.V).vector().get_local()

		# initial guess
		init = interpolate(Constant(0.0), self.V)

		u_old = {n : init for n in range(self.num_steps + 1)}

		start_time_sqp = time.perf_counter()

		self.compute_invariants()

		# initialize list to store increments
		self.incr_list = []
		# self.cost_func_list = []
		self.error_sequence_list = []
		# self.error_sequence_list.append(self.compute_inf_errors(y_old, u_old, p_old))

		it = 0
		while True:

			logging.info(f'solving subproblem, iteration {it}')
			# call subproblem solver
			u = self.subproblem_solve(u_old)

			# check if subproblem was solved
			if u is None:
				logging.error(f'subproblem {it} not solved')
				return 1
	
			# compute increment
			diff_vec = np.zeros(self.num_steps + 1)
			for n in range(self.num_steps + 1):
				diff_vec[n] = norm( u[n].vector() - u_old[n].vector(), 'linf')

			max_diff = np.max(diff_vec)
			incr = max_diff
			# store increment
			self.incr_list.append(incr)
			# self.error_sequence_list.append(self.compute_inf_errors(y, u, p))
			# logging.info(f'L-inf differences wrt old iterates: \nstate: {max_state_diff} \ncontrol: {max_control_diff} \nadjoint: {max_adjoint_diff}')
			logging.info(f'increment: {incr}\n')

			# stopping criterion
			if it > 1 and incr < self.tol:
				logging.info('stopping criterion met')
				break

			elif it > self.max_it_nm:
				# store computed triple anyway
				self.u = u
				logging.error(f'NO CONVERGENCE REACHED: maximum number of iterations {self.max_it_sqp} for SQP method excedeed')
				return 1

			u_old = {n : Function(self.V) for n in range(self.num_steps + 1)}			

			# update previous iterates
			for n in range(self.num_steps + 1):
				u_old[n].assign(u[n])

			it += 1

		# store solution 
		self.u = u

		logging.info('--------------------------SQP CONVERGENCE REACHED--------------------------\n')

		# output time at end of while loop
		end_time_sqp = time.perf_counter()
		logging.info(f'total elapsed time: {end_time_sqp - start_time_sqp}')

		return 0

####################################################################

def plot_1D_vary_dt(lam):

	# python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	# independent of choice of constraints

	g = Constant(0.0)

	T 		= 1.0
	beta 	= Constant( 1.0 )
	u_0   	= Expression(' sin(pi*x[0])', degree=5)

	u_exact = Expression(' cos(2*pi*t) * sin(pi*x[0])', t=0, degree=5)
	
	
	f = Expression(
		'''	- pow(pi,2) * cos(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1), cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) 
			- sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) 
			- 2*pi*sin(2*pi*t) * sin(pi*x[0])
		''',
		t=0,
		degree=5)

	g = Constant(0.0)

	mu, tol, max_it_nm = 1, 1e-10, 20
	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	num_steps_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]#, 8192]
	
	L2_same_mesh_list = []
	# Linf_same_mesh_list = []

	total_time_start = time.perf_counter()
	# build mesh
	# 1D
	mesh = IntervalMesh(64, 0.0, 1.0)
	
	# initialize old error
	old_error = 100
	
	for N_steps in num_steps_list:

		# initialize instance
		P = Quasi_Linear_PDE_t(T, N_steps)
		# set up problem by callig class attrubutes
		P.set_space(mesh, 1)
		P.set_dirichlet_boundary_conditions(Constant(0.0), boundary)
		P.set_neumann_boundary_conditions(Constant(0.0), None)
		P.set_equation(f, g, u_0)
		P.set_exact_solution(u_exact)
		# set quasi-linear parameters
		P.set_non_linearity(mu, csi, csi_p, csi_pp)
		P.set_maxit_and_tol(tol, max_it_nm)

		# call solver
		P.newton_solve()
		
		P.compute_errors()
						
		# check if errors are stagnating
		if P.L2_error_norm >= old_error:
			logging.error('solution error is stagnating or increasing!')
			# store errors
			L2_same_mesh_list.append(P.L2_error_norm)
			# exit the loop
			break
			

		# store old errors
		old_error = P.L2_error_norm.copy()
		# store errors
		L2_same_mesh_list.append(P.L2_error_norm)
	
	L2_same_mesh_array = np.array(L2_same_mesh_list)

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP AND MESHSIZE: {total_time_end - total_time_start} s')

	print(L2_same_mesh_array)
	print(orders.compute_one_order(L2_same_mesh_array))

	# diameter = 2.0

	diameter = 1

	L = np.shape(L2_same_mesh_array)[0]

	x_time = np.linspace(num_steps_list[0], num_steps_list[ L-1 ], 	100)

	# plotting
	plt.figure()
	fig, ax1 = plt.subplots(1)

	ax1.loglog(
		num_steps_list[:L],
		L2_same_mesh_array,
		label=r'$||\, u-u_h \,||_{L^2}$',
		marker='s',
		color='b',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='b')
	ax1.loglog(x_time,
		np.power(x_time, -1) * T ,
		label=r'$\tau$',
		linestyle='--',
		linewidth=0.7,
		color='c')
	ax1.set_title(fr'$L^2$-error plotted against number of timesteps, $\lambda =$ {lam}')
	ax1.set(xlabel=r'$T/\tau$', ylabel='error')
	ax1.legend(loc="lower left")

	fig.set_size_inches(6, 4 )
	plt.savefig(f'visualization/1D/L2_relative_errors_1D_vary_dt.pdf')

	return 0

def plot_1D_vary_size(lam):

	# python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	# independent of choice of constraints

	g = Constant(0.0)

	T 		= 1.0
	beta 	= Constant( 1.0 )
	u_0   	= Expression(' sin(pi*x[0])', degree=5)

	u_exact = Expression(' cos(2*pi*t) * sin(pi*x[0])', t=0, degree=5)
	
	
	f = Expression(
		'''	- pow(pi,2) * cos(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1), cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) 
			- sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) 
			- 2*pi*sin(2*pi*t) * sin(pi*x[0])
		''',
		t=0,
		degree=5)

	g = Constant(0.0)

	mu, tol, max_it_nm = 1, 1e-10, 20
	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	mesh_size_list = [2, 4, 8, 16, 32, 64]#, 128]#, 256]
	# mesh_size_list = [5, 10, 20, 40, 80]
	L2_same_timestep_list = []

	total_time_start = time.perf_counter()

	# initialize old error
	old_error = 100

	for N_h in mesh_size_list:

		# build mesh
		# 1D
		new_mesh = IntervalMesh(N_h, 0.0, 1.0)
		# initialize instance
		P = Quasi_Linear_PDE_t(T, 4096)
		# set up problem by callig class attrubutes
		P.set_space(new_mesh, 1)
		P.set_dirichlet_boundary_conditions(Constant(0.0), boundary)
		P.set_neumann_boundary_conditions(Constant(0.0), None)
		P.set_equation(f, g, u_0)
		P.set_exact_solution(u_exact)
		# set quasi-linear parameters
		P.set_non_linearity(mu, csi, csi_p, csi_pp)
		P.set_maxit_and_tol(tol, max_it_nm)

		# call solver
		P.newton_solve()
		
		P.compute_relative_errors()

		# build mesh
		# 1D
		# new_mesh = IntervalMesh(N_h, 0.0, 1.0)
		# # initialize instance
		# if N_h <= 8:
		# 	P = Quasi_Linear_Problem_Box(T, 1024)
		# elif N_h <= 32:
		# 	P = Quasi_Linear_Problem_Box(T, 2048)
		# else:
		# 	P = Quasi_Linear_Problem_Box(T, 4096)
		
		# check if errors are stagnating
		# check if errors are stagnating
		if P.L2_error_norm >= old_error:
			logging.error('solution error is stagnating or increasing!')
			# store errors
			L2_same_timestep_list.append(P.L2_error_norm)
			# exit the loop
			break

		# store old errors
		old_error = P.L2_error_norm.copy()
		# store errors
		L2_same_timestep_list.append(P.L2_error_norm)
		
	L2_same_timestep_array = np.array(L2_same_timestep_list)

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP AND MESHSIZE: {total_time_end - total_time_start} s')

	print(L2_same_timestep_array)
	print(orders.compute_one_order(L2_same_timestep_array))

	# diameter = 2.0

	diameter = 1

	L = np.shape(L2_same_timestep_array)[0]

	x_mesh = np.linspace(mesh_size_list[0], mesh_size_list[ L-1 ], 100)

	# plotting
	plt.figure()
	fig, ax2 = plt.subplots(1)

	ax2.loglog(
		mesh_size_list[:L],
		L2_same_timestep_array,
		label=r'$||\, u-u_h \,||_{L^2}$',
		marker='s',
		color='b',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='b')
	
	ax2.loglog(
		x_mesh,
		diameter**2*np.power(x_mesh, -2),
		label=r'$h^2$',
		linestyle='--',
		linewidth=0.7,
		color='purple')
	ax2.loglog(
		x_mesh,
		diameter*np.power(x_mesh, -1),
		label=r'$h^1$',
		linestyle='--',
		linewidth=0.7,
		color='m')
	ax2.loglog(
		x_mesh,
		diameter**(1/2)*np.power(x_mesh, -1/2),
		label=r'$h^{1/2}$',
		linestyle='--',
		linewidth=0.7,
		color='violet')
	
	ax2.set_title(fr'$L^2$-error plotted against size of the mesh, $\lambda =$ {lam}')
	ax2.set(xlabel=f'{diameter}/h', ylabel='error')
	ax2.legend(loc="lower left")

	fig.set_size_inches(6, 4)
	plt.savefig(f'visualization/1D/L2_relative_errors_1D_vary_size.pdf')

	return 0


def example_1D(lam, mesh_size, num_steps):
	
	# python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	# independent of choice of constraints

	g = Constant(0.0)

	T 		= 1.0
	beta 	= Constant( 1.0 )
	u_0   	= Expression(' sin(pi*x[0])', degree=5)

	u_exact = Expression(' cos(2*pi*t) * sin(pi*x[0])', t=0, degree=5)
	
	
	f = Expression(
		'''	- pow(pi,2) * cos(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1), cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) 
			- sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) 
			- 2*pi*sin(2*pi*t) * sin(pi*x[0])
		''',
		t=0,
		degree=5)

	g = Constant(0.0)

	mu, tol, max_it_nm = 1, 1e-10, 20
	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	total_time_start = time.perf_counter()

	# build mesh
	# 1D
	new_mesh = IntervalMesh(mesh_size, 0.0, 1.0)
	# initialize instance
	P = Quasi_Linear_PDE_t(T, num_steps)
	# set up problem by callig class attrubutes
	P.set_space(new_mesh, 1)
	P.set_dirichlet_boundary_conditions(Constant(0.0), boundary)
	P.set_neumann_boundary_conditions(Constant(0.0), None)
	P.set_equation(f, g, u_0)
	P.set_exact_solution(u_exact)
	# set quasi-linear parameters
	P.set_non_linearity(mu, csi, csi_p, csi_pp)
	P.set_maxit_and_tol(tol, max_it_nm)

	# call solver
	P.newton_solve()
	# compute errors
	P.visualize_1D(0, 1, 128, 'visualization')
	P.visualize_1D_exact(0, 1, 128, 'visualization')

	# P.visualize_paraview('visualization_sqp/paraview/1D')

	P.compute_errors()

	logging.info(fr'computed order of convergence is $q =${math.log(P.incr_list[-1]/P.incr_list[-2]) / math.log(P.incr_list[-2]/P.incr_list[-3])}')

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0


def plot_2D_vary_size(lam):
	
	## python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	def boundary_D(x, on_boundary):
		if on_boundary:
			if near(x[1], 1, 1e-8) or near(x[0], 1, 1e-8):
				return True
			else:
				return False
		else:
			return False

	def boundary_N(x, on_boundary):
		if on_boundary:
			if near(x[1], -1, 1e-8) or near(x[0], -1, 1e-8):
				return True
			else:
				return False
		else:
			return False
	
	mu, tol, max_it_nm = 1, 1e-8, 20

	T 		= 1.0
	beta 	= Constant( 1.0 )
	u_0   	= Expression(
		'sin(pi*x[0])*sin(pi*x[1])',
		degree=5)

	u_exact = Expression(
		'cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])',
		t=0,
		degree=5)
	
	f = Expression(
		'''	- pow(pi,2) * cos(2*pi*t) 
			*( 
				cos(2*pi*t) * (pow(sin(pi*x[0])*cos(pi*x[1]),2) + pow(cos(pi*x[0])*sin(pi*x[1]),2)) 
				* pow(exp(1), cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])) 
				* pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])) + 1, -2) 
				- 2*sin(pi*x[0])*sin(pi*x[1]) 
				* ( pow( pow(exp(1), -cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) ) + 1, -1) + 1 )
			) 
			- 2*pi*sin(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])
		''',
		t=0,
		degree=5)

	g = Constant(0.0)

	mu, tol, max_it_nm = 1, 1e-8, 20

	mesh_size_list = [3, 6, 12, 24]
	
	L2_same_timestep_list = []
	# Linf_same_mesh_list = []

	total_time_start = time.perf_counter()
	
	# initialize old error
	old_error = 100
	
	for N_h in mesh_size_list:

		# initialize instance
		P = Quasi_Linear_PDE_t(T, 1024)
		# build mesh
		# 2D
		new_mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), N_h, N_h)
		# set up problem by callig class attrubutes
		P.set_space(new_mesh, 1)
		P.set_dirichlet_boundary_conditions(Constant(0.0), boundary)
		P.set_neumann_boundary_conditions(Constant(0.0), None)
		P.set_equation(f, g, u_0)
		P.set_exact_solution(u_exact)
		# set quasi-linear parameters
		P.set_non_linearity(mu, csi, csi_p, csi_pp)
		P.set_maxit_and_tol(tol, max_it_nm)
		# call solver
		P.newton_solve()
		# compute errors
		P.compute_errors()
				
		# check if errors are stagnating
		if P.L2_error_norm >= old_error:
			logging.error('solution error is stagnating or increasing!')
			# store error
			L2_same_timestep_list.append(P.L2_error_norm)
			# exit the loop
			break

		# store old errors
		old_error = P.L2_error_norm.copy()
		# store errors
		L2_same_timestep_list.append(P.L2_error_norm)
	
	L2_same_timestep_array = np.array(L2_same_timestep_list)

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER MESHSIZE: {total_time_end - total_time_start} s')

	print(L2_same_timestep_array)
	print(orders.compute_one_order(L2_same_timestep_array))

	# diameter = 2.0

	diameter = 1

	L = np.shape(L2_same_timestep_array)[0]

	x_mesh = np.linspace(mesh_size_list[0], mesh_size_list[ L-1 ], 100)

	# plotting
	plt.figure()
	fig, ax2 = plt.subplots(1)

	ax2.loglog( 
		mesh_size_list[:L],
		L2_same_timestep_array,
		label=r'$||\, u-u_h \,||_{L^2}$',
		marker='s',
		color='b',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='b')
	ax2.loglog(
		x_mesh,
		diameter**2	*np.power(x_mesh, -2),
		label=r'$h^2$',
		linestyle='--',
		linewidth=0.7,
		color='purple')
	ax2.loglog(
		x_mesh,
		diameter*np.power(x_mesh, -1),
		label=r'$h^1$',
		linestyle='--',
		linewidth=0.7,
		color='m')
	ax2.loglog(
		x_mesh,
		diameter**(1/2)*np.power(x_mesh, -1/2),
		label=r'$h^{1/2}$',
		linestyle='--',
		linewidth=0.7,
		color='violet')
	ax2.set_title(fr'$L^2$-error plotted against size of the mesh, $\lambda =$ {lam}')
	ax2.set(xlabel=r'$T/\tau$', ylabel='error')
	ax2.legend(loc="lower left")

	fig.set_size_inches(6, 4)
	plt.savefig(f'visualization/2D/L2_relative_errors_2D_vary_size_distributed.pdf')

	return 0

def plot_2D_vary_dt(lam):

	
	## python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	def boundary_D(x, on_boundary):
		if on_boundary:
			if near(x[1], 1, 1e-8) or near(x[0], 1, 1e-8):
				return True
			else:
				return False
		else:
			return False

	def boundary_N(x, on_boundary):
		if on_boundary:
			if near(x[1], -1, 1e-8) or near(x[0], -1, 1e-8):
				return True
			else:
				return False
		else:
			return False
	
	mu, tol, max_it_nm = 1, 1e-8, 20

	T 		= 1.0
	beta 	= Constant( 1.0 )
	u_0   	= Expression(
		'sin(pi*x[0])*sin(pi*x[1])',
		degree=5)

	u_exact = Expression(
		'cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])',
		t=0,
		degree=5)
	
	f = Expression(
		'''	- pow(pi,2) * cos(2*pi*t) 
			*( 
				cos(2*pi*t) * (pow(sin(pi*x[0])*cos(pi*x[1]),2) + pow(cos(pi*x[0])*sin(pi*x[1]),2)) 
				* pow(exp(1), cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])) 
				* pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])) + 1, -2) 
				- 2*sin(pi*x[0])*sin(pi*x[1]) 
				* ( pow( pow(exp(1), -cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) ) + 1, -1) + 1 )
			) 
			- 2*pi*sin(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])
		''',
		t=0,
		degree=5)

	g = Constant(0.0)

	mu, tol, max_it_nm = 1, 1e-8, 20

	num_steps_list = [4, 8, 16, 32, 64, 128, 256, 512]
	
	L2_same_mesh_list = []
	# Linf_same_mesh_list = []

	total_time_start = time.perf_counter()

	# build mesh
	# 2D
	mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), 24, 24)
	
	# initialize old error
	old_error = 100
	
	for N_steps in num_steps_list:
		# initialize instance
		P = Quasi_Linear_PDE_t(T, N_steps)
		# set up problem by callig class attrubutes
		P.set_space(mesh, 1)
		P.set_dirichlet_boundary_conditions(Constant(0.0), boundary)
		P.set_neumann_boundary_conditions(Constant(0.0), None)
		P.set_equation(f, g, u_0)
		P.set_exact_solution(u_exact)
		# set quasi-linear parameters
		P.set_non_linearity(mu, csi, csi_p, csi_pp)
		P.set_maxit_and_tol(tol, max_it_nm)
		# call solver
		P.newton_solve()
		# compute errors
		P.compute_errors()
		# store errors
		# Linf_same_mesh_list.append(P.Linf_error_norm)
				
		# check if errors are stagnating
		if P.L2_error_norm >= old_error:
			logging.error('solution error is stagnating or increasing!')
			# store error
			L2_same_mesh_list.append(P.L2_error_norm)
			# exit the loop
			break

		# store old errors
		old_error = P.L2_error_norm.copy()
		# store errors
		L2_same_mesh_list.append(P.L2_error_norm)
	
	L2_same_mesh_array = np.array(L2_same_mesh_list)

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP: {total_time_end - total_time_start} s')

	print(L2_same_mesh_array)
	print(orders.compute_one_order(L2_same_mesh_array))

	# diameter = 2.0

	diameter = 1

	L = np.shape(L2_same_mesh_array)[0]

	x_time = np.linspace(num_steps_list[0], num_steps_list[ L-1 ], 	100)

	# plotting
	plt.figure()
	fig, ax1 = plt.subplots(1)

	ax1.loglog( 
		num_steps_list[:L],
		L2_same_mesh_array,
		label=r'$||\, u-u_h \,||_{L^2}$',
		marker='s',
		color='b',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='b')
	ax1.loglog(
		x_time,
		np.power(x_time, -1) * T ,
		label=r'$\tau$',
		linestyle='--',
		linewidth=0.7,
		color='c')
	# if P.control_type == 'neumann boundary':
	ax1.loglog(
		x_time,
		np.sqrt(np.power(x_time, -1) * T),
		label=r'$\tau^{1/2}$',
		linestyle='--',
		linewidth=0.7,
		color='teal')
	ax1.set_title(fr'$L^2$-error plotted against number of timesteps, $\lambda =$ {lam}')
	ax1.set(xlabel=r'$T/\tau$', ylabel='error')
	ax1.legend(loc="lower left")

	fig.set_size_inches(6, 4)
	plt.savefig(f'visualization/2D/L2_relative_errors_2D_vary_dt_distributed.pdf')

	return 0

def example_2D(lam, mesh_size, num_steps):

	## python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	def boundary_D(x, on_boundary):
		if on_boundary:
			if near(x[1], 1, 1e-8) or near(x[0], 1, 1e-8):
				return True
			else:
				return False
		else:
			return False

	def boundary_N(x, on_boundary):
		if on_boundary:
			if near(x[1], -1, 1e-8) or near(x[0], -1, 1e-8):
				return True
			else:
				return False
		else:
			return False
	
	mu, tol, max_it_nm = 1, 1e-8, 20

	T 		= 1.0
	beta 	= Constant( 1.0 )
	u_0   	= Expression(
		'sin(pi*x[0])*sin(pi*x[1])',
		degree=5)

	u_exact = Expression(
		'cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])',
		t=0,
		degree=5)
	
	f = Expression(
		'''	- pow(pi,2) * cos(2*pi*t) 
			*( 
				cos(2*pi*t) * (pow(sin(pi*x[0])*cos(pi*x[1]),2) + pow(cos(pi*x[0])*sin(pi*x[1]),2)) 
				* pow(exp(1), cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])) 
				* pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])) + 1, -2) 
				- 2*sin(pi*x[0])*sin(pi*x[1]) 
				* ( pow( pow(exp(1), -cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) ) + 1, -1) + 1 )
			) 
			- 2*pi*sin(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])
		''',
		t=0,
		degree=5)

	g = Constant(0.0)

	total_time_start = time.perf_counter()

	# build mesh
	# 2D
	new_mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), mesh_size, mesh_size)
	# initialize instance
	P = Quasi_Linear_PDE_t(T, num_steps)
	# set up problem by callig class attrubutes
	P.set_space(new_mesh, 1)
	P.set_dirichlet_boundary_conditions(Constant(0.0), boundary)
	P.set_neumann_boundary_conditions(Constant(0.0), None)
	P.set_equation(f, g, u_0)
	P.set_exact_solution(u_exact)
	# set quasi-linear parameters
	P.set_non_linearity(mu, csi, csi_p, csi_pp)
	P.set_maxit_and_tol(tol, max_it_nm)
	# call solver
	P.newton_solve()
	# compute errors
	P.compute_errors()
	# visualization
	P.visualize_paraview('visualization/paraview/2D/')
	# P.plot_inf_errors('visualization/2D/')

	logging.info(fr'computed order of convergence is $q =${math.log(P.incr_list[-1]/P.incr_list[-2]) / math.log(P.incr_list[-2]/P.incr_list[-3])}')

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0


if __name__ == '__main__':

	level = logging.INFO
	fmt = '[%(levelname)s] %(asctime)s - %(message)s'
	logging.basicConfig(level=level, format=fmt)

	# plot_1D_vary_dt(1e-2)
	# plot_1D_vary_size(1e-2)
	# example_1D(1e-2, 20, 400)
	
	# plot_2D_vary_size(1e-2)
	# plot_2D_vary_dt(1e-2)
	# example_2D(1e-2, 30, 900)

	# plot_2D_N_vary_size(1e-2)
	# plot_2D_N_vary_dt(1e-2)
	# example_2D_N(1e-2, 10, 100)

	logging.info('FINISHED')
