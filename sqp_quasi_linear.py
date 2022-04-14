from fenics import *
from mshr import * 
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import math
import sys
import logging

import primal_dual, orders

class Quasi_Linear_Problem_Box(primal_dual.Linear_Problem_Box):


	"""

	Attributes
	----------
	compute_proj_residuals_flag : bool
		Tells if to compute the projection residuals or not.
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
	tol_pd : float
		Arbitrary small parameter needed for the primal-dual stopping criterion.
	max_it_sqp : int
		Maximum number of SQP iterations.
	max_it_pd : int
		Maximum number of primal-dual active set strategy iterations.

	M_y_sp : scipy.sparse.csr.csr_matrix
		Mass matrix of shape (mesh_size, mesh_size),
		associated to the state space at a given time.
	B_sp_blocks : scipy.sparse.csr.csr_matrix
		Above diagonal block matrix of shape (mesh_size * num_steps, mesh_size * num_steps), 
		entries are the beta-mass matrices (not time dependant).
	M_proj_sp_blocks : scipy.sparse.csr.csr_matrix
		Diagonal block matrix of shape (mesh_size * num_steps, mesh_size * num_steps), 
		entires are the mass-matrices that need to multiply the adjoint state before to project it (no boundary conditions).
	u_final : numpy.ndarray
		Array of shape (mesh_size * num_steps,),
		space discretization of ontrol at time T.
	U_a : numpy.ndarray
		Array of shape (mesh_size * num_steps,), 
		space and time discretization of lower constraint at all times but the final one.
	U_b : numpy.ndarray
		Array of shape (mesh_size * num_steps,), 
		space and time discretization of upper constraint at all times but the final one.
	b_target_array : numpy.ndarray
		Array of shape (mesh_size * num_steps,),
		space and time discretization of y_target function in the cost functional.
	b_array : numpy.ndarray
		Array of shape (mesh_size * num_steps,),
		space and time discretization of the state equation RHS (contains y_0, u_T, f).
	y : dict[int, dolfin.function.function.Function]
		keys from 0 to num_steps, 
		values are the solution state functions at key-th timestep.
	u : dict[int, dolfin.function.function.Function]
		keys from 0 to num_steps, 
		values are the solution control functions at key-th timestep.
	p : dict[int, dolfin.function.function.Function]
		keys from 0 to num_steps, 
		values are the solution adjoint state functions at key-th timestep.
	
	Methods
	-------
	set_non_linearity(mu, csi, csi_p, csi_pp)
	set_maxit_and_tol(tol, max_it_sqp, max_it_pd)
	compute_associated_adjoint(u_t)
	compute_invariants()
	subproblem_solve(y_old, p_old)
	sqp_solve()
	compute_LSE_terms(y_old)
	assemble_LSE_rhs(u_vec, rhs_lin_term)
	assemble_AE_rhs(y_vec)
	compute_associated_adjoint(u_vec)
	compute_proj_residuals(X_a, X_b, I_a_b, proj, control)
	plot_residuals(path)

	"""

	compute_proj_residuals_flag = False
	tol_nm = 1e-8

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

	def set_maxit_and_tol(self, tol, tol_pd, max_it_sqp, max_it_pd):

		"""

		Parameters
		----------
		tol : float
			Arbitrary small parameter needed for the SQP stopping criterion.
		tol_pd : float
			Arbitrary small parameter needed for the primal-dual stopping criterion.
		max_it_sqp : int
			Maximum number of SQP iterations.
		max_it_pd : int
			Maximum number of primal-dual active set strategy iterations.

		"""		

		self.tol = tol
		self.tol_pd = tol_pd
		self.max_it_sqp = max_it_sqp
		self.max_it_pd = max_it_pd

	def compute_invariants(self):

		""" Computes all the time independant variables
		needed in the sub-problem solver and stores them as class attributes:
		M_y_sp, B_sp_blocks, M_proj_sp_blocks,
		u_final, U_a, U_b, b_target_array, b_array.

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		p_final = interpolate(self.p_end, self.V).vector().get_local() 
		y_init = interpolate(self.y_0, self.V).vector().get_local()
		
		y = TrialFunction(self.V)
		p = TrialFunction(self.V)
		# control space ?
		u = TrialFunction(self.V)
		v = TestFunction(self.V)

		start_time = time.perf_counter()

		m_y	= y*v*dx

		if self.control_type == 'distributed':
			rhs_u = self.dt*self.beta*u*v*dx
			m_u = u*v*dx
			m_p = (1.0/self.lam)*self.beta*p*v*dx

		elif self.control_type == 'neumann boundary':
			rhs_u = self.dt*self.beta*u*v*ds
			m_u = u*v*ds
			m_p = (1.0/self.lam)*self.beta*p*v*ds

		elif self.control_type == 'time':
			rhs_u = self.dt*self.beta*y*v*dx
			m_u = u*v*dx
			m_p = (1.0/self.lam)*self.beta*p*v*dx
			p_int = (1.0/self.lam)*self.beta*v*dx

		# assembling FENICS mass matrices for projection formula
		M_u, M_p = assemble(m_u), assemble(m_p)

		# assembling FENICS matices for state equation
		M_y, B = assemble(m_y), assemble(rhs_u)

		# apply dirichlet boundary conditions
		if not self.dirichlet_boundary is None:
			# primal
			self.bc = DirichletBC(
				self.V,
				self.y_D, 
				self.dirichlet_boundary)
			self.bc.apply(M_y)
			if not self.control_type == 'time':
				self.bc.apply(B)

		# state and adjoint equations
		B_mat = as_backend_type(B).mat()

		B_sp = sps.csr_matrix(B_mat.getValuesCSR()[::-1], shape = B_mat.size)

		# gradient equation and rhs mass matrices
		M_u_mat, M_p_mat, M_y_mat = as_backend_type(M_u).mat(), as_backend_type(M_p).mat(), as_backend_type(M_y).mat()

		M_u_sp, M_p_sp, M_y_sp = sps.csr_matrix(M_u_mat.getValuesCSR()[::-1], shape = M_u_mat.size), sps.csr_matrix(M_p_mat.getValuesCSR()[::-1], shape = M_p_mat.size), sps.csr_matrix(M_y_mat.getValuesCSR()[::-1], shape = M_y_mat.size)

		self.M_y_sp = M_y_sp

		# quadratic and cubic polinomials make the size of nodal basis greater
		if self.degree > 1:
			self.mesh_size = np.shape(y_init)[0]

		# interested in all controls but the one at initial time
		B_sp_blocks = sps.block_diag([B_sp for i in range(1, self.num_steps)], format='csc')
		self.B_sp_blocks = sps.bmat(
			[ 	[None, B_sp_blocks], 
				[np.zeros((self.mesh_size, self.mesh_size)), None]	],
			format='csc')

		M_u_sp_blocks = sps.block_diag([M_u_sp for i in range(self.num_steps)], format='csc')
		M_p_sp_blocks = sps.block_diag([M_p_sp for i in range(self.num_steps)], format='csc')

		if self.control_type == 'distributed' or self.control_type == 'neumann boundary':
			# compute inverse of the control mass matrix
			M_u_sp_inv = spla.spsolve( M_u_sp, sps.identity(self.mesh_size))
			M_u_sp_inv_blocks = sps.block_diag([M_u_sp_inv for i in range(self.num_steps)], format='csc')
			self.M_proj_sp_blocks = M_u_sp_inv_blocks.dot(M_p_sp_blocks)

		elif self.control_type == 'time':	
			self.P = assemble(p_int)
			M_p_int = np.vstack([self.P for i in range(self.mesh_size)])
			M_p_int_sp = sps.csr_matrix(M_p_int, shape=np.shape(M_p_int))
			self.M_proj_sp_blocks = sps.block_diag([M_p_int_sp for i in range(self.num_steps)], format='csc')
			
		# compute the term depending on y_0 in the RHS of the state equation
		b_y_0 = M_y_sp.dot(y_init)
		# compute final control
		self.u_final = self.compute_final_control(p_final)
		# compute the term depending on u_T in the RHS of the state equation
		b_u_T = B_sp.dot(self.u_final)
		# RHS of state equation (without the term containing f)
		b_array = np.hstack((
			b_y_0, 
			np.zeros(self.mesh_size*(self.num_steps - 2)), 
			b_u_T))

		# initialize a list to store the term containing y_target^k in RHS of adjoint equation
		b_target_list = []
		# initialize an array to store the term containing f^k in RHS of state equation
		b_f_list = []
		# initialize lists of discrete vectors of u_a, u_b of size (num_steps x size)
		U_a_list, U_b_list = [], []

		# time stepping 
		t = 0
		for n in range(self.num_steps + 1):

			# update target function, f on the rhs and box constraints with current time
			self.y_target.t = t
			self.u_a.t, self.u_b.t = t, t
			self.f.t, self.g.t = t, t

			# right hand side of AE and SE are time dependent
			rhs_target = - self.dt*self.y_target*v*dx			
			rhs_f = self.dt*self.f*v*dx + self.dt*self.g*v*ds
			# assemble right hand side AE and SE
			b_target = assemble(rhs_target)
			b_f = assemble(rhs_f)
			# apply dirichlet boundary conditions
			if not self.dirichlet_boundary is None:
				self.bc_adj = DirichletBC(
					self.V, 
					self.p_D, 
					self.dirichlet_boundary)
				self.bc_adj.apply(b_target)
				self.bc = DirichletBC(
					self.V, 
					self.y_D, 
					self.dirichlet_boundary)
				self.bc.apply(b_f)

			# in the first entry we would have have to add the term containing y_0, but it's multiplied by D^0 so it will be added in subproblem_solve()
		
			# don't care for final time
			if n < self.num_steps:
				b_target_list += list(b_target.get_local())						
				U_a_list += list(interpolate(self.u_a, self.V).vector().get_local())
				U_b_list += list(interpolate(self.u_b, self.V).vector().get_local())

			# don't care for initial time
			if n > 0:
				b_f_list += list(b_f.get_local())

			# update time
			t += self.dt

		# convert to numpy arrays and store
		self.U_a, self.U_b = np.array(U_a_list), np.array(U_b_list)
		self.b_target_array = np.array(b_target_list)

		# update rhs of SE with the term containing f
		b_array += np.array(b_f_list)
		self.b_array = b_array

		# quadratic and cubic polinomials make the size of nodal basis greater
		if self.degree > 1:
			self.mesh_size = np.shape(b_target.get_local())[0]

		end_time = time.perf_counter()

		logging.info(f'invariants computed in {end_time - start_time} s\n')

		return 0		

	def subproblem_solve(self, y_old, p_old):

		"""

		Parameters
		----------
		y_old : dict[int, dolfin.function.function.Function]
			keys from 0 to num_steps, 
			values are the previous iterate state functions at key-th timestep.
		p_old : dict[int, dolfin.function.function.Function]
			keys from 0 to num_steps, 
			values are the previous iterate adjoint state functions at key-th timestep.

		Returns
		-------
		tuple
			tuple of dict[int, dolfin.function.function.Function]
			corresponding to solution triple to subproblem if successful,
			tuple of None otherwise.

		"""

		# time at the begininng of primal dual strategy 
		start_time = time.perf_counter()

		# Define variational problem
		y = TrialFunction(self.V)
		# p = TrialFunction(self.V)
		v = TestFunction(self.V)

		start_time = time.perf_counter()

		# initialize lists
		A_sp_list, D_sp_list, L_1_list, L_2_list = [], [], [], []

		for k in range(self.num_steps + 1):

			# primal equation is , for all v in V
			a_0 = self.dt*self.mu*self.csi(y_old[k])*							inner(grad(y), 		grad(v))*dx
			a_1 = self.dt*self.mu*self.csi_p(y_old[k])*		y*					inner(grad(y_old[k]), grad(v))*dx

			# adjoint equation is , for all v in V
			a_2 = self.dt*self.mu*self.csi_p(y_old[k])*		p_old[k]*			inner(grad(y),		grad(v))*dx
			a_3 = self.dt*self.mu*self.csi_p(y_old[k])*		y*					inner(grad(p_old[k]), grad(v))*dx
			a_4 = self.dt*self.mu*self.csi_pp(y_old[k])*	p_old[k]*y*			inner(grad(y_old[k]), grad(v))*dx

			# assembling FENICS matices for state and adjoint equation
			A, A_adj = assemble(a_0 + a_1), assemble(a_2 + a_3 + a_4)

			# apply dirichlet boundary conditions
			if self.dirichlet_boundary is not None:
				# primal
				self.bc = DirichletBC(self.V, self.y_D, self.dirichlet_boundary)
				self.bc.apply(A)
				# adjoint
				self.bc_adj = DirichletBC(self.V, self.p_D, self.dirichlet_boundary)
				self.bc.apply(A_adj)

			# convert to sparse matrices
			A_mat, A_adj_mat = as_backend_type(A).mat(), as_backend_type(A_adj).mat()

			A_sp, A_adj_sp = sps.csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size), sps.csr_matrix(A_adj_mat.getValuesCSR()[::-1], shape = A_adj_mat.size)

			# load vectors
			el_1 = self.dt*self.mu*self.csi_p(y_old[k])*	y_old[k]*			inner(grad(y_old[k]), grad(v))*dx
			el_2 = self.dt*self.mu*self.csi_p(y_old[k])*	y_old[k]*			inner(grad(p_old[k]), grad(v))*dx
			el_3 = self.dt*self.mu*self.csi_p(y_old[k])*	p_old[k]*			inner(grad(y_old[k]), grad(v))*dx  
			el_4 = self.dt*self.mu*self.csi_pp(y_old[k])*	y_old[k]*p_old[k]*	inner(grad(y_old[k]), grad(v))*dx
			
			L_1, L_2 = assemble(el_1), assemble(el_2 + el_3 + el_4)

			# apply dirichlet boundary conditions
			if self.dirichlet_boundary is not None:
				# primal
				self.bc.apply(L_1)
				# adjoint
				self.bc_adj.apply(L_2)

			# list appending
			A_sp_list.append(A_sp)

			for L_1_entry, L_2_entry in zip(L_1, L_2):
				# interested in every timestep but the initial one
				if k > 0:
					L_1_list.append(L_1_entry)
				# interested in every timestep but the final one
				if k < self.num_steps:
					L_2_list.append(L_2_entry)

			if k == self.num_steps:
				continue

			# interested in every timestep but the final one
			D_sp_list.append(- self.dt*self.M_y_sp + A_adj_sp)

		# convert to arrays
		L_1_array = np.array(L_1_list)
		L_2_array = np.array(L_2_list)

		# interested in all blocks but the one at initial final time
		D_sp_blocks = sps.block_diag([D_sp_list[i]	for i in range(1, self.num_steps)], format='csc')
		D_sp_blocks	= sps.bmat(
			[ 	[None, np.zeros((self.mesh_size, self.mesh_size))], 
				[D_sp_blocks, None]	],
			format='csc')

		# assemble block matrix for lhs of SE: all times but initial on diagonal
		A_sp_blocks	= sps.bmat(	
			[ [	self.M_y_sp + A_sp_list[i + 1] if i == j 
				else - self.M_y_sp if i - j == 1
				else None 
				for j in range(self.num_steps) ]
				for i in range(self.num_steps) ],
			format='csc')

		# assemble block matrix for lhs of AE: all times but final on diagonal
		A_sp_adj_blocks	= sps.bmat(	
			[ [	self.M_y_sp + A_sp_list[i] if i == j 
				else - self.M_y_sp if j - i == 1
				else None 
				for j in range(self.num_steps) ]
				for i in range(self.num_steps) ],
			format='csc')

		# compute the term depending on y_0 in the rhs os adjoint equation
		b_adj_y_0 = -D_sp_list[0].toarray().dot(self.y_init)
		b_adj_y_0_array = np.hstack((b_adj_y_0, np.zeros((self.num_steps-1) * self.mesh_size)))
		
		# size of block matrices
		time_size = self.mesh_size*self.num_steps

		proj = np.zeros((time_size))

		# primal dual active set looping
		it = 0
		while True:
			# compute active indices matrices
			X_a, X_b = self.compute_active_sets(proj)
			# logging.info(f'active sets computed, iteration: {it}')
			# logging.info(f'non zero count: lower -> {sps.csr_matrix.count_nonzero(X_a)}, upper -> {sps.csr_matrix.count_nonzero(X_b)}')

			if it > 1:
				# output the count of the indices that changed
				logging.info(f'active sets differ from previous by: lower - > {self.compare_active_sets(X_a , X_a_old)}, upper -> {self.compare_active_sets(X_b , X_b_old)}')
				# stopping criterion
				if self.same_active_sets(X_a, X_a_old) and self.same_active_sets(X_b, X_b_old):
					# if the condition is met we've reached optimality
					logging.info('CONVERGENCE REACHED IN SUBPROBLEM :)')
					break

				# RHS of projection formula
				res = X_a.dot(self.U_a) + X_b.dot(self.U_b) + I_a_b.dot(proj) - u_vec
				# convert to dictionary of fcts
				res_func = self.to_dict_of_functions(np.hstack((res, np.zeros(self.mesh_size))))
				# initialize
				res_L_2 = 0
				res_L_inf = []
				for n in range(self.num_steps):

					if self.control_type == 'neumann_boundary':
						res_L_2 += self.dt * norm(res_func[n], mesh=self.boundary_mesh) ** 2
						res_L_inf.append(norm(res_func[n].vector(), 'linf'))
					else:
						res_L_2 += self.dt * norm(res_func[n]) ** 2
						res_L_inf.append(norm(res_func[n].vector(), 'linf'))
				# L2 and Linf errors
				res_L_2 = np.sqrt(res_L_2)
				res_L_inf = np.max(res_L_inf)	
		
				# imprecise stopping criterion
				if np.max([res_L_inf, res_L_2]) < 1e-8:
					# if the condition is met we've reached optimality
					logging.info('CONVERGENCE(tol) REACHED IN SUBPROBLEM')
					break

			if it > self.max_it_pd:
				logging.error('NO CONVERGENCE REACHED IN SUBPROBLEM :(')
				break

			# inactive indices matrix
			I_a_b = sps.identity(time_size) - X_a - X_b		

			# RHS of projection formula
			b_control_array	= X_a.dot(self.U_a) + X_b.dot(self.U_b)

			# assemble ( (3 x size x num_steps) x (3 x size x num_steps) ) sparse lhs of the linear system by blocks
			Global_matrix = sps.bmat(
				[ 	[None,								A_sp_blocks,	- self.B_sp_blocks 		],
					[A_sp_adj_blocks,					D_sp_blocks,	None					],
					[I_a_b.dot(self.M_proj_sp_blocks),	None,			sps.identity(time_size)	]	],
				format='csc')

			# assemble (3 x size x num_steps) dense rhs of linear system 
			Global_right_term = np.hstack(	
				[	self.b_array + L_1_array,
					b_adj_y_0_array + self.b_target_array + L_2_array,
					b_control_array	]	)

			# time0 = time.perf_counter()
			# print(f'b_control_array, nonzero = {np.count_nonzero(b_control_array)}')
			# print(f'I_a_b, nonzero = {I_a_b.count_nonzero()}')

			# spilu = spla.spilu(Global_matrix, drop_tol=1e-8)

			# # spilu_approx = spla.spilu(Global_matrix)

			# # if it > 1:
			# # 	# self.spy_sparse(Global_matrix)
			# # 	self.spy_sparse(spilu.L @ spilu.U)
			# # 	return None, None, None

			# print(f'approx inverse, nonzero = {spilu.nnz}')

			# time1 = time.perf_counter()

			# logging.info(f'incomplete LU factorization in: {time1 - time0} s')

			# sol_sp = spilu.solve(Global_right_term)

			# # compute solution of sparse linear system	
			# time2 = time.perf_counter()
			# logging.info(f'sparse system of size {np.shape(sol_sp)[0]} x {np.shape(sol_sp)[0]} solved in: {time2 - time1} s')

			# compute solution of sparse linear system	
			time1 = time.perf_counter()
			sol_sp = spla.spsolve(Global_matrix, Global_right_term)
			time2 = time.perf_counter()
			logging.info(f'sparse system of size {np.shape(sol_sp)[0]} x {np.shape(sol_sp)[0]} solved in: {time2 - time1} s')

			# split to vector of state, control and adjoint state and store it adding initial and final known values
			p_vec = sol_sp[ 			: time_size   ]
			y_vec = sol_sp[ time_size 	: 2*time_size ] 
			u_vec = sol_sp[ 2*time_size :			  ]

			proj  = - self.M_proj_sp_blocks.dot(np.copy(p_vec))
			
			# update old active indices matrices
			X_a_old = X_a.copy()
			X_b_old = X_b.copy()

			it += 1

		#########################################

		control = np.copy(u_vec)

		# split to vector of state, control and adjoint state and store it adding initial and final known values
		p_vec = np.hstack( ( p_vec,	self.p_final ) )
		y_vec = np.hstack( ( self.y_init, y_vec	 ) )	
		u_vec = np.hstack( ( u_vec, self.u_final ) )

		# we store also a dictionary of functions, will be useful for visualization and computation of errors
		# initialization
		y_t = {n : Function(self.V) for n in range(self.num_steps + 1)}						
		u_t = {n : Function(self.V) for n in range(self.num_steps + 1)}			
		p_t = {n : Function(self.V) for n in range(self.num_steps + 1)}
		
		for n in range(self.num_steps + 1):

			for solution, additional, solution_vec in zip(
				(y_t, u_t, p_t), 
				(Function(self.V), Function(self.V), Function(self.V)),
				(y_vec, u_vec, p_vec)	):

				if n < self.num_steps:
					additional.vector().set_local(solution_vec[n * self.mesh_size : (n + 1) * self.mesh_size])
				else:
					additional.vector().set_local(solution_vec[n * self.mesh_size :					  ])

				solution[n].assign(additional)

		# time at end of primal dual strategy
		end_time = time.perf_counter()

		logging.info(f'SUBPROBLEM SOLVED in {end_time - start_time} s and {it} iterations')

		if it > self.max_it_pd:
			# store computed triple anyway
			self.y, self.u, self.p = y_t, u_t, p_t
			return None, None, None

		if self.compute_proj_residuals_flag:

			# compute associated adjoint state
			p_y_u = self.compute_associated_adjoint(control)

			# compute the projection residual
			res = self.compute_proj_residuals(	
				b_control_array,
				I_a_b,
				- self.M_proj_sp_blocks.dot(p_y_u),
				control)

			# store the projection residual
			self.proj_res_list.append(res)

			logging.info(f'projection residual: {res}')

		return y_t, u_t, p_t

	def sqp_solve(self):

		"""

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		logging.info(f'size of mesh is {self.mesh_size} with {self.num_steps} timesteps')
		logging.info(f'maximum number of iterations for SQP is: {self.max_it_sqp}, with tolerance: {self.tol}')
		logging.info(f'maximum number of iterations for active set strategy is: {self.max_it_pd}')

		# set initial conditions vectors
		self.y_init = interpolate(self.y_0, 	self.V).vector().get_local()
		self.p_final = interpolate(self.p_end, 	self.V).vector().get_local()

		# initial guess
		init = interpolate(Constant(0.0), self.V)

		y_old = {n : init for n in range(self.num_steps + 1)}
		u_old = {n : init for n in range(self.num_steps + 1)}
		p_old = {n : init for n in range(self.num_steps + 1)}

		start_time_sqp = time.perf_counter()

		self.compute_invariants()

		if self.compute_proj_residuals_flag:
			# initialize list to store projection residuals
			self.proj_res_list = []
		# initialize list to store increments
		self.incr_list = []
		# self.cost_func_list = []
		self.error_sequence_list = []
		self.error_sequence_list.append(self.compute_inf_errors(y_old, u_old, p_old))

		it = 0
		while True:

			logging.info(f'solving subproblem, iteration {it}')
			# call subproblem solver
			y, u, p = self.subproblem_solve(y_old, p_old)

			# check if subproblem was solved
			if y is None and u is None and p is None:
				logging.error(f'subproblem {it} not solved')
				return 1
	
			# compute increment
			state_diff_vec, control_diff_vec, adjoint_diff_vec = np.zeros(self.num_steps + 1), np.zeros(self.num_steps + 1), np.zeros(self.num_steps + 1)
			for n in range(self.num_steps + 1):
				state_diff_vec[n] = norm( y[n].vector() - y_old[n].vector(), 'linf')
				control_diff_vec[n] = norm( u[n].vector() - u_old[n].vector(), 'linf')
				adjoint_diff_vec[n] = norm( p[n].vector() - p_old[n].vector(), 'linf')

			max_state_diff, max_control_diff, max_adjoint_diff = np.max(state_diff_vec), np.max(control_diff_vec), np.max(adjoint_diff_vec)
			incr = max_state_diff + max_control_diff + max_adjoint_diff
			# store increment
			self.incr_list.append(incr)
			self.error_sequence_list.append(self.compute_inf_errors(y, u, p))
			logging.info(f'increment: {incr}\n')

			# stopping criterion
			if it > 1 and incr < self.tol:
				logging.info('SQP stopping criterion met')
				break

			elif it > self.max_it_sqp:
				# store computed triple anyway
				self.y, self.u, self.p = y, u, p
				logging.error(f'NO CONVERGENCE REACHED: maximum number of iterations {self.max_it_sqp} for SQP method excedeed')
				return 1

			y_old = {n : Function(self.V) for n in range(self.num_steps + 1)}						
			u_old = {n : Function(self.V) for n in range(self.num_steps + 1)}			
			p_old = {n : Function(self.V) for n in range(self.num_steps + 1)}

			# update previous iterates
			for n in range(self.num_steps + 1):
				y_old[n].assign(y[n])
				u_old[n].assign(u[n])
				p_old[n].assign(p[n])

			it += 1

		# store optimal triple
		self.y, self.u, self.p = y, u, p

		if self.control_type == 'time':
			# store the values of -1/lam * B* p for every timestep
			gradient_term = {}
			for i in range(self.num_steps + 1):
				gradient_term[i] = - self.P @ self.p[i].vector().get_local()

			self.gradient_term = gradient_term	

		logging.info('--------------------------SQP CONVERGENCE REACHED--------------------------\n')

		# output time at end of while loop
		end_time_sqp = time.perf_counter()
		logging.info(f'total elapsed time: {end_time_sqp - start_time_sqp}')

		return 0

	def compute_LSE_terms(self, y_old):

		"""

		Parameters
		----------
		y_old : dict[int, dolfin.function.function.Function]
			keys from 0 to num_steps, 
			values are the previous iterate state functions at key-th timestep.
		
		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		# Define variational problem
		y = TrialFunction(self.V)
		# p = TrialFunction(self.V)
		v = TestFunction( self.V)

		# initialize lists
		A_sp_list = []
		L_1_list = []

		for k in range(self.num_steps + 1):

			# primal equation is , for all v in V
			a_0 = self.dt*self.mu*self.csi(y_old[k])*		inner(grad(y), 			grad(v))*dx
			a_1 = self.dt*self.mu*self.csi_p(y_old[k])*y*	inner(grad(y_old[k]), 	grad(v))*dx

			# assembling FENICS matices for state and adjoint equation
			A = assemble(a_0 + a_1)

			# apply dirichlet boundary conditions
			if not self.dirichlet_boundary is None:
				# primal
				self.bc = DirichletBC(
					self.V, 
					self.y_D, 
					self.dirichlet_boundary)
				self.bc.apply(A)

			# convert to sparse matrices
			A_mat = as_backend_type(A).mat()

			A_sp = sps.csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size)
			
			# list appending
			A_sp_list.append(A_sp.copy())

			# load vector
			el_1 = self.dt*self.mu*self.csi_p(y_old[k])*y_old[k]*inner(grad(y_old[k]), grad(v))*dx
			L_1 = assemble(el_1)

			if not self.dirichlet_boundary is None:
				self.bc.apply(L_1)

			# interested in every timestep but the initial one
			if k > 0:
				L_1_list += list(L_1)

		# assemble block matrix for lhs of SE: all times but initial on diagonal
		A_sp_blocks = sps.bmat(	
			[ [	self.M_y_sp + A_sp_list[i + 1] if i == j 
				else - self.M_y_sp if i - j == 1
				else None 
				for j in range(self.num_steps) ]
				for i in range(self.num_steps) ],
			format='csc')

		# assemble block matrix for lhs of AE: all times but final on diagonal
		A_sp_adj_blocks = sps.bmat(	
			[ [	self.M_y_sp + A_sp_list[i] if i == j 
				else - self.M_y_sp if j - i == 1
				else None 
				for j in range(self.num_steps) ]
				for i in range(self.num_steps) ],
			format='csc')
		
		# linearization term needed in the rhs of LSE 
		L_1_array = np.array(L_1_list)

		return A_sp_blocks, A_sp_adj_blocks, L_1_array

	def assemble_LSE_rhs(self, u_vec, rhs_lin_term):

		"""

		Parameters
		----------
		u_vec : numpy.ndarray
			Array of shape (mesh_size * num_steps,).
		rhs_lin_term : numpy.ndarray
			Array of shape (mesh_size * num_steps,),
			corresponding to the linearization term in the RHS.

		Returns
		-------
		numpy.ndarray
			Array of shape (mesh_size * num_steps,),
			corresponding to the linearized state equation RHS (calculated with u_vec).

		"""

		return self.B_sp_blocks.dot(u_vec) + self.b_array + rhs_lin_term

	def assemble_AE_rhs(self, y_vec):

		"""

		Parameters
		----------
		y_vec : numpy.ndarray
			Array of shape (mesh_size * num_steps,).

		Returns
		-------
		numpy.ndarray
			Array of shape (mesh_size * num_steps,),
			corresponding to the state adjoint equation RHS (calculated with y_vec).

		"""

		M_sub_diag_sp_blocks = sps.block_diag([self.dt*self.M_y_sp	for i in range(1, self.num_steps)], format='csc')
		M_sub_diag_sp_blocks = sps.bmat(
			[ 	[None, np.zeros((self.mesh_size, self.mesh_size))],
				[M_sub_diag_sp_blocks, None] ])

		M_y_0_term = np.hstack( ( 
			self.dt*self.M_y_sp.dot(self.y_init), 
			np.zeros(self.mesh_size*(self.num_steps - 1)) ) )

		return 	M_sub_diag_sp_blocks.dot(y_vec) + M_y_0_term + self.b_target_array

	
	def compute_associated_adjoint(self, u_vec):

		""" Computes the adjoint state associated to the control.

		Parameters
		----------
		u_t : dict[int, dolfin.function.function.Function]
			keys from 0 to num_steps, 
			values are the control functions at key-th timestep.
				
		Returns
		-------
		numpy.ndarray
			Array of shape (mesh_size * num_steps,),
			correponding to the associated adjoint state.

		"""

		# initialization
		init = interpolate(Constant(0.0), self.V)

		y_old = {n : init for n in range(self.num_steps + 1)}
		p_old = {n : init for n in range(self.num_steps + 1)}

		start_time_nm = time.perf_counter()

		it = 0
		while True:

			# compute the lhs terms, and the rhs linearization term
			lhs, lhs_adj, rhs_lin_term = self.compute_LSE_terms(y_old)
			# assemble rhs of the linearized SE
			rhs = self.assemble_LSE_rhs(u_vec, rhs_lin_term)
			# solve for state
			y_vec = spla.spsolve(lhs, rhs)
			# convert to dictioary
			y = self.to_dict_of_functions(np.hstack((self.y_init, y_vec)))

			# compute L-infinity norm
			state_diff_vec = np.zeros(self.num_steps + 1)

			for n in range(self.num_steps + 1):

				state_diff_vec[n] = norm( y[n].vector() - y_old[n].vector(), 'linf')

			incr = np.max(state_diff_vec)

			# stopping criterion
			if it > 1 and incr < self.tol:
				break

			# update previous iterates
			y_old = {n : Function(self.V) for n in range(self.num_steps + 1)}

			for n in range(self.num_steps + 1):
				y_old[n].assign(y[n])

			it += 1
		# assemble rhs of the AE
		rhs_adj = self.assemble_AE_rhs(y_vec)
		# solve for associated adjoint state
		p_vec = spla.spsolve(lhs_adj, rhs_adj)

		logging.info(f'associated adjoint state computed in {time.perf_counter() - start_time_nm} and {it-1} Newton iterations')

		return p_vec

	def compute_proj_residuals(self, active, inactive_indices, proj, control):

		"""

		Parameters
		----------
		active : numpy.ndarray
			Array of shape (mesh_size * num_steps,),
			it is given by the lower active indices X_a matrix applied to the lower constraint array U_a
			plus the upper active indices matrix X_b applied to the upper constraint array U_b.			
		inactive_indices : scipy.sparse.csr.csr_matrix
			Diagonal matrix of shape (mesh_size * num_steps, mesh_size * num_steps),
			it is the difference between the identity and the sum of the active indices matrices.
		proj : numpy.ndarray
			Array of shape (mesh_size * num_steps,),
			space and time discretization of element to project in the admissible set,
			it is computed from p, the associated state to the control solution of the subproblem u.
		control : numpy.ndarray
			Array of shape (mesh_size * num_steps,),
			space and time discretization of the control solution to the subproblem.

		Returns
		-------
		float
			L2 norm of the projection residual.

		"""

		# compute array of projecion residuals
		res_vec = active + inactive_indices.dot(proj) - control

		res_func = self.to_dict_of_functions(np.hstack((res_vec, np.zeros(self.mesh_size))))

		res_L_2 = 0
		for n in range(self.num_steps):

			if self.control_type == 'neumann_boundary':
				res_L_2 += self.dt * norm(res_func[n], mesh=self.boundary_mesh) ** 2
			else:
				res_L_2 += self.dt * norm(res_func[n]) ** 2

		res_L_2 = np.sqrt(res_L_2)
		
		return res_L_2

	def plot_inf_errors(self, path):

		"""

		Parameters
		----------
		path : str
			Identifier of the directory where to save the residual plots.

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		# plotting
		plt.figure()
		fig, ax = plt.subplots(1)
		
		error_array = np.array(self.error_sequence_list)

		its = np.array(range(np.shape(error_array)[0]))

		ax.semilogy(
			its, 
			error_array[:,0], 
			marker='o', 
			color='r', 
			linewidth=0.7, 
			label=r'$||\, y-y_h \,||_{L^\infty}$', 
			markerfacecolor='none', 
			markeredgecolor='r')

		ax.semilogy(
			its, 
			error_array[:,1], 
			marker='s', 
			color='b', 
			linewidth=0.7, 
			label=r'$||\, u-u_h \,||_{L^\infty}$', 
			markerfacecolor='none', 
			markeredgecolor='b')

		ax.semilogy(
			its, 
			error_array[:,2], 
			marker='D', 
			color='lime', 
			linewidth=0.7, 
			label=r'$||\, p-p_h \,||_{L^\infty}$', 
			markerfacecolor='none', 
			markeredgecolor='lime')		

		ax.set_title(fr'$N_T =${self.num_steps}, $N_h = ${int(self.mesh_size)}')
		ax.set(xlabel=r'SQP iteration $n$', ylabel=r'$L^\infty$-error')
		ax.legend(loc="lower left")

		plt.setp(ax, xticks=its)

		fig.set_size_inches(6, 6)

		plt.savefig(path + '/inf_errors.pdf')

		return 0

	def plot_residuals(self, path):

		"""

		Parameters
		----------
		path : str
			Identifier of the directory where to save the residual plots.

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		# plotting
		plt.figure()
		fig, ax = plt.subplots(1)
		if self.compute_proj_residuals_flag:
			res_array = np.array(self.proj_res_list)

		its = np.array(range(1, 1 + np.shape(np.array(self.incr_list))[0]))

		ax.semilogy(
			its, 
			self.incr_list, 
			marker='o', 
			color='r', 
			linewidth=0.7, 
			label=r'incr$_n$', 
			markerfacecolor='none', 
			markeredgecolor='r')		
		# ax.semilogy( its, self.cost_func_list, 		marker='D', color='lime', linewidth=0.7, label=r'J(y_n,\,u_n)$', markerfacecolor='none', markeredgecolor='lime')
		
		if self.compute_proj_residuals_flag:
			ax.semilogy( 
				its, 
				res_array, 
				marker='s', 
				color='b', 
				linewidth=0.7, 
				label=r'res$_{L^2} (u_n)$', 
				markerfacecolor='none', 
				markeredgecolor='b')

		ax.set_title(fr'increments and projection residuals $N_T =${self.num_steps}, $N_h = ${int(self.mesh_size)}')
		ax.set(xlabel=r'SQP iteration $n$')
		ax.legend(loc="lower left")

		plt.setp( ax, xticks=its )

		fig.set_size_inches(6, 6)

		plt.savefig(path + '/residuals.pdf')

		return 0


######################################### DISTRIBUTED CONTROL TEST EXAMPLLES ##########################################

def example_1D(lam, mesh_size, num_steps):
	## python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	g = Constant(0.0)

	T 		= 1.0
	beta 	= Constant( 1.0 )
	y_0   	= Expression(' sin(pi*x[0])', degree=5)
	p_end 	= Constant(0.0) 

	y_target= Expression(' cos(2*pi*t) * sin(pi*x[0]) * (1 - 2*lam*pi) - lam*pow(pi,2) * sin(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) - sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) + lam * pow(pi,2)*sin(2*pi*t)*cos(2*pi*t)*pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) * (   pow(cos(pi*x[0]),2)* ( sin(pi*x[0])*cos(2*pi*t)* ( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) - 1 ) - pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) - 1 ) + pow(sin(pi*x[0]), 2)*( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) + 1 )	) * pow( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) + 1, -3) ', t=0, degree=5, lam=lam)

	y_exact = Expression(' 		 		cos(2*pi*t) * sin(pi*x[0])', t=0, degree=5)
	p_exact = Expression('		 - lam* sin(2*pi*t) * sin(pi*x[0])', t=0, degree=5, lam=1e-2)
	
	# space and time dependant constraints
	u_b = Expression(' 2 * (0.2 + 0.5 * x[0]) * t', t=0, degree=2)
	u_a = Expression(' - 2 * (0.2 + 0.5 * x[0]) * (1 - t)', t=0, degree=2)

	u_exact = Expression(
		'''( sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ?
			 sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ) < 2 * t * (0.2 + 0.5 * x[0]) ?
			 (sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ?
			  sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t)) : 2 * t * (0.2 + 0.5 * x[0])''',
		t=0,
		degree=5)
	
	f = Expression(
		'''	- pow(pi,2) * cos(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1), cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) 
			- sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) 
			- 2*pi*sin(2*pi*t) * sin(pi*x[0])
			- ((sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ?
				sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t)) < 2 * t * (0.2 + 0.5 * x[0]) ?
		 		(sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ?
		  		sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t)) : 2 * t * (0.2 + 0.5 * x[0]))''',
		t=0,
		degree=5)

	g = Constant(0.0)

	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-08, 1e-10, 20, 20
	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	total_time_start = time.perf_counter()

	# build mesh
	# 1D
	new_mesh = IntervalMesh(mesh_size, 0.0, 1.0)
	# initialize instance
	P = Quasi_Linear_Problem_Box(T, num_steps, 'distributed')
	# set up problem by callig class attrubutes
	P.set_state_space(new_mesh, 1)
	P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary)
	P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), None)
	# P.set_control_space(new_mesh, 1)
	P.set_control_constraints(u_a, u_b)
	P.set_cost(lam, y_target)
	P.set_state_equation(beta, f, g, y_0, p_end)
	P.set_exact_solution(y_exact, u_exact, p_exact)
	# set quasi-linear parameters
	P.set_non_linearity(mu, csi, csi_p, csi_pp)
	P.set_maxit_and_tol(tol, tol_pd, max_it_sqp, max_it_pd)
	# decide if to compute projection residuals
	P.compute_proj_residuals_flag = True
	# call solver
	P.sqp_solve()
	# compute errors
	P.visualize_1D(0, 1, 128, 'visualization_sqp')
	P.visualize_1D_exact(0, 1, 128, 'visualization_sqp')

	P.visualize_paraview('visualization_sqp/paraview/1D')

	P.compute_errors()

	# if P.compute_proj_residuals_flag:
	P.plot_residuals('visualization_sqp')
	P.plot_inf_errors('visualization_sqp')
	# P.plot_objective('visualization_sqp')

	print('inf_errors\n',P.error_sequence_list)
	print('increments\n',P.incr_list)
	print('orders\n',orders.compute_EOC_orders(P.incr_list))


	logging.info(fr'computed order of convergence is $q =${math.log(P.incr_list[-1]/P.incr_list[-2]) / math.log(P.incr_list[-2]/P.incr_list[-3])}')

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

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
	
	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-8, 1e-10, 20, 20

	T 		= 1.0
	beta 	= Constant( 1.0 )
	y_0   	= Expression(
		'sin(pi*x[0])*sin(pi*x[1])',
		degree=5)

	p_end 	= Constant(0.0) 

	y_target= Expression(
		'''	cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) * (1 - 2*lam*pi)
			+ 2*lam*pow(pi,2)*sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) 
			* ( pow( pow(exp(1), -cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) ) + 1, -1) + 1 ) 
			- lam*pow(pi,2)*sin(2*pi*t)*cos(2*pi*t)*(pow(sin(pi*x[0])*cos(pi*x[1]),2) + pow(cos(pi*x[0])*sin(pi*x[1]),2))
			*(
				2*pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) 
				* pow(pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) + 1, -2) 
				+ cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])
				* pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) 
				* ( pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) - 1 ) 
				* pow( pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1]) ) + 1, -3)
			)
			+ 2*lam*pow(pi,2)*cos(2*pi*t)*sin(2*pi*t) * pow(sin(pi*x[0]),2)*pow(sin(pi*x[1]),2)
			* pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) * pow(pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) + 1, -2)
		''',
		t=0,
		degree=5,
		lam=1e-2)

	y_exact = Expression(
		'cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])',
		t=0,
		degree=5)

	p_exact = Expression(
		'- lam* sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])',
		t=0,
		degree=5,
		lam=1e-2)
	
	# time dependant constraints
	u_b = Expression(
		'2 * t',
		t=0,
		degree=2)
	u_a = Expression(
		'-2 * (1 - t)',
		t=0,
		degree=2)

	u_exact = Expression(
		'''( sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) > - 2 * (1 - t) ?
			 sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) : - 2 * (1 - t) ) < 2 * t ?
			 (sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) > - 2 * (1 - t) ?
			  sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) : - 2 * (1 - t)) : 2 * t''',
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
			-(	(sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) > - 2 * (1 - t) ?
				sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) : - 2 * (1 - t)) < 2 * t ?
			 	(sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) > - 2 * (1 - t) ?
			  	sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) : - 2 * (1 - t)) : 2 * t)
		''',
		t=0,
		degree=5)

	g = Constant(0.0)

	total_time_start = time.perf_counter()

	# build mesh
	# 2D
	new_mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), mesh_size, mesh_size)
	# initialize instance
	P = Quasi_Linear_Problem_Box(T, num_steps, 'distributed')
	# set up problem by callig class attrubutes
	P.set_state_space(new_mesh, 1)
	P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary)
	P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), None)
	# P.set_control_space(new_mesh, 1)
	P.set_control_constraints(u_a, u_b)
	P.set_cost(lam, y_target)
	P.set_state_equation(beta, f, g, y_0, p_end)
	P.set_exact_solution(y_exact, u_exact, p_exact)
	# set quasi-linear parameters
	P.set_non_linearity(mu, csi, csi_p, csi_pp)
	P.set_maxit_and_tol(tol, tol_pd, max_it_sqp, max_it_pd)
	P.compute_proj_residuals_flag = True
	# call solver
	P.sqp_solve()
	# compute errors
	P.compute_errors()
	# visualization
	# P.visualize_paraview('visualization_sqp/paraview/2D/distributed')
	P.plot_inf_errors('visualization_sqp/2D/distributed')

	if P.compute_proj_residuals_flag:
		P.plot_residuals('visualization_sqp/2D/distributed')

	print('inf_errors\n',P.error_sequence_list)
	print('increments\n',P.incr_list)
	print('orders\n',orders.compute_EOC_orders(P.incr_list))

	# logging.info(fr'computed order of convergence is $q =${math.log(P.incr_list[-1]/P.incr_list[-2]) / math.log(P.incr_list[-2]/P.incr_list[-3])}')

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0

################################## NEUMANN CONTROL TEST EXAMPLES ###########################

def example_2D_N(lam, mesh_size, num_steps):

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
	
	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-10, 1e-10, 20, 20

	T 		= 1.0
	beta 	= Constant( 1.0 )
	y_0   	= Expression(
		'cos(pi*x[0])*cos(pi*x[1])',
		degree=5)

	p_end 	= Constant(0.0) 

	y_target= Expression(
		'''	cos(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) * (1 - 2*lam*pi)
			+ 2*lam*pow(pi,2)*sin(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) 
			* ( pow( pow(exp(1), -cos(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) ) + 1, -1) + 1 ) 
			- lam*pow(pi,2)*sin(2*pi*t)*cos(2*pi*t)*(pow(sin(pi*x[0])*cos(pi*x[1]),2) + pow(cos(pi*x[0])*sin(pi*x[1]),2))
			*(
				2*pow(exp(1),cos(2*pi*t)*cos(pi*x[0])*cos(pi*x[1])) 
				* pow(pow(exp(1),cos(2*pi*t)*cos(pi*x[0])*cos(pi*x[1])) + 1, -2) 
				+ cos(2*pi*t)*cos(pi*x[0])*cos(pi*x[1])
				* pow(exp(1),cos(2*pi*t)*cos(pi*x[0])*cos(pi*x[1])) 
				* ( pow(exp(1),cos(2*pi*t)*cos(pi*x[0])*cos(pi*x[1])) - 1 ) 
				* pow( pow(exp(1),cos(2*pi*t)*cos(pi*x[0])*cos(pi*x[1]) ) + 1, -3)
			)
			+ 2*lam*pow(pi,2)*cos(2*pi*t)*sin(2*pi*t) * pow(cos(pi*x[0]),2)*pow(cos(pi*x[1]),2)
			* pow(exp(1),cos(2*pi*t)*cos(pi*x[0])*cos(pi*x[1])) * pow(pow(exp(1),cos(2*pi*t)*cos(pi*x[0])*cos(pi*x[1])) + 1, -2)
		''',
		t=0,
		degree=5,
		lam=1e-2)

	y_exact = Expression(
		'cos(2*pi*t) * cos(pi*x[0])*cos(pi*x[1])',
		t=0,
		degree=5)

	p_exact = Expression(
		'- lam* sin(2*pi*t) * cos(pi*x[0])*cos(pi*x[1])',
		t=0,
		degree=5,
		lam=1e-2)
	
	# space dependant constraints
	u_b = Expression(
		'(0.2 + 0.5 * x[0]) * (0.2 + 0.5 * x[1])',
		t=0,
		degree=2)
	u_a = Expression(
		'- (0.8 - 0.5 * x[0]) * (0.8 - 0.5 * x[1])',
		t=0,
		degree=2)

	u_exact = Expression(
		'''( sin(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) > - (0.8 - 0.5 * x[0]) * (0.8 - 0.5 * x[1])  ?
			 sin(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) : - (0.8 - 0.5 * x[0]) * (0.8 - 0.5 * x[1])  ) < (0.2 + 0.5 * x[0]) * (0.2 + 0.5 * x[1]) ?
			 (sin(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) > - (0.8 - 0.5 * x[0]) * (0.8 - 0.5 * x[1])  ?
			  sin(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) : - (0.8 - 0.5 * x[0]) * (0.8 - 0.5 * x[1]) ) : (0.2 + 0.5 * x[0]) * (0.2 + 0.5 * x[1])''',
		t=0,
		degree=5)
	
	f = Expression(
		'''	- pow(pi,2) * cos(2*pi*t) 
			*( 
				cos(2*pi*t) * (pow(sin(pi*x[0])*cos(pi*x[1]),2) + pow(cos(pi*x[0])*sin(pi*x[1]),2)) 
				* pow(exp(1), cos(2*pi*t) * cos(pi*x[0])*cos(pi*x[1])) 
				* pow(pow(exp(1),cos(2*pi*t) * cos(pi*x[0])*cos(pi*x[1])) + 1, -2) 
				- 2*cos(pi*x[0])*cos(pi*x[1]) 
				* ( pow( pow(exp(1), -cos(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) ) + 1, -1) + 1 )
			) 
			- 2*pi*sin(2*pi*t)*cos(pi*x[0])*cos(pi*x[1])
		''',
		t=0,
		degree=5)

	g = Expression(
		'''	-((sin(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) > - (0.8 - 0.5 * x[0]) * (0.8 - 0.5 * x[1])  ?
			sin(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) : - (0.8 - 0.5 * x[0]) * (0.8 - 0.5 * x[1]) ) < (0.2 + 0.5 * x[0]) * (0.2 + 0.5 * x[1]) ?
		 	(sin(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) > - (0.8 - 0.5 * x[0]) * (0.8 - 0.5 * x[1])  ?
		  	sin(2*pi*t) * cos(pi*x[0])*cos(pi*x[1]) : - (0.8 - 0.5 * x[0]) * (0.8 - 0.5 * x[1]) ) : (0.2 + 0.5 * x[0]) * (0.2 + 0.5 * x[1]))''',
		t=0,
		degree=5)

	total_time_start = time.perf_counter()

	# build mesh
	# 2D
	new_mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), mesh_size, mesh_size)
	# initialize instance
	P = Quasi_Linear_Problem_Box(T, num_steps, 'neumann boundary')
	# set up problem by callig class attrubutes
	P.set_state_space(new_mesh, 1)
	P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), None)
	P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), boundary)
	# P.set_control_space(new_mesh, 1)
	P.set_control_constraints(u_a, u_b)
	P.set_cost(lam, y_target)
	P.set_state_equation(beta, f, g, y_0, p_end)
	P.set_exact_solution(y_exact, u_exact, p_exact)
	# set quasi-linear parameters
	P.set_non_linearity(mu, csi, csi_p, csi_pp)
	P.set_maxit_and_tol(tol, tol_pd, max_it_sqp, max_it_pd)
	P.compute_proj_residuals_flag = True
	# call solver
	P.sqp_solve()
	# compute errors
	P.compute_errors()
	# visualization
	P.visualize_paraview('visualization_sqp/paraview/2D/neumann')
	P.plot_inf_errors('visualization_sqp/2D/neumann')

	if P.compute_proj_residuals_flag:
		P.plot_residuals('visualization_sqp/2D/neumann')

	print('inf_errors\n', P.error_sequence_list)
	print('increments\n', P.incr_list)
	print('orders\n',orders.compute_EOC_orders(P.incr_list))

	# logging.info(fr'computed order of convergence is $q =${math.log(P.incr_list[-1]/P.incr_list[-2]) / math.log(P.incr_list[-2]/P.incr_list[-3])}')

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0

######################## PURELY TIME DEP TEST EXAMPLES #######################

def example_1D_t(lam, mesh_size, num_steps):
	# python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	g = Constant(0.0)
	T = 1.0
	beta = Expression(
		'''	
			( ( x[0] >= pow(3,-1) ? 1 : 0 )*( x[0] <= 2*pow(3,-1) ? 1 : 0 ) )
		''',
		degree=5)

	y_0   	= Expression('sin(pi*x[0])', degree=5)
	p_end 	= Constant(0.0) 

	y_target= Expression(' cos(2*pi*t) * sin(pi*x[0]) * (1 - 2*lam*pi) - lam*pow(pi,2) * sin(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) - sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) + lam * pow(pi,2)*sin(2*pi*t)*cos(2*pi*t)*pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) * (   pow(cos(pi*x[0]),2)* ( sin(pi*x[0])*cos(2*pi*t)* ( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) - 1 ) - pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) - 1 ) + pow(sin(pi*x[0]), 2)*( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) + 1 )	) * pow( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) + 1, -3) ', t=0, degree=5, lam=lam)

	y_exact = Expression(' 		 		cos(2*pi*t) * sin(pi*x[0])', t=0, degree=5)
	p_exact = Expression('		 - lam* sin(2*pi*t) * sin(pi*x[0])', t=0, degree=5, lam=lam)
	
	# constant constraints
	u_b 	= Constant( 0.25 )
	u_a 	= Constant(-0.25 )

	u_exact = Expression(
		'''	
			( 2*pow(pi,-1) * sin(2*pi*t) > u_a ? 2*pow(pi,-1) * sin(2*pi*t) : u_a ) < u_b ? (2*pow(pi,-1) * sin(2*pi*t) > u_a ? 2*pow(pi,-1) * sin(2*pi*t) : u_a) : u_b
		''', 
		t=0, 
		u_a=u_a, 
		u_b=u_b, 
		degree=5)
	
	f = Expression(
		'''	- pow(pi,2) * cos(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1), cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) 
			- sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) 
			- 2*pi*sin(2*pi*t) * sin(pi*x[0])
			- ( x[0] >= pow(3,-1) ? 1 : 0 )*( x[0] <= 2*pow(3,-1) ? 1 : 0 ) *
				(( pow(pi,-1) * sin(2*pi*t) > u_a ?
				pow(pi,-1) * sin(2*pi*t) : u_a ) < u_b ?
				( pow(pi,-1) * sin(2*pi*t) > u_a ?
				pow(pi,-1) * sin(2*pi*t) : u_a ) : u_b ) 
		''', 
		t=0,
		u_a=u_a, 
		u_b=u_b,
		degree=5)

	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-10, 1e-10, 20, 20
	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	total_time_start = time.perf_counter()

	# build mesh
	# 1D
	new_mesh = IntervalMesh(mesh_size, 0.0, 1.0)
	# initialize instance
	P = Quasi_Linear_Problem_Box(T, num_steps, 'time')
	# set up problem by callig class attrubutes
	P.set_state_space(new_mesh, 1)
	P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary)
	P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), None)
	# P.set_control_space(new_mesh, 1)
	P.set_control_constraints(u_a, u_b)
	P.set_cost(lam, y_target)
	P.set_state_equation(beta, f, g, y_0, p_end)
	P.set_exact_solution(y_exact, u_exact, p_exact)
	# set quasi-linear parameters
	P.set_non_linearity(mu, csi, csi_p, csi_pp)
	P.set_maxit_and_tol(tol, tol_pd, max_it_sqp, max_it_pd)
	# decide if to compute projection residuals
	P.compute_proj_residuals_flag = True
	# call solver
	P.sqp_solve()
	# compute errors
	P.visualize_1D(0, 1, 128, 'visualization_sqp')
	P.visualize_1D_exact(0, 1, 128, 'visualization_sqp')

	P.visualize_paraview('visualization_sqp/paraview/1D')

	P.visualize_purely_time_dep('visualization_sqp/time')

	# P.compute_errors()

	if P.compute_proj_residuals_flag:
		P.plot_residuals('visualization_sqp')
	P.plot_inf_errors('visualization_sqp')
	# P.plot_objective('visualization_sqp')

	print('inf_errors\n', P.error_sequence_list)
	print('increments\n', P.incr_list)

	print('orders\n',orders.compute_EOC_orders(P.incr_list))

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0

def example_2D_t(lam, mesh_size, num_steps):
	# python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	T 		= 1.0

	beta = Expression(
		'''	
			( ( x[0] >= pow(3,-1) ? 1 : 0 )*( x[0] <= 2*pow(3,-1) ? 1 : 0 )
				*( x[1] >= pow(3,-1) ? 1 : 0 )*( x[1] <= 2*pow(3,-1) ? 1 : 0 ) )
		''',
		degree=5)

	y_0   	= Expression(
		'sin(pi*x[0])*sin(pi*x[1])',
		degree=5)

	p_end 	= Constant(0.0) 

	y_target= Expression(
		'''	cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) * (1 - 2*lam*pi)
			+ 2*lam*pow(pi,2)*sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) 
			* ( pow( pow(exp(1), -cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1]) ) + 1, -1) + 1 ) 
			- lam*pow(pi,2)*sin(2*pi*t)*cos(2*pi*t)*(pow(sin(pi*x[0])*cos(pi*x[1]),2) + pow(cos(pi*x[0])*sin(pi*x[1]),2))
			*(
				2*pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) 
				* pow(pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) + 1, -2) 
				+ cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])
				* pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) 
				* ( pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) - 1 ) 
				* pow( pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1]) ) + 1, -3)
			)
			+ 2*lam*pow(pi,2)*cos(2*pi*t)*sin(2*pi*t) * pow(sin(pi*x[0]),2)*pow(sin(pi*x[1]),2)
			* pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) * pow(pow(exp(1),cos(2*pi*t)*sin(pi*x[0])*sin(pi*x[1])) + 1, -2)
		''',
		t=0,
		degree=5,
		lam=1e-2)

	y_exact = Expression(
		'cos(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])',
		t=0,
		degree=5)

	p_exact = Expression(
		'- lam* sin(2*pi*t) * sin(pi*x[0])*sin(pi*x[1])',
		t=0,
		degree=5,
		lam=1e-2)
	
	u_b = Constant(0.05)
	u_a = Constant(-0.05)

	u_exact = Expression(
		'''( pow(pi, -2)*sin(2*pi*t)  > - 0.05 ?
			 pow(pi, -2)*sin(2*pi*t)  : - 0.05 ) < 0.05 ?
			 (pow(pi, -2)*sin(2*pi*t)  > - 0.05 ?
			  pow(pi, -2)*sin(2*pi*t)  : - 0.05) : 0.05''',
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
			- ( x[0] >= pow(3,-1) ? 1 : 0 )*( x[0] <= 2*pow(3,-1) ? 1 : 0 )
				*( x[1] >= pow(3,-1) ? 1 : 0 )*( x[1] <= 2*pow(3,-1) ? 1 : 0 )
			*( ( pow(pi, -2)*sin(2*pi*t)  > - 0.05 ?
				 pow(pi, -2)*sin(2*pi*t)  : - 0.05 ) < 0.05 ?
				 (pow(pi, -2)*sin(2*pi*t)  > - 0.05 ?
				  pow(pi, -2)*sin(2*pi*t)  : - 0.05) : 0.05)
		''',
		t=0,
		degree=5)

	g = Constant(0.0)

	total_time_start = time.perf_counter()

	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-10, 1e-10, 20, 20
	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	# build mesh
	# 2D
	new_mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), mesh_size, mesh_size)
	# initialize instance
	P = Quasi_Linear_Problem_Box(T, num_steps, 'time')
	# set up problem by callig class attrubutes
	P.set_state_space(new_mesh, 1)
	P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary)
	P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), None)
	# P.set_control_space(new_mesh, 1)
	P.set_control_constraints(u_a, u_b)
	P.set_cost(lam, y_target)
	P.set_state_equation(beta, f, g, y_0, p_end)
	P.set_exact_solution(y_exact, u_exact, p_exact)
	# set quasi-linear parameters
	P.set_non_linearity(mu, csi, csi_p, csi_pp)
	P.set_maxit_and_tol(tol, tol_pd, max_it_sqp, max_it_pd)
	P.compute_proj_residuals_flag = True
	# call solver
	P.sqp_solve()
	# compute errors
	P.compute_errors()
	# visualization
	P.visualize_paraview('visualization_sqp/paraview/2D/time')
	P.plot_inf_errors('visualization_sqp/2D/time')

	P.visualize_purely_time_dep('visualization_sqp/2D/time')

	if P.compute_proj_residuals_flag:
		P.plot_residuals('visualization_sqp/2D/time')

	print('inf_errors\n', P.error_sequence_list)
	print('increments\n', P.incr_list)
	print('orders\n', orders.compute_EOC_orders(P.incr_list))

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0

#######################################################################################
		
if __name__ == '__main__':
	level = logging.INFO
	fmt = '[%(levelname)s] %(asctime)s - %(message)s'
	logging.basicConfig(level=level, format=fmt)

	# example_1D(1e-2, 64, 4096)

	# example_1D_t(1e-2, 64, 4096)
	
	# example_2D(1e-2, 16, 256)

	# example_2D_N(1e-2, 16, 256)

	# example_2D_t(1e-2, 20, 400)

	logging.info('FINISHED')
	
