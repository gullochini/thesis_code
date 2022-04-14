from fenics import *
from mshr import * 
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import math
import sys
import logging

import primal_dual, orders

class Quasi_Linear_Problem_Box(primal_dual.Linear_Problem_Box):

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
	tol_nm : float
		Arbitrary small parameter needed for the Newton's method stopping criterion.
	tol_sm : float
		Arbitrary small parameter needed for the semi-smooth stopping criterion.
	max_it_nm : int
		Maximum number of Newton's step in SE approxiamtion.
	max_it_sm : int
		Maximum number of semi-smooth iterations.
	M_y_sp : scipy.sparse.csr.csr_matrix
		Mass matrix of shape (mesh_size, mesh_size),
		associated to the state space at a given time.
	B_u_sp_blocks : scipy.sparse.csr.csr_matrix
		Above diagonal block matrix of shape (mesh_size * num_steps, mesh_size * num_steps), 
		entries are the dt-beta-mass matrices (not time dependant).
	M_sub_diag_sp_blocks : scipy.sparse.csr.csr_matrix
		Below diagonal block matrix of shape (mesh_size * num_steps, mesh_size * num_steps), 
		entries are the dt-mass matrices (not time dependant).
	B_diag_sp_blocks : scipy.sparse.csr.csr_matrix
		Diagonal block matrix of shape (mesh_size * num_steps, mesh_size * num_steps), 
		entries are the dt-beta-mass matrices (not time dependant).
	M_diag_sp_blocks : scipy.sparse.csr.csr_matrix
		Diagonal block matrix of shape (mesh_size * num_steps, mesh_size * num_steps), 
		entries are the dt-mass matrices (not time dependant).
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
	M_y_0_term : numpy.ndarray
		Array of shape (mesh_size * num_steps,),
		containing space discretization of state at initial time.
	C_sp_blocks : scipy.sparse.csr.csr_matrix
		Diagonal block matrix of shape (mesh_size * num_steps, mesh_size * num_steps).		
	M_delta : scipy.sparse.csr.csr_matrix
		Block matrix of shape (mesh_size * num_steps, mesh_size * num_steps),
		corresponds to p'(u).
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
	set_maxit_and_tol(tol, tol_nm, tol_sm, max_it_nm, max_it_sm)
	to_dict_of_functions(vec)
	to_vec(D)
	compute_projection_residuals()
	plot_residuals(path)
	compute_invariants()
	compute_SE_terms(y_old)
	compute_AE_terms(y_old, p_old)
	assemble_SE_rhs(u_vec)
	assemble_AE_rhs(y_vec)
	compute_associated_state_and_adjoint(u_vec)
	semi_smooth_solve()
	subproblem_solve(u_vec, p_vec)

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

	def set_maxit_and_tol(self, tol_nm, tol_sm, max_it_nm, max_it_sm):

		"""

		Parameters
		----------
		tol_nm : float
			Arbitrary small parameter needed for the Newton's method stopping criterion.
		tol_sm : float
			Arbitrary small parameter needed for the semi-smooth stopping criterion.
		max_it_nm : int
			Maximum number of Newton's step in SE approxiamtion.
		max_it_sm : int
			Maximum number of semi-smooth iterations.

		"""	

		self.tol_nm = tol_nm
		self.tol_sm = tol_sm
		self.max_it_nm = max_it_nm
		self.max_it_sm = max_it_sm

	def to_dict_of_functions(self, vec):

		"""

		Parameters
		----------
		vec : numpy.ndarray
			Array of shape (mesh_size * (num_steps + 1),).

		Returns
		-------
		dict[int, dolfin.function.function.Function]
			keys from 0 to num_steps, 
			values are functions at key-th timestep.

		"""

		D = {n : Function(self.V) for n in range(self.num_steps + 1)}
		additional = Function(self.V)
		
		for n in range(self.num_steps + 1):

			if n < self.num_steps:
				additional.vector().set_local(vec[n*self.mesh_size : (n + 1)*self.mesh_size])
			else:
				additional.vector().set_local(vec[n*self.mesh_size : ])
			
			D[n].assign(additional)

		return D

	def to_vec(self, D):

		"""

		Parameters
		----------
		D : dict[int, dolfin.function.function.Function]
			keys from 0 to num_steps, 
			values are functions at key-th timestep.

		Returns
		-------
		numpy.ndarray
			Arrat of shape (mesh_size * (num_steps + 1),).

		"""

		vec = np.zeros(self.mesh_size*(self.num_steps + 1))

		for n in range(self.num_steps + 1):

			if n < self.num_steps:
				vec[n*self.mesh_size : (n + 1)*self.mesh_size] = D[n].vector().get_local()
			else:
				vec[n*self.mesh_size : ] = interpolate(D[n], self.V).vector().get_local()

		return vec

	def compute_projection_residuals(self):

		"""

		Returns
		-------
		tuple
			tuple of floats,
			entries are L2 and Linf norms of the projection residual, respectively.

		"""

		res_func = self.to_dict_of_functions(np.hstack((self.res, np.zeros(self.mesh_size))))

		res_L_2 = 0
		res_L_inf = []
		for n in range(self.num_steps):

			if self.control_type == 'neumann_boundary':
				res_L_2 += self.dt * norm(res_func[n], mesh=self.boundary_mesh) ** 2
				res_L_inf.append(norm(res_func[n].vector(), 'linf'))
			else:
				res_L_2 += self.dt * norm(res_func[n]) ** 2
				res_L_inf.append(norm(res_func[n].vector(), 'linf'))

		res_L_2 = np.sqrt(res_L_2)
		res_L_inf = np.max(res_L_inf)
		
		return res_L_2, res_L_inf

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

		res_array = np.array(self.proj_res_list)
		its = np.array(range(1, 1 + np.shape(np.array(res_array))[0]))

		ax.semilogy( 
			its, 
			self.incr_list, 	
			marker='o', 
			color='r', 
			linewidth=0.7, 
			label=r'incr$_n$', 		  
			markerfacecolor='none', 
			markeredgecolor='r')		
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
		ax.set(xlabel=r'Semi-Smooth iteration $n$')
		ax.legend(loc="lower left")

		plt.setp( ax, xticks=its )

		fig.set_size_inches(6, 6)

		plt.savefig(path + '/residuals.pdf')

		return 0

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
		ax.set(xlabel=r'Semi-smooth iteration $n$', ylabel=r'$L^\infty$-error')
		ax.legend(loc="lower left")

		plt.setp(ax, xticks=its)

		fig.set_size_inches(6, 6)

		plt.savefig(path + '/inf_errors.pdf')

		return 0

	def compute_invariants(self):

		""" Computes all the time independant variables
		needed in the sub-problem solver and stores them as class attributes:
		M_y_sp, M_proj_sp_blocks, B_u_sp_blocks, B_diag_sp_blocks, 
		M_diag_sp_blocks, M_sub_diag_sp_blocks,
		u_final, U_a, U_b, b_target_array, b_array, M_y_0_term.

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		v = TestFunction(self.V)
		p = TrialFunction(self.V)
		y = TrialFunction(self.V)
		u = TrialFunction(self.V)

		if self.control_type == 'distributed':
			b_u = self.dt*self.beta*u*v*dx
			m_u = u*v*dx
			m_p = (1.0/self.lam)*self.beta*p*v*dx

		elif self.control_type == 'neumann boundary':
			b_u = self.dt*self.beta*u*v*ds
			m_u = u*v*ds
			m_p = (1.0/self.lam)*self.beta*p*v*ds

		elif self.control_type == 'time':
			b_u = self.dt*self.beta*u*v*dx
			m_u = u*v*dx
			m_p = (1.0/self.lam)*self.beta*p*v*dx
			p_int = (1.0/self.lam)*self.beta*v*dx
		
		M_p, M_u = assemble(m_p), assemble(m_u)

		# quadratic and cubic polinomials make the size of nodal basis greater
		if self.degree > 1:
			self.mesh_size = np.shape(self.y_init)[0]

		M_p_mat, M_u_mat = as_backend_type(M_p).mat(), as_backend_type(M_u).mat()
		M_p_sp, M_u_sp = sps.csr_matrix(M_p_mat.getValuesCSR()[::-1], shape = M_p_mat.size), sps.csr_matrix(M_u_mat.getValuesCSR()[::-1], shape = M_u_mat.size)
		M_u_sp_inv = spla.spsolve(M_u_sp, sps.identity(self.mesh_size))

		self.M_p_sp_blocks = sps.block_diag(
			[M_p_sp for i in range(self.num_steps)], format='csc')

		M_u_sp_inv_blocks = sps.block_diag(
			[M_u_sp_inv for i in range(self.num_steps)], format='csc')

		if self.control_type == 'distributed' or self.control_type == 'neumann boundary':
			# compute inverse of the control mass matrix
			M_u_sp_inv = spla.spsolve( M_u_sp, sps.identity(self.mesh_size))
			M_u_sp_inv_blocks = sps.block_diag([M_u_sp_inv for i in range(self.num_steps)], format='csc')
			self.M_proj_sp_blocks = M_u_sp_inv_blocks.dot(self.M_p_sp_blocks)

		elif self.control_type == 'time':	
			self.P = assemble((1.0/self.lam)*self.beta*v*dx)
			M_p_int = np.vstack([self.P for i in range(self.mesh_size)])
			M_p_int_sp = sps.csr_matrix(M_p_int, shape=np.shape(M_p_int))
			self.M_proj_sp_blocks = sps.block_diag([M_p_int_sp for i in range(self.num_steps)], format='csc')
			
		m_y = v*y*dx
		B_u, M_y = assemble(b_u), assemble(m_y)

		# apply dirichlet boundary conditions
		if not self.dirichlet_boundary is None:
			# primal
			self.bc = DirichletBC(self.V, self.y_D, self.dirichlet_boundary)
			if not self.control_type == 'time':
				self.bc.apply(B_u)
			# adjoint
			self.bc_adj = DirichletBC(self.V, self.p_D, self.dirichlet_boundary)
			self.bc_adj.apply(M_y)

		B_u_mat, M_y_mat = as_backend_type(B_u).mat(), as_backend_type(M_y).mat()
		B_u_sp, self.M_y_sp	= sps.csr_matrix(B_u_mat.getValuesCSR()[::-1], shape = B_u_mat.size), sps.csr_matrix(M_y_mat.getValuesCSR()[::-1], shape = M_y_mat.size)

		B_u_sp_blocks = sps.block_diag(
			[B_u_sp for i in range(1, self.num_steps)], format='csc')

		self.B_u_sp_blocks 	= sps.bmat(
			[ 	[None, B_u_sp_blocks], 
				[np.zeros((self.mesh_size, self.mesh_size)), None] ])

		self.B_diag_sp_blocks = sps.block_diag(
			[B_u_sp for i in range(self.num_steps)], format='csc')

		self.M_diag_sp_blocks = sps.block_diag(
			[self.dt*self.M_y_sp for i in range(self.num_steps)], format='csc')

		M_sub_diag_sp_blocks = sps.block_diag(
			[self.dt*self.M_y_sp for i in range(1, self.num_steps)], format='csc')

		self.M_sub_diag_sp_blocks = sps.bmat(
			[ 	[None, np.zeros((self.mesh_size, self.mesh_size))], 
				[M_sub_diag_sp_blocks, None] ])

		# compute the term depending on y_0 in the RHS of the state equation
		b_y_0 = self.M_y_sp.dot(self.y_init)
		# compute final control
		self.u_final = self.compute_final_control(self.p_final)
		# compute the term depending on u_T in the RHS of the state equation
		b_u_T = B_u_sp.dot(self.u_final)
		# RHS of state equation (without the term containing f)
		b_array = np.hstack( (
			b_y_0, 
			np.zeros(self.mesh_size*(self.num_steps - 2)), 
			b_u_T) )

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

		self.M_y_0_term = np.hstack( ( 
			self.dt*self.M_y_sp.dot(self.y_init), 
			np.zeros(self.mesh_size*(self.num_steps - 1)) ) )

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
		v = TestFunction(self.V)

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
		self.A_sp_blocks = sps.bmat(	
			[ [	self.M_y_sp + A_sp_list[i + 1] if i == j 
				else - self.M_y_sp if i - j == 1
				else None 
				for j in range(self.num_steps) ]
				for i in range(self.num_steps) ],
			format='csc')

		# assemble block matrix for lhs of AE: all times but final on diagonal
		self.A_sp_adj_blocks = sps.bmat(	
			[ [	self.M_y_sp + A_sp_list[i] if i == j 
				else - self.M_y_sp if j - i == 1
				else None 
				for j in range(self.num_steps) ]
				for i in range(self.num_steps) ],
			format='csc')
		
		# store linearization term that'll go on the rhs of LSE 
		self.L_1_array = np.array(L_1_list)

		return 0

	def compute_AE_terms(self, y_old, p_old):

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
		int
			0 if successful, 1 otherwise.

		"""

		# p_old_vec = self.to_vec(p_old)[: self.num_steps*self.mesh_size]
		# p_beta_vec = self.lam * self.M_p_sp_blocks.dot(p_old_vec)
		# p_beta = self.to_dict_of_functions(np.hstack((p_beta_vec, self.p_final)))

		# Define variational problem
		y = TrialFunction(self.V)
		# p = TrialFunction(self.V)
		v = TestFunction( self.V)

		# initialize lists
		D_sp_list = []
		L_2_list = []

		for k in range(self.num_steps):

			# adjoint equation is , for all v in V
			a_2 = self.dt*self.mu*self.csi_p(y_old[k])*		p_old[k]*		inner(grad(y),			grad(v))*dx
			a_3 = self.dt*self.mu*self.csi_p(y_old[k])*		y*				inner(grad(p_old[k]), 	grad(v))*dx
			a_4 = self.dt*self.mu*self.csi_pp(y_old[k])*	p_old[k]*y*		inner(grad(y_old[k]), 	grad(v))*dx

			# assembling FENICS matices for state and adjoint equation
			A_adj = assemble(a_2 + a_3 + a_4)

			# apply dirichlet boundary conditions
			if not self.dirichlet_boundary is None:
				# adjoint
				self.bc_adj = DirichletBC(self.V, self.p_D, self.dirichlet_boundary)
				self.bc.apply(A_adj)

			# convert to sparse matrices
			A_adj_mat = as_backend_type(A_adj).mat()

			A_adj_sp = sps.csr_matrix(A_adj_mat.getValuesCSR()[::-1], shape = A_adj_mat.size)

			# interested in every timestep but the final one
			D_sp_list.append(- self.dt*self.M_y_sp + A_adj_sp.copy())

		# interested in all blocks but the one at initial final time
		D_sp_blocks = sps.block_diag([D_sp_list[i] for i in range(1, self.num_steps)], format='csc')
		self.D_sp_blocks = sps.bmat(
			[	[None, np.zeros((self.mesh_size, self.mesh_size))], 
				[ D_sp_blocks, None] ])
		self.D_0_term = np.hstack( ( 
			D_sp_list[0].dot(self.y_init), 
			np.zeros(self.mesh_size*(self.num_steps - 1)) ) )
				
		return 0

	def assemble_LSE_rhs(self, u_vec):

		"""

		Parameters
		----------
		u_vec : numpy.ndarray

		Returns
		-------
		numpy.ndarray
			Array of shape (num_steps * mesh_size,),
			corresponding to the linearized state equation RHS (calculated with u_vec).

		"""

		return self.B_u_sp_blocks.dot(u_vec) + self.b_array + self.L_1_array

	def assemble_AE_rhs(self, y_vec):

		"""

		Parameters
		----------
		y_vec : numpy.ndarray

		Returns
		-------
		numpy.ndarray
			Array of shape (num_steps * mesh_size,),
			corresponding to the adjoint state equation RHS (calculated with y_vec).

		"""

		return self.M_sub_diag_sp_blocks.dot(y_vec) + self.b_target_array + self.M_y_0_term

	def compute_associated_state_and_adjoint(self, u_vec):

		"""

		Parameters
		----------
		u_vec : numpy.ndarray

		Returns
		-------
		tuple
			tuple of numpy.ndarray

		"""

		# initialization
		init = interpolate(Constant(0.0), self.V)

		y_old = {n : init for n in range(self.num_steps + 1)}
		# p_old = {n : init for n in range(self.num_steps + 1)}

		start_time_nm = time.perf_counter()

		it = 0
		while True:

			self.compute_LSE_terms(y_old)

			y_vec = spla.spsolve(self.A_sp_blocks, self.assemble_LSE_rhs(u_vec))

			y = self.to_dict_of_functions(np.hstack((self.y_init, y_vec)))

			# compute L-infinity norms
			state_diff_vec, adjoint_diff_vec = np.zeros(self.num_steps + 1), np.zeros(self.num_steps + 1)

			for n in range(self.num_steps + 1):

				state_diff_vec[n] = norm( y[n].vector() - y_old[n].vector(), 'linf')

			res = np.max(state_diff_vec)

			# stopping criterion
			if it > 1 and res < self.tol_nm:
				logging.info('Newton method stopping criterion met')
				break

			y_old = {n : Function(self.V) for n in range(self.num_steps + 1)}						

			# update previous iterates
			for n in range(self.num_steps + 1):
				y_old[n].assign(y[n])

			it += 1

		logging.info(f'computation of associated state took {it} iterations and {time.perf_counter() - start_time_nm} s')

		p_vec = spla.spsolve(self.A_sp_adj_blocks, self.assemble_AE_rhs(y_vec))

		return y_vec, p_vec

	def semi_smooth_solve(self):

		"""

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		logging.info(f'size of mesh is {self.mesh_size} with {self.num_steps} timesteps')
		logging.info(f'max_it for semi-smooth Newton method is: {self.max_it_sm}, with tolerance: {self.tol_sm}')
		logging.info(f'max_it for Newton method in the computation of associated state and adjoint is: {self.max_it_nm}, with tolerance: {self.tol_nm}')

		# set initial conditions vectors
		self.y_init = interpolate(self.y_0, self.V).vector().get_local()
		self.p_final = interpolate(self.p_end, self.V).vector().get_local()
		self.u_final = self.compute_final_control(self.p_final)
		
		init = interpolate(Constant(0.0), self.V)

		u_old_vec = np.hstack([ init.vector().get_local() for n in range(self.num_steps)])
		
		start_time_sm = time.perf_counter()

		self.compute_invariants()

		logging.info(f'invariants computed in {time.perf_counter() - start_time_sm} s')

		y_old_vec, p_old_vec = self.compute_associated_state_and_adjoint(u_old_vec)

		# compute AE terms
		self.compute_AE_terms(			
			self.to_dict_of_functions(np.hstack((self.y_init, y_old_vec))),
			# p_old_vec
			self.to_dict_of_functions(np.hstack((p_old_vec, self.p_final)))
			)

		self.proj_res_list, self.incr_list = [], []
		# self.cost_func_list = []
		self.error_sequence_list = []
		self.error_sequence_list.append(self.compute_inf_errors(
			self.to_dict_of_functions(np.hstack((self.y_init, y_old_vec))),
			self.to_dict_of_functions(np.hstack((u_old_vec, self.u_final))),
			self.to_dict_of_functions(np.hstack((p_old_vec, self.p_final))) ) )

		it = 0
		while True:

			logging.info(f'solving subproblem, iteration {it}')

			start_time_sub = time.perf_counter()
			# call subproblem solver
			u_vec = self.subproblem_solve(u_old_vec, p_old_vec)
			logging.info(f'sub-problem solved in {time.perf_counter() - start_time_sub} s')

			# compute associated state and adjoint
			y_vec, p_vec = self.compute_associated_state_and_adjoint(u_vec)

			# convert to dictionaries
			y_t = self.to_dict_of_functions(np.hstack((self.y_init, y_vec)))
			u_t = self.to_dict_of_functions(np.hstack((u_vec, self.u_final)))
			p_t = self.to_dict_of_functions(np.hstack((p_vec, self.p_final)))

			y_old_t = self.to_dict_of_functions(np.hstack((self.y_init, y_old_vec)))
			u_old_t = self.to_dict_of_functions(np.hstack((u_old_vec, self.u_final)))
			p_old_t = self.to_dict_of_functions(np.hstack((p_old_vec, self.p_final)))

			# # evaluate cost functional
			# cost = self.evaluate_cost_functional(y_t, u_t)
			# self.cost_func_list.append(cost)
			# logging.info(f'objective functional: {cost}')

			# compute increment
			state_diff_vec, control_diff_vec, adjoint_diff_vec = [], [], []
			for n in range(self.num_steps + 1):

				state_diff_vec.append(norm( y_t[n].vector() - y_old_t[n].vector(), 'linf'))
				control_diff_vec.append(norm( u_t[n].vector() - u_old_t[n].vector(), 'linf'))
				adjoint_diff_vec.append(norm( p_t[n].vector() - p_old_t[n].vector(), 'linf'))

			max_state_diff, max_control_diff, max_adjoint_diff = np.max(np.array(state_diff_vec)), np.max(np.array(control_diff_vec)), np.max(np.array(adjoint_diff_vec))
			incr = max_state_diff + max_control_diff + max_adjoint_diff
			# store increment
			self.incr_list.append(incr)
			logging.info(f'increment: {incr}')

			# compute L-inf and L-2 projection residuals
			res_L_2, res_L_inf = self.compute_projection_residuals()
			logging.info(f'L-2 residual: {res_L_2}')
			logging.info(f'L-inf residual: {res_L_inf}\n')

			self.proj_res_list.append(res_L_2)

			self.error_sequence_list.append(self.compute_inf_errors(y_t, u_t, p_t))

			# stopping criterion
			if it > 1 and (res_L_2 < self.tol_sm and res_L_inf < self.tol_sm):
				logging.info('SEMI-SMOOTH STOPPING CRITERION MET')
				break
		
			elif it > self.max_it_sm:

				y = self.to_dict_of_functions(np.hstack((self.y_init, y_vec)))
				u = self.to_dict_of_functions(np.hstack((u_vec, self.u_final)))
				p = self.to_dict_of_functions(np.hstack((p_vec, self.p_final)))

				# store computed triple anyway
				self.y, self.u, self.p = y, u, p

				logging.error(f'NO CONVERGENCE REACHED: maximum number of iterations {self.max_it_sm} for semi-smooth Newton method excedeed')

				return 1

			# compute AE terms
			self.compute_AE_terms(y_t, p_t)

			# update previous iterates
			y_old_vec = np.copy(y_vec)
			u_old_vec = np.copy(u_vec)
			p_old_vec = np.copy(p_vec)
			
			it += 1

		# store optimal triple
		self.y, self.u, self.p = y_t, u_t, p_t

		if self.control_type == 'time':
			# store the values of -1/lam * B* p for every timestep
			gradient_term = {}
			for i in range(self.num_steps + 1):
				gradient_term[i] = - self.P @ self.p[i].vector().get_local()

			self.gradient_term = gradient_term	

		# output time at end of while loop
		end_time_sm = time.perf_counter()
		logging.info(f'total elapsed time: {end_time_sm - start_time_sm}')

		return 0	

	def subproblem_solve(self, u_vec, p_vec):

		"""

		Parameters
		----------
		u_vec : numpy.ndarray
		p_vec : numpy.ndarray

		Returns
		-------
		numpy.ndarray
			Array of shape (mes_size * num_steps, ) corresponding to u_old + delta_u.

		"""
		
		# size of block matrices
		time_size = self.mesh_size*self.num_steps

		# compute active indices matrices
		X_a, X_b = self.compute_active_sets(-self.M_proj_sp_blocks.dot(p_vec))
		# inactive indices matrix
		I_a_b = sps.identity(time_size) - X_a - X_b	

		P_diag_sp_blocks = I_a_b.dot(self.M_proj_sp_blocks)

		# RHS of projection formula
		self.res = X_a.dot(self.U_a) + X_b.dot(self.U_b) - P_diag_sp_blocks.dot(p_vec) - u_vec

		# assemble ( (5 x size x num_steps) x (5 x size x num_steps) ) sparse lhs of the linear system by blocks
		Global_matrix = sps.bmat(
			[	[None, 					None, 					self.A_sp_blocks,			-self.B_u_sp_blocks 	],
				[None, 					self.A_sp_adj_blocks,	-self.M_sub_diag_sp_blocks,	None					],
				[self.A_sp_adj_blocks, 	None, 					self.D_sp_blocks,			None 					],
				[P_diag_sp_blocks,		P_diag_sp_blocks, 		None,						sps.identity(time_size)	]	],
			format='csc')

		# assemble (3 x size x num_steps) dense rhs of linear system 
		Global_right_term = np.hstack(
			[	np.zeros((3*time_size)),
				self.res					]	)


		# time0 = time.perf_counter()

		# print(f'global matrix, nonzero = {Global_matrix.count_nonzero()}')

		# spilu = spla.spilu(Global_matrix)

		# # self.spy_sparse(spilu.L @ spilu.U)

		# print(f'approx inverse, nonzero = {spilu.nnz}')

		# time1 = time.perf_counter()

		# logging.info(f'incomplete LU factorization in: {time1 - time0} s')

		# sol_sp = spilu.solve(Global_right_term)

		# # compute solution of sparse linear system	
		# time2 = time.perf_counter()
		# logging.info(f'sparse system of size {np.shape(sol_sp)[0]} x {np.shape(sol_sp)[0]} solved in: {time2 - time1} s')


		# print('---------------spy------------------')
		# self.spy_sparse(Global_matrix)
		# print('----------------ok------------------')

		# compute solution of sparse linear system	
		time1 = time.perf_counter()
		sol_sp = spla.spsolve(Global_matrix, Global_right_term)
		time2 = time.perf_counter()
		logging.info(f'sparse system of size {np.shape(sol_sp)[0]} x {np.shape(sol_sp)[0]} solved in: {time2 - time1} s')
	
		# solution vector slicing corresponding to control increment
		delta_u = sol_sp[3*time_size :]
		
		return u_vec + delta_u


############################### DISTRIBUTED CONTROL TEST EXAMPLES ######################################

def example_1D(lam, mesh_size, num_steps):
	## python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	T 		= 1.0
	beta 	= Constant(1.0)
	y_0   	= Expression(' sin(pi*x[0])', degree=5)
	p_end 	= Constant(0.0) 

	y_target= Expression(' cos(2*pi*t) * sin(pi*x[0]) * (1 - 2*lam*pi) - lam*pow(pi,2) * sin(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) - sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) + lam * pow(pi,2)*sin(2*pi*t)*cos(2*pi*t)*pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) * (   pow(cos(pi*x[0]),2)* ( sin(pi*x[0])*cos(2*pi*t)* ( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) - 1 ) - pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) - 1 ) + pow(sin(pi*x[0]), 2)*( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) + 1 )	) * pow( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) + 1, -3) ', t=0, degree=5, lam=lam)

	y_exact = Expression(' 		 		cos(2*pi*t) * sin(pi*x[0])', t=0, degree=5)
	p_exact = Expression('		 - lam* sin(2*pi*t) * sin(pi*x[0])', t=0, degree=5, lam=lam)
	
	# space and time dependant constraints
	u_b = Expression(' 2 * (0.2 + 0.5 * x[0]) * t', t=0, degree=2)
	u_a = Expression(' - 2 * (0.2 + 0.5 * x[0]) * (1 - t)', t=0, degree=2)
	
	u_exact = Expression('( sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ? sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ) < 2 * t * (0.2 + 0.5 * x[0]) ? (sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ? sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t)) : 2 * t * (0.2 + 0.5 * x[0])', t=0, degree=5)
	f 		= Expression(' - pow(pi,2) * cos(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) - sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) - 2*pi*sin(2*pi*t) * sin(pi*x[0]) - ((sin(2*pi*t) * sin(pi*x[0]) > u_a ? sin(2*pi*t) * sin(pi*x[0]) : u_a) < u_b ? (sin(2*pi*t) * sin(pi*x[0]) > u_a ? sin(2*pi*t) * sin(pi*x[0]) : u_a) : u_b)', t=0, u_a=u_a, u_b=u_b, degree=5)

	g = Constant(0.0)
	
	mu, tol_nm, tol_sm, max_it_nm, max_it_sm = 1, 1e-10, 1e-4, 20, 20
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
	P.set_maxit_and_tol(tol_nm, tol_sm, max_it_nm, max_it_sm)
	# call solver
	P.semi_smooth_solve()

	P.visualize_paraview('visualization_semi_smooth_quasi_linear/paraview/1D/distributed')
	
	P.visualize_1D(0, 1, 128, 'visualization_semi_smooth_quasi_linear')
	P.visualize_1D_exact(0, 1, 128, 'visualization_semi_smooth_quasi_linear')
	# compute errors
	P.compute_relative_errors()

	P.plot_residuals('visualization_semi_smooth_quasi_linear')
	P.plot_inf_errors('visualization_semi_smooth_quasi_linear')
	# P.plot_objective('visualization_semi_smooth_quasi_linear')

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
	
	mu, tol_nm, tol_sm, max_it_nm, max_it_sm = 1, 1e-10, 1e-4, 20, 12


	T = 1.0
	beta = Constant( 1.0 )
	y_0 = Expression(
		'sin(pi*x[0])*sin(pi*x[1])',
		degree=5)

	p_end = Constant(0.0) 

	y_target = Expression(
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
	P.set_maxit_and_tol(tol_nm, tol_sm, max_it_nm, max_it_sm)
	# call solver
	P.semi_smooth_solve()
	# compute errors
	P.compute_relative_errors()
	# visualization
	P.visualize_paraview('visualization_semi_smooth_quasi_linear/paraview/2D/distributed')
	
	P.plot_residuals('visualization_semi_smooth_quasi_linear/2D/distributed')
	P.plot_inf_errors('visualization_semi_smooth_quasi_linear/2D/distributed')

	print('inf_errors\n',P.error_sequence_list)
	print('increments\n',P.incr_list)
	print('orders\n',orders.compute_EOC_orders(P.incr_list))
	logging.info(fr'computed order of convergence is $q =${math.log(P.incr_list[-1]/P.incr_list[-2]) / math.log(P.incr_list[-2]/P.incr_list[-3])}')

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0

######################### PURELY TIME DEP CONTROL TEST EXAMPLES ########################

################################### 1D ###################################

def example_1D_t(lam, mesh_size, num_steps):
	## python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	g = Constant(0.0)

	T 		= 1.0
	beta 	= Expression(
		'''	
			( x[0] >= pow(3,-1) ? 1 : 0 )*( x[0] <= 2*pow(3,-1) ? 1 : 0 )
		''',
		degree=5)
	y_0   	= Expression(' sin(pi*x[0])', degree=5)
	p_end 	= Constant(0.0) 

	y_target= Expression(' cos(2*pi*t) * sin(pi*x[0]) * (1 - 2*lam*pi) - lam*pow(pi,2) * sin(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) - sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) + lam * pow(pi,2)*sin(2*pi*t)*cos(2*pi*t)*pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) * (   pow(cos(pi*x[0]),2)* ( sin(pi*x[0])*cos(2*pi*t)* ( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) - 1 ) - pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) - 1 ) + pow(sin(pi*x[0]), 2)*( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) + 1 )	) * pow( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) + 1, -3) ', t=0, degree=5, lam=lam)

	y_exact = Expression(' 		 		cos(2*pi*t) * sin(pi*x[0])', t=0, degree=5)
	p_exact = Expression('		 - lam* sin(2*pi*t) * sin(pi*x[0])', t=0, degree=5, lam=lam)
	
	# constant constraints
	u_b 	= Constant( 0.25 )
	u_a 	= Constant(-0.25 )

	u_exact = Expression(
		'''( pow(pi,-1) * sin(2*pi*t) > u_a ? pow(pi,-1) * sin(2*pi*t) : u_a ) < u_b ? (pow(pi,-1) * sin(2*pi*t) > u_a ? pow(pi,-1) * sin(2*pi*t) : u_a) : u_b''', 
		t=0, 
		u_a=u_a, 
		u_b=u_b, 
		degree=5)

	f = Expression(
		''' - pow(pi,2) * cos(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) - sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) - 2*pi*sin(2*pi*t) * sin(pi*x[0]) 
		- ((pow(pi,-1) * sin(2*pi*t) > u_a ? pow(pi,-1) * sin(2*pi*t) : u_a) < u_b ? (pow(pi,-1) * sin(2*pi*t) > u_a ? pow(pi,-1) * sin(2*pi*t) : u_a) : u_b)''', 
		t=0, 
		u_a=u_a, 
		u_b=u_b, 
		degree=5)

	mu, tol_nm, tol_sm, max_it_nm, max_it_sm = 1, 1e-10, 1e-4, 20, 20
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
	P.set_maxit_and_tol(tol_nm, tol_sm, max_it_nm, max_it_sm)
	# call solver
	P.semi_smooth_solve()

	P.visualize_paraview('visualization_semi_smooth_quasi_linear/paraview/1D/time')
	P.visualize_1D(0, 1, 128, 'visualization_semi_smooth_quasi_linear/time')
	P.visualize_1D_exact(0, 1, 128, 'visualization_semi_smooth_quasi_linear/time')
	# compute errors
	P.compute_errors()

	P.visualize_purely_time_dep('visualization_semi_smooth_quasi_linear/time')

	P.plot_residuals('visualization_semi_smooth_quasi_linear/time')
	P.plot_inf_errors('visualization_semi_smooth_quasi_linear/time')
	# P.plot_objective('visualization_semi_smooth_quasi_linear/time')

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

	T = 1.0

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

	mu, tol_nm, tol_sm, max_it_nm, max_it_sm = 1, 1e-10, 1e-8, 20, 20
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
	P.set_maxit_and_tol(tol_nm, tol_sm, max_it_nm, max_it_sm)
	# P.compute_proj_residuals_flag = True
	# call solver
	P.semi_smooth_solve()
	# compute errors
	P.compute_errors()
	# visualization
	P.visualize_paraview('visualization_semi_smooth_quasi_linear/paraview/2D/time')
	P.visualize_purely_time_dep('visualization_semi_smooth_quasi_linear/2D/time')
	
	P.plot_residuals('visualization_semi_smooth_quasi_linear/2D/time')
	P.plot_inf_errors('visualization_semi_smooth_quasi_linear/2D/time')

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
	# example_2D(1e-2, 16, 256)

	# example_1D_t(1e-2, 64, 4096)
	# example_2D_t(1e-2, 16, 256)

	logging.info('FINISHED')
