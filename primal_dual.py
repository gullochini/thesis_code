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
import OCP

class Linear_Problem_Box(OCP.Problem):


	"""

	Attributes
	----------
	u_a : dolfin.function.expression.Expression
		Lower constraint function, can be satially and time dependant.
	u_b : dolfin.function.expression.Expression
		Upper constraint function, can be satially and time dependant.
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
	primal_dual_solve(max_it)
	compute_active_sets(proj)
	compute_final_control(p_final)
	same_active_sets(X, X_old)
	compare_active_sets(X, X_old)

	"""

	def set_control_constraints(self, lower, upper):

		"""

		Parameters
		----------
		u_a : dolfin.function.expression.Expression
			Lower constraint function, can be spatially and time dependant.
		u_b : dolfin.function.expression.Expression
			Upper constraint function, can be spatially and time dependant.

		"""

		self.u_a = lower
		self.u_b = upper

	def primal_dual_solve(self, max_it):

		"""

		Parameters
		----------
		max_it : int
			Maximum number of iteration for the primal-dual active set strategy.

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		logging.info(f'size of mesh is {self.mesh_size} with {self.num_steps} timesteps')

		# set initial conditions
		y_init 	= interpolate(self.y_0, 	self.V).vector().get_local()
		p_final	= interpolate(self.p_end, 	self.V).vector().get_local()

		# Define variational problem
		y = TrialFunction(self.V)
		p = TrialFunction(self.V)
		# control space ?
		u = TrialFunction(self.V)
		v = TestFunction(self.V)

		start_time = time.perf_counter()

		# primal equation is lhs = rhs_u + rhs, for all v in V
		lhs = y*v*dx + self.dt*dot(grad(y), grad(v))*dx

		m_y	= y*v*dx

		# adjoint equation is lhs_adj = rhs_y_adj + rhs_adj, for all v in V
		rhs_y_adj = self.dt*y*v*dx

		# gradient equation is m_1 = - m_2, for all v in V
		if self.control_type == 'distributed':
			rhs_u = self.dt*self.beta*u*v*dx
			m_1 = u*v*dx
			m_2 = (1.0/self.lam)*self.beta*p*v*dx

		elif self.control_type == 'neumann boundary':
			rhs_u = self.dt*self.beta*u*v*ds
			m_1 = u*v*ds
			m_2 = (1.0/self.lam)*self.beta*p*v*ds

		# assembling FENICS mass matrices for first order condition
		M_1, M_2, M_y = assemble(m_1), assemble(m_2), assemble(m_y)

		# assembling FENICS matices for state equation
		A, B = assemble(lhs), assemble(rhs_u)

		# assembling FENICS matices for adjoint equation
		B_adj = assemble(rhs_y_adj)

		# apply dirichlet boundary conditions
		if self.dirichlet_boundary is not None:
			# primal
			self.bc = DirichletBC(self.V, self.y_D, self.dirichlet_boundary)
			self.bc.apply(A)
			self.bc.apply(B)
			self.bc.apply(M_y)
			# adjoint
			self.bc_adj = DirichletBC(self.V, self.p_D, self.dirichlet_boundary)
			self.bc_adj.apply(B_adj)

		# state and adjoint equations
		A_mat, B_mat, B_adj_mat = as_backend_type(A).mat(), as_backend_type(B).mat(), as_backend_type(B_adj).mat()

		A_sp, B_sp, B_adj_sp = sps.csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size), sps.csr_matrix(B_mat.getValuesCSR()[::-1], shape = B_mat.size), sps.csr_matrix(B_adj_mat.getValuesCSR()[::-1], shape = B_adj_mat.size)

		# gradient equation and rhs mass matrices
		M_1_mat, M_2_mat, M_y_mat = as_backend_type(M_1).mat(), as_backend_type(M_2).mat(), as_backend_type(M_y).mat()

		M_1_sp, M_2_sp, M_y_sp = sps.csr_matrix(M_1_mat.getValuesCSR()[::-1], shape = M_1_mat.size), sps.csr_matrix(M_2_mat.getValuesCSR()[::-1], shape = M_2_mat.size), sps.csr_matrix(M_y_mat.getValuesCSR()[::-1], shape = M_y_mat.size)

		# interested in all controls but the one at initial time
		B_sp_blocks = sps.block_diag([B_sp for i in range(1, self.num_steps)], format='csr')
		B_sp_blocks = sps.bmat([ [None, B_sp_blocks], [np.zeros((self.mesh_size, self.mesh_size)), None] ])

		# interested in all states but the one at final time
		B_adj_sp_blocks = sps.block_diag([B_adj_sp	for i in range(1, self.num_steps)], format='csr')
		B_adj_sp_blocks	= sps.bmat([ [None, np.zeros((self.mesh_size, self.mesh_size))], [ B_adj_sp_blocks, None] ])

		M_1_sp_blocks = sps.block_diag([M_1_sp for i in range(self.num_steps)], format='csr')
		M_2_sp_blocks = sps.block_diag([M_2_sp for i in range(self.num_steps)], format='csr')

		# compute inverse of the control mass matrix
		M_1_sp_inv = spla.spsolve(M_1_sp, sps.identity(self.mesh_size))

		# we need projection formula block matrix
		M_1_sp_inv_blocks = sps.block_diag([M_1_sp_inv	for i in range(self.num_steps)], format='csr')


		A_sp_blocks	= sps.bmat(
			[ [	A_sp if i == j 
				else - M_y_sp if i - j == 1
				else None 
				for j in range(self.num_steps) ]
				for i in range(self.num_steps) ],
			format='csr')

		# RHS OF STATE EQUATION
		# compute the term depending on y_0 in the rhs of state equation
		b_y_0 = M_y.array().dot(y_init)

		u_final = self.compute_final_control(p_final)
		# compute the term depending on u_T in the rhs of the state equation
		b_u_T = B.array().dot(u_final)

		# define rhs of state equation
		b_array	= np.hstack((b_y_0, np.zeros(self.mesh_size*(self.num_steps - 2)), b_u_T))

		# RHS OF ADJOINT EQUATION
		# compute the term depending on y_0 in the rhs os adjoint equation
		b_adj_y_0 = B_adj.array().dot(y_init)
		# initialize an array to store rhs of adjoint equation
		b_adj_array = np.array([])
		b_f_array = np.array([])
		# initialize discrete vectors of u_a, u_b of size (num_steps x size) for 
		U_a, U_b = np.array([]), np.array([])
		# time stepping 
		t = 0
		for n in range(self.num_steps + 1):

			# update target function, f on the rhs and box constraints with current time
			self.y_target.t = t
			self.u_a.t, self.u_b.t = t, t
			self.f.t, self.g.t = t, t

			# right hand side of AE and SE are time dependent
			rhs_adj = - self.dt*self.y_target*v*dx			
			rhs_f = self.dt*self.f*v*dx + self.dt*self.g*v*ds
			# assemble right hand side AE and SE
			b_adj = assemble(rhs_adj)
			b_f	= assemble(rhs_f)
			# apply dirichlet boundary conditions
			if self.dirichlet_boundary is not None:
				self.bc_adj = DirichletBC(self.V, self.p_D, self.dirichlet_boundary)
				self.bc_adj.apply(b_adj)
				self.bc = DirichletBC(self.V, self.y_D, self.dirichlet_boundary)
				self.bc.apply(b_f)

			# in the first entry we have to add the term containing y_0
			if n == 0:
				b_adj_array = np.hstack((b_adj_array, b_adj_y_0 + b_adj.get_local()))
			# we don't care for the final time
			elif n < self.num_steps:
				b_adj_array = np.hstack((b_adj_array, b_adj.get_local()))
			# we don't care for the initial time
			if n > 0:
				b_f_array = np.hstack((b_f_array, b_f.get_local()))

			if n < self.num_steps:
				U_a = np.hstack(
					(	U_a,
						interpolate(self.u_a, self.V).vector().get_local() ) )
				U_b = np.hstack(
					(	U_b,
						interpolate(self.u_b, self.V).vector().get_local() ) )	

			# update time
			t += self.dt

		self.U_a, self.U_b = U_a, U_b

		# update rhs of SE with the term containing f
		b_array += b_f_array

		# quadratic a cubic polinomials make the size of nodal basis greater
		if self.degree > 1:
			self.mesh_size = np.shape(b_adj.get_local())[0]

		# size of block matrices
		time_size = self.mesh_size*self.num_steps

		proj = np.zeros((time_size))

		# primal dual active set looping
		it = 0
		while True:
			# compute active set matrices using the history of p_n
			X_a, X_b = self.compute_active_sets(proj)
			
			logging.info(f'active sets computed, iteration: {it}')
			logging.info(f'non zero count: lower > {sps.csr_matrix.count_nonzero(X_a)}, upper > {sps.csr_matrix.count_nonzero(X_b)}')

			#print('sub problem iteration', it, '\nlow active set has sizes: ',sps.csr_matrix.count_konzero(X_a), '\nup active set has size: ', sps.csr_matrix.count_konzero(X_b))
			if it > 1:
				# output the count of the indices that changed
				logging.info(f'> lower active set differ from previous by: {self.compare_active_sets(X_a, X_a_old)} indices')
				logging.info(f'> upper active set differ from previous by: {self.compare_active_sets(X_b, X_b_old)} indices')
				# stopping criterion
				if self.same_active_sets(X_a, X_a_old) and self.same_active_sets(X_b, X_b_old):
					# if the condition is met we've reached optimality
					logging.info('CONVERGENCE REACHED :)')
					break
				
			if it > max_it:
				logging.error('NO CONVERGENCE REACHED!')
				return 1

			# RHS of prjection formula

			I_a_b = sps.identity(time_size) - X_a - X_b		

			b_control_array	= X_a.dot(self.U_a) + X_b.dot(self.U_b)

			# assemble ( (3 x size x num_steps) x (3 x size x num_steps) ) sparse lhs of the linear system by blocks
			Global_matrix = sps.bmat(
				[	[None,												A_sp_blocks,		- B_sp_blocks 			],
					[A_sp_blocks.transpose(),							- B_adj_sp_blocks,	None					],
					[I_a_b.dot(M_1_sp_inv_blocks).dot(M_2_sp_blocks),	None,				sps.identity(time_size)	]	],
				format='csr')

			# assemble (3 x size x num_steps) dense rhs of linear system 
			Global_right_term = np.hstack(
				[	b_array,
					b_adj_array,
					b_control_array	]	)

			# compute solution of sparse linear system	
			time1 = time.perf_counter()
			sol_sp = spla.spsolve(Global_matrix, Global_right_term)
			time2 = time.perf_counter()
			logging.info(f'sparse system of size {np.shape(sol_sp)[0]} x {np.shape(sol_sp)[0]} solved in: {time2 - time1} s')

			# split to vector of state, control and adjoint state
			p_vec = sol_sp[ 			: time_size   ]
			y_vec = sol_sp[ time_size 	: 2*time_size ] 
			u_vec = sol_sp[ 2*time_size :			  ]

			proj = - M_1_sp_inv_blocks.dot(M_2_sp_blocks).dot(np.copy(p_vec))

			X_a_old = X_a.copy()
			X_b_old = X_b.copy()

			it += 1
		
		# split to vector of state, control and adjoint state and store it adding initial and final known values
		# now the vectors have size (num_steps + 1) x size
		p_vec = np.hstack( ( p_vec,  p_final ) )
		y_vec = np.hstack( ( y_init, y_vec 	 ) )	
		u_vec = np.hstack( ( u_vec,  u_final ) ) 

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
					additional.vector().set_local(solution_vec[n*self.mesh_size : (n + 1)*self.mesh_size])
				else:
					additional.vector().set_local(solution_vec[n*self.mesh_size : ])

				solution[n].assign(additional)

		logging.debug(f'the maximum values are: state {np.max(y_vec)}, control {np.max(u_vec)}, adjoint {np.max(p_vec)}')
		# logging.debug(f'the maximum values are: state {y_t[0](0.5)}, control {u_t[0](0.5)}, adjoint {p_t[0](0.5)}')

		# store optimal triple
		self.y, self.u, self.p = y_t, u_t, p_t

		# output time at end of algorithm
		end_time = time.perf_counter()
		logging.info(f'elapsed time: {end_time - start_time} s\n')

		return 0

	def compute_active_sets(self, proj):	

		"""

		Parameters
		----------
		proj : numoy.ndarray
			1D array of shape (num_steps * mesh_size,) to project in the admissible set.

		Returns
		-------
		tuple
			tuple of scipy.sparse.csr.csr_matrix with two entries of shape (num_steps * mesh_size, num_steps * mesh_size).
			The first entry of the tuple is the active indices matrix for the lower constraint, the second for the upper.
			They are diagonal matrices, entry on the diagonal is 1 if the index at given time is active, 0 otherwise.

		"""

		act_lower, act_upper = np.zeros(self.mesh_size), np.zeros(self.mesh_size)

		for n in range(self.num_steps):

			# access block with discretize constraints
			lower_vec = self.U_a[n*self.mesh_size : (n + 1)*self.mesh_size]
			upper_vec = self.U_b[n*self.mesh_size : (n + 1)*self.mesh_size]

			to_project = proj[n*self.mesh_size : (n + 1)*self.mesh_size]

			# vectors telling us whether the box constraints are active (iff entry is strictly greater than zero)
			act_lower = np.sign(lower_vec - to_project)
			act_upper = np.sign(to_project - upper_vec)	

			# replace only negative values with 0
			act_lower[act_lower < 0] = 0
			act_upper[act_upper < 0] = 0

			# assemble sparse diagonal matrices corresponding to active sets at time n * dt
			X_lower = sps.diags(
				act_lower, 
				0,
				shape=(self.mesh_size, self.mesh_size),
				format="csr" )
			X_upper = sps.diags(
				act_upper, 
				0,
				shape=(self.mesh_size, self.mesh_size),
				format="csr" )

			if n == 0:
				X_lower_blocks = X_lower
				X_upper_blocks = X_upper
			else:
				X_lower_blocks = sps.bmat(
					[ 	[X_lower_blocks, None],
						[None, X_lower]	],
					format="csr")
				X_upper_blocks = sps.bmat(
					[ [X_upper_blocks, None],
					[None, X_upper]],
					format="csr")

		# returns dictionary of active set matrices
		return X_lower_blocks, X_upper_blocks

	def compute_final_control(self, p_final):

		"""

		Parameters
		----------
		p_final : numpy.ndarray
			Array of size mesh_size,
			space discretization of the adjoint state at final time.

		Returns
		-------
		numpy.ndarray
			1D array of shape (mesh_size,)

		"""

		self.u_a.t = self.T
		self.u_b.t = self.T

		act_lower, act_upper = np.zeros(self.mesh_size), np.zeros(self.mesh_size)

		# discretize constraints
		lower_vec = interpolate(self.u_a, self.V).vector().get_local()
		upper_vec = interpolate(self.u_b, self.V).vector().get_local()

		# vectors telling us whether the box constraints are active (iff entry is strictly greater than zero)
		act_lower = np.sign(lower_vec - p_final)
		act_upper = np.sign(p_final - upper_vec)	

		# replace only negative values with 0
		act_lower[act_lower < 0] = 0
		act_upper[act_upper < 0] = 0			

		# assemble sparse diagonal matrices corresponding to active sets
		X_lower_final = sps.diags(
			act_lower,
			0,
			shape=(self.mesh_size, self.mesh_size),
			format="csr" )
		X_upper_final = sps.diags(
			act_upper,
			0,
			shape=(self.mesh_size, self.mesh_size),
			format="csr" )

		# relies on the fact that p_final = 0
		u_final = 0
		u_final += X_lower_final.dot(interpolate(self.u_a, self.V).vector().get_local())
		u_final += X_upper_final.dot(interpolate(self.u_b, self.V).vector().get_local())

		return u_final

	@staticmethod
	def same_active_sets(X, X_old):

		"""

		Parameters
		----------
		X : scipy.sparse.csr.csr_matrix
		X_old : scipy.sparse.csr.csr_matrix

		Returns
		-------
		bool
			True if parameters are equal, False otherwise.

		"""

		diff = sps.csr_matrix.count_nonzero(X - X_old)

		return not diff > 0

	@staticmethod
	def compare_active_sets(X, X_old):

		"""

		Parameters
		----------
		X : scipy.sparse.csr.csr_matrix
		X_old : scipy.sparse.csr.csr_matrix

		Returns
		-------
		int
			count of not equal entries

		"""

		return sps.csr_matrix.count_nonzero(X - X_old)

	def plot_objective(self, path):

		"""

		Parameters
		----------
		path : str
			Identifier of the directory where to save the cost functional plots.

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		# plotting
		plt.figure()
		fig, ax = plt.subplots(1)
		
		its = np.array(range(np.shape(np.array(self.cost_func_list))[0]))

		ax.semilogy(
			its,
			self.cost_func_list,
			marker='o',
			color='b',
			linewidth=0.7,
			label=r'$J(y_n,\,u_n)$',
			markerfacecolor='none',
			markeredgecolor='lime')
		
		ax.set_title(fr'objective functional $N_T =${self.num_steps}, $N_h = ${int(self.mesh_size)}')
		ax.set(xlabel=r'SQP iteration $n$')
		ax.legend(loc="lower left")

		plt.setp( ax, xticks=its )

		fig.set_size_inches(6, 6)

		plt.savefig(path + '/cost.pdf')

		return 0

#######################################################################################
		
if __name__ == '__main__':
	
	pass