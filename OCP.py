from fenics import *
from mshr import * 
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import time
import math
import sys
import logging


class Problem:

	"""

	Attributes
	----------
	T : float
		Final time. T = num_steps * dt
	num_steps : int
		Number of timesteps.
	dt : float
		Timestep tau.
	control_type : str
		Type of control, e.g. neumann boundary, distributed or time dependant.
	mesh :

	mesh_size : int
		Number of nodes in the mesh.
	degree : int
		Degree of FEM space (1 is linear, 2 quadratic...).
	V : dolfin.function.functionspace.FunctionSpace
		State and adjoint FEM space.
	dirichlet_boundary : function
		Returns true if argument is in the Dirichlet boundary.
	bc : dolfin.fem.dirichletbc.DirichletBC
		Dirichlet boundary condition of the state space.
	bc_adj : dolfin.fem.dirichletbc.DirichletBC
		Dirichlet boundary condition of the adjoint state space.
	y_D : dolfin.function.expression.Expression
		RHS of dirichlet boundary condition of the state space.
	p_D : dolfin.function.expression.Expression
		RHS of dirichlet boundary condition of the adjoint state space.		
	neumann_boundary : function
		Returns true if argument is in the Neumann boundary.
	y_N : dolfin.function.expression.Expression
		RHS of Neumann boundary condition of the state space.
	p_N : dolfin.function.expression.Expression
		RHS of Neumann boundary condition of the adjoint state space.
	lam : float
		Regularization parameter in the cost functional.
		Lambda too small might make the control solution to explode.
	y_target : dolfin.function.expression.Expression
		Target function in the cost functional.
	beta : dolfin.function.expression.Expression
		Beta function in the RHS of the state equation.
	f : dolfin.function.expression.Expression
		Summand in the RHS of state equation, useful to construct test examples.
	g : dolfin.function.expression.Expression
		Summand in the RHS of state equation defined on the Neumann boundary, useful to construct test examples.
	y_0 : dolfin.function.expression.Expression
		Initial condition for the state.
	p_end : dolfin.function.expression.Expression
		Final condition for the adjoint state. Take p_end = 0.
	y_exact : dolfin.function.expression.Expression
		Exact solution state, needed to compute errors in constructed test examples.
	u_exact : dolfin.function.expression.Expression
		Exact solution control, needed to compute errors in constructed test examples.
	p_exact : dolfin.function.expression.Expression
		Exact solution adjoint state, needed to compute errors in constructed test examples.		

	Methods
	-------
	__init__(T, num_steps, contol_type)
	set_state_space(mesh, degree)
	set_dirichlet_boundary_conditions(y_D, p_D, dirichlet_boundary)
	set_neumann_boundary_conditions(y_N, p_N, neumann_boundary)
	set_control_space(mesh, degree)
	set_cost(lam, y_target)
	set_state_equation(beta, f, y_0, p_end)
	set_exact_solution(y_exact, u_exact, p_exact)
	compute_errors()
	compute_relative_errors()
	evaluate_cost_functional()
	to_dict_of_functions(vec)
	to_vec(D)
	visualize_1D(lower, upper, n, path)
	visualize_1D_exact(lower, upper, n, path)
	visualize_2D(path)
	spy_sparse(A)
	visualize_purely_time_dep(path)

	"""

	def __init__(self, T, num_steps, control_type):

		"""

		Parameters
		----------
		T : float
			Final time. T = num_steps * dt
		num_steps : int
			Number of timesteps.
		control_type : str
			Control type, can be 'distributed', 'neuamnn boundary' or 'time'

		"""

		self.T = T
		self.num_steps = num_steps
		self.dt = T/num_steps
		self.control_type = control_type

	def set_state_space(self, mesh, degree):

		"""

		Parameters
		----------
		mesh :
		degree : int
			Degree of FEM space (1 is linear, 2 quadratic...).

		"""

		self.mesh = mesh
		self.mesh_size = np.shape(mesh.coordinates())[0]
		self.degree = degree
		self.V = FunctionSpace(self.mesh, 'P', degree)
		if self.control_type == 'neumann boundary':
			self.boundary_mesh = BoundaryMesh(self.mesh, 'exterior')
			self.U = FunctionSpace(self.boundary_mesh, 'P', degree)

	def set_dirichlet_boundary_conditions(self, y_D, p_D, dirichlet_boundary):

		"""

		Parameters
		----------
		y_D : dolfin.function.expression.Expression
			RHS of dirichlet boundary condition of the state space.
		p_D : dolfin.function.expression.Expression
			RHS of dirichlet boundary condition of the adjoint state space.		
		dirichlet_boundary : function
			Returns true if argument is in the Dirichlet boundary.

		"""

		self.dirichlet_boundary = dirichlet_boundary
		if dirichlet_boundary == None:
			pass
		else:
			self.y_D = y_D
			self.p_D = p_D

	def set_neumann_boundary_conditions(self, y_N, p_N, neumann_boundary):

		"""
		
	    Parameters
	    ----------
		y_N : dolfin.function.expression.Expression
			RHS of Neumann boundary condition of the state space.
		p_N : dolfin.function.expression.Expression
			RHS of Neumann boundary condition of the adjoint state space.
		neumann_boundary : function
			Returns true if argument is in the Neumann boundary.

	    """
		
		self.neumann_boundary = neumann_boundary
		if neumann_boundary == None:
			pass
		else:
			self.y_N = y_N
			self.p_N = p_N

	def set_cost(self, lam, y_target):

		"""

		Parameters
		----------
		lam : float
			Regularization parameter in the cost functional.
			Lambda too small might make the control solution to explode.
		y_target : dolfin.function.expression.Expression
			Target function in the cost functional.

		"""

		self.lam = lam
		self.y_target = y_target

	def set_state_equation(self, beta, f, g, y_0, p_end):

		"""

		Parameters
		----------
		beta : dolfin.function.expression.Expression
			Beta function in the RHS of the state equation.
		f : dolfin.function.expression.Expression
			Summand in the RHS of state equation, useful to construct test examples.
		g : dolfin.function.expression.Expression
			Neumann boundary summand in the RHS of state equation, useful to construct test examples.
		y_0 : dolfin.function.expression.Expression
			Initial condition for the state.
		p_end : dolfin.function.expression.Expression
			Final condition for the adjoint state. Take p_end = 0.
			
		"""

		self.beta = beta
		self.f = f
		self.g = g
		self.y_0 = y_0
		self.p_end = p_end

	def set_exact_solution(self, y_exact, u_exact, p_exact):

		"""

		Parameters
		----------
		y_exact : dolfin.function.expression.Expression
			Exact solution state, needed to compute errors in constructed test examples.
		u_exact : dolfin.function.expression.Expression
			Exact solution control, needed to compute errors in constructed test examples.
		p_exact : dolfin.function.expression.Expression
			Exact solution adjoint state, needed to compute errors in constructed test examples.		

		"""

		self.y_exact = y_exact
		self.u_exact = u_exact
		self.p_exact = p_exact

	def compute_errors(self):

		"""

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		print('computing errors...')

		# L2 error
		L2_error_norm_sq = [0, 0, 0]
		self.u_exact.t, self.y_exact.t, self.p_exact.t = 0.0, 0.0, 0.0
		exact = [self.y_exact, 	self.u_exact,	self.p_exact]
		sol = [self.y, 			self.u,			self.p]
 
		for n in range(self.num_steps + 1 ):

			for i, (cont, discrete) in enumerate(zip(exact, sol)):

				if i == 1:
					if self.control_type == 'neumann_boundary':
						L2_error_norm_sq[i] += self.dt * errornorm(cont, discrete[n], mesh=self.boundary_mesh)**2
					elif self.control_type == 'time':
						# extrapolate constant values
						cont_0 = interpolate(cont, self.V).vector().get_local()[0]
						discrete_0 = interpolate(discrete[n], self.V).vector().get_local()[0]
						L2_error_norm_sq[i] += self.dt * (cont_0 - discrete_0)**2
					elif self.control_type == 'distributed':# or self.control_type == 'time':
						L2_error_norm_sq[i] += self.dt * errornorm(cont, discrete[n], mesh=self.mesh)**2
				else:
					L2_error_norm_sq[i] += self.dt * errornorm(cont, discrete[n], mesh=self.mesh)**2

				cont.t += self.dt
		
		# if self.control_type == 'time':
		# 	self.L2_error_norm = [sqrt(L2_error_norm_sq[0]), sqrt(L2_error_norm_sq[1]), sqrt(L2_error_norm_sq[2])]
		# else:
		self.L2_error_norm = np.sqrt(L2_error_norm_sq)

		logging.info( f'L2-errors w.r.t. number of timesteps {self.num_steps} and number of meshpoints {self.mesh_size} :' )
		names = ['state', 'control', 'adjoint']
		for i in range(3):
			logging.info( f'{names[i]} ------------ {self.L2_error_norm[i]}')
			
		print('\n')

		return 0

	def compute_relative_errors(self):

		"""

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		print('computing errors...')

		# L2 relative error
		L2_error_norm_sq = [0, 0, 0]
		exact_norm_sq = [0, 0, 0]
		self.u_exact.t, self.y_exact.t, self.p_exact.t = 0.0, 0.0, 0.0
		exact = [self.y_exact, 	self.u_exact,	self.p_exact]
		sol = [self.y, 			self.u,			self.p]
 
		for n in range(self.num_steps + 1):

			for i, (cont, discrete) in enumerate(zip(exact, sol)):

				if i == 1:
					if self.control_type == 'neumann boundary':
						exact_norm_sq[i] += self.dt * norm(cont, 'L2', mesh=self.boundary_mesh)**2
						L2_error_norm_sq[i] += self.dt * errornorm(cont, discrete[n], mesh=self.boundary_mesh)**2

					elif self.control_type == 'time':
						cont_0 = interpolate(cont, self.V).vector().get_local()[0]
						discrete_0 = interpolate(discrete[n], self.V).vector().get_local()[0]
						exact_norm_sq[i] += self.dt * cont_0**2
						L2_error_norm_sq[i] += self.dt * (cont_0 - discrete_0)**2
						
					elif self.control_type == 'distributed':
						exact_norm_sq[i] += self.dt * norm(cont, 'L2', mesh=self.mesh)**2
						L2_error_norm_sq[i] += self.dt * errornorm(cont, discrete[n], mesh=self.mesh)**2
				else:
					exact_norm_sq[i] += self.dt * norm(cont, 'L2', mesh=self.mesh)**2
					L2_error_norm_sq[i] += self.dt * errornorm(cont, discrete[n], mesh=self.mesh)**2

				cont.t += self.dt
				
		self.L2_error_norm = np.sqrt(L2_error_norm_sq)/np.sqrt(exact_norm_sq)

		logging.info( f'L2 relative errors w.r.t. number of timesteps {self.num_steps} and number of meshpoints {self.mesh_size} :' )
		names = ['state', 'control', 'adjoint']
		for i in range(3):
			logging.info( f'{names[i]} ------------ {self.L2_error_norm[i]}')

		print('\n')

		return 0

	def compute_inf_errors(self, y, u, p):

		"""

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		# L2 relative error
		Linf_error_norm = [0, 0, 0]
		exact_norm = [0, 0, 0]
		self.u_exact.t, self.y_exact.t, self.p_exact.t = 0.0, 0.0, 0.0
		exact = [self.y_exact, self.u_exact, self.p_exact]
		sol = [y, u, p]
 
		for n in range(self.num_steps + 1):

			for i, (cont, discrete) in enumerate(zip(exact, sol)):

				if i == 1 and self.control_type == 'neumann boundary':

					Linf_error_norm[i] += self.dt * np.max(np.abs(cont.compute_vertex_values(self.boundary_mesh) - interpolate(discrete[n], self.U).compute_vertex_values(self.boundary_mesh)))
				else:
					Linf_error_norm[i] += self.dt * norm(interpolate(cont, self.V).vector() - discrete[n].vector(),'linf')

				cont.t += self.dt
				
		# Linf_error_norm = Linf_error_norm/exact_norm

		return Linf_error_norm

	def evaluate_cost_functional(self, y, u):

		try:
			y = self.to_dict_of_functions(np.hstack((self.y_init, y)))
		except:
			type(y) == dict

		try:
			u = self.to_dict_of_functions(np.hstack((u, self.u_final)))
		except:
			type(u) == dict

		J_y, J_u = 0, 0
		self.y_target.t = 0

		for i in range(self.num_steps + 1):

			J_y += self.dt * errornorm(self.y_target, y[i], mesh=self.mesh)**2

			if self.control_type == 'neumann boundary':
				J_u += self.lam * self.dt * norm(interpolate(u[i], self.U) , 'L2', mesh=self.boundary_mesh)**2
			else:
				J_u += self.lam * self.dt * norm(u[i] , 'L2', mesh=self.mesh)**2

			self.y_target.t += self.dt

		self.y_target.t = 0

		return .5 * (J_y + J_u), .5 * J_y, .5 * J_u


	def to_dict_of_functions(self, vec):

		"""

		Parameters
		----------
		vec : numpy.ndarray

		Returns
		-------
		dict[int, dolfin.function.function.Function]
			keys from 0 to num_steps, 
			values are functions at key-th timestep.


		"""

		D = { n : Function(self.V) for n in range(self.num_steps + 1) }
		additional = Function(self.V)
		
		for n in range(self.num_steps + 1):

			if n < self.num_steps:
				additional.vector().set_local(
					vec[n*self.mesh_size : (n + 1)*self.mesh_size])
			else:
				additional.vector().set_local(
					vec[n*self.mesh_size : ])
			
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
			Arrat of size (mesh_size*(num_steps + 1)).

		"""

		vec = np.zeros(self.mesh_size*(self.num_steps + 1))

		for n in range(self.num_steps + 1):

			if n < self.num_steps:
				vec[n*self.mesh_size : (n + 1)*self.mesh_size] = D[n].vector().get_local()
			else:
				vec[n*self.mesh_size : ] = D[n].vector().get_local()

		return vec

############################# visualization ################################

	# 1D plotting of computed solutions
	def visualize_1D(self, lower, upper, n, path):

		"""

		Parameters
		----------
		lower : float
		upper : float
		n : int
		path : str

		Returns
		-------
		int
			0 if successful, 1 otherwise

		"""

		if not ((n & (n-1) == 0) and n != 0):
			logging.error(f'insert power of 2 to visualize 1D solution')
			return 1

		new_dt = self.T / n

		space = np.linspace( lower, upper, num=n + 1 )
		
		X, Y = np.meshgrid( space, np.array([range(n + 1)]) )

		state_colors, control_colors, adjoint_colors = np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1))

		# assemble arrays of colors
		for t in range(n + 1):
			time  = int(self.num_steps * t/n)
			# print(time)
			for x in range(n + 1):
				state_colors[t, x] = self.y[time](space[x])
				control_colors[t, x] = self.u[time](space[x])
				adjoint_colors[t, x] = self.p[time](space[x])

		vmin = np.min(
			[	np.min(state_colors),
				np.min(control_colors),
				np.min(adjoint_colors)	]	)
		vmax = np.max(
			[	np.max(state_colors),
				np.max(control_colors),
				np.max(adjoint_colors)	]	)
		
		# plotting
		plt.figure()
		fig, (ax1, ax2, ax3) = plt.subplots(3)

		# control
		plot_control = ax1.scatter(
			X,
			new_dt * Y,
			s=0.5,
			c=control_colors,
			# vmin=vmin, vmax=vmax,
			cmap='jet')#plt.cm.get_cmap('CMRmap', 5))
		ax1.set_title('control')
		ax1.set(xlabel='x', ylabel='t')
		ax1.set(
			xlim=(lower, upper),
			ylim=(0, self.T))
		plt.colorbar(
			plot_control,
			ax=ax1,
			ticks=[np.min(control_colors), np.max(control_colors)])

		# state
		plot_state = ax2.scatter(
			X,
			new_dt * Y,
			s=0.5,
			c=state_colors,
			# vmin=vmin, vmax=vmax,
			cmap='jet')#, marker='s')
		ax2.set_title('state')
		ax2.set(xlabel='x', ylabel='t')
		ax2.set(
			xlim=(lower, upper),
			ylim=(0, self.T))
		plt.colorbar(
			plot_state,
			ax=ax2,
			ticks=[np.min(state_colors), np.max(state_colors)])

		# adjoint state
		plot_adjoint = ax3.scatter(
			X,
			new_dt * Y,
			s=0.5,
			c=adjoint_colors,
			# vmin=vmin, vmax=vmax,
			cmap='jet')#, marker='s')
		ax3.set_title('adjoint')
		ax3.set(xlabel='x', ylabel='t')
		ax3.set(
			xlim=(lower, upper),
			ylim=(0, self.T))
		plt.colorbar(
			plot_adjoint,
			ax=ax3,
			ticks=[np.min(adjoint_colors), np.max(adjoint_colors)])

		plt.setp(
			(ax1, ax2, ax3),
			xticks=[lower, 0.5*(lower + upper), upper],
			yticks=[0, self.T])
		# adjust dimension and distance
		fig.subplots_adjust(hspace=0.3)
		fig.set_size_inches(5, 12)

		plt.savefig(path + '/graphs.pdf')

		return 0

	# 1D plotting of exact solutions
	def visualize_1D_exact(self, lower, upper, n, path):

		"""

		Parameters
		----------
		lower : float
		upper : float
		n : int
		path : str

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		if not ((n & (n-1) == 0) and n != 0):
			logging.error(f'insert power of 2 to visualize 1D solution')
			return 1

		new_dt = self.T / n

		if self.control_type == 'neumann boundary':
			diam = upper - lower
			h = diam/self.mesh_size

		self.y_exact.t, self.u_exact.t, self.p_exact.t = 0, 0, 0

		space = np.linspace( lower, upper, num=n + 1 )
	
		X, Y = np.meshgrid( space, np.array([range(n + 1)]) )

		state_colors, control_colors, adjoint_colors = np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1)), np.zeros((n + 1, n + 1))

		# assemble arrays of colors
		for t in range(n + 1):
			time  = int(self.num_steps * t/n)
			# print(time)
			for x in range(n + 1):
				state_colors[t, x] = self.y_exact(space[x])
				adjoint_colors[t, x] = self.p_exact(space[x])
				control_colors[t, x] = self.u_exact(space[x])

			self.y_exact.t += new_dt
			self.u_exact.t += new_dt
			self.p_exact.t += new_dt

		vmin = np.min(
			[	np.min(state_colors),
				np.min(control_colors),
				np.min(adjoint_colors)	]	)
		vmax = np.max(
			[	np.max(state_colors),
				np.max(control_colors),
				np.max(adjoint_colors)	]	)

		# plotting
		plt.figure()
		fig, (ax1, ax2, ax3) = plt.subplots(3)

		# control
		plot_control = ax1.scatter(
			X,
			new_dt * Y,
			s=.5,
			c=control_colors,
			# vmin=vmin,
			# vmax=vmax,
			cmap='jet')
		ax1.set_title('exact control')
		ax1.set(xlabel='x', ylabel='t')
		ax1.set(
			xlim=(lower, upper),
			ylim=(0, self.T))
		plt.colorbar(
			plot_control,
			ax=ax1,
			ticks=[np.min(control_colors), np.max(control_colors)])

		# state
		plot_state = ax2.scatter(
			X,
			new_dt * Y,
			s=.5,
			c=state_colors,
			vmin=vmin, 
			vmax=vmax, 
			cmap='jet')#, marker='s')
		ax2.set_title('exact state')
		ax2.set(xlabel='x', ylabel='t')
		ax2.set(
			xlim=(lower, upper),
			ylim=(0, self.T))
		plt.colorbar(
			plot_state,
			ax=ax2,
			ticks=[np.min(state_colors), np.max(state_colors)])

		# adjoint state
		plot_adjoint = ax3.scatter(
			X,
			new_dt * Y,
			s=.5,
			c=adjoint_colors,
			vmin=vmin*self.lam,
			vmax=vmax*self.lam,
			cmap='jet')#, marker='s')
		ax3.set_title('exact adjoint')
		ax3.set(xlabel='x', ylabel='t')
		ax3.set(
			xlim=(lower, upper),
			ylim=(0, self.T))
		plt.colorbar(
			plot_adjoint,
			ax=ax3,
			ticks=[np.min(adjoint_colors), np.max(adjoint_colors)])

		plt.setp(
			(ax1, ax2, ax3),
			xticks=[lower, 0.5*(lower + upper), upper],
			yticks=[0, self.T])
		# adjust dimension and distance
		fig.subplots_adjust(hspace=0.3)
		fig.set_size_inches(5, 12)

		# plt.savefig(f'visualization_coupled/graphs_1D/graphs_exact.pdf')
		plt.savefig(path + '/graphs_exact.pdf')

		return 0

	# 2D visualization with paraview
	def visualize_paraview(self, path):

		"""

		Parameters
		----------
		path : str

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		xdmffile_control = XDMFFile(path + '/control/solution_control.xdmf')
		xdmffile_state = XDMFFile(path + '/state/solution_state.xdmf')
		xdmffile_adj = XDMFFile(path + '/adj/solution_adj.xdmf')

		xdmffile_exact_control = XDMFFile(path + '/control/exact_solution_control.xdmf')
		xdmffile_exact_state = XDMFFile(path + '/state/exact_solution_state.xdmf')
		xdmffile_exact_adj = XDMFFile(path + '/adj/exact_solution_adj.xdmf')


		t = 0

		add_state, add_adj, add_control = Function(self.V), Function(self.V), Function(self.V)
		add_exact_state, add_exact_adj = Function(self.V), Function(self.V)
		
		if self.control_type == 'neumann boundary':
			add_exact_control = Function(self.U)
			# add_control = Function(self.U)
		else:
			add_exact_control = Function(self.V)
			# add_control = Function(self.V)

		for i in range(self.num_steps + 1):

			add_state.assign(self.y[i])
			add_adj.assign(self.p[i])
			add_control.assign(self.u[i])

			# Save all control functions to file
			xdmffile_control.write(add_control, t)
			# Save all state functions to file
			xdmffile_state.write(add_state, t)
			# Save all adj state functions to file
			xdmffile_adj.write(add_adj, t)

			add_exact_state.assign(project(self.y_exact, self.V))
			add_exact_adj.assign(project(self.p_exact, self.V))

			if self.control_type == 'neumann boundary':
				add_exact_control.assign(project(self.u_exact, self.U))
			else:
				add_exact_control.assign(project(self.u_exact, self.V))


			# Save all control functions to file
			xdmffile_exact_control.write(add_exact_control, t)
			# Save all state functions to file
			xdmffile_exact_state.write(add_exact_state, t)
			# Save all adj state functions to file
			xdmffile_exact_adj.write(add_exact_adj, t)

			# update time
			t += self.dt
			self.y_exact.t, self.u_exact.t, self.p_exact.t = t, t, t

		return 0

	def visualize_purely_time_dep(self, path):


		"""

		Parameters
		----------
		path : str

		Returns
		-------
		int
			0 if successful, 1 otherwise

		"""

		control_list = []
		time_list = []
		gradient_list = []
		t=0
		for i in range(self.num_steps + 1):
			# extrapolate constant values
			control_list.append(interpolate(self.u[i], self.V).vector().get_local()[0])
			time_list.append(t)
			gradient_list.append(self.gradient_term[i])
			t += self.dt

		# plotting
		plt.figure()
		fig, ax = plt.subplots(1, constrained_layout=True)

		# gradient term
		plot_gradient = ax.plot(
			time_list,
			gradient_list,
			label=r'$-\frac{1}{\lambda}(B^*p)(t)$',
			# marker='D',
			linewidth=0.5,
			linestyle='--',
			color='0.5')
		# control
		plot_control = ax.plot(
			time_list,
			control_list,
			label=r'$u(t)$',
			# marker='D',
			linewidth=0.7,
			color='b')
		

		# ax.set_title('purely time dep. control')
		ax.set(ylabel='control', xlabel='t')
		ax.set(
			# ylim=(min(control_list), max(control_list)),
			xlim=(0, self.T))

		ax.legend(loc="lower left")

		plt.savefig(path + '/control.pdf')

		return 0



	@staticmethod
	def spy_sparse(A):

		"""

		Parameters
		----------
		A : csr_matrix

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		non_zero_count = A.count_nonzero()
		rows, columns = A.get_shape()

		plt.figure()
		fig, ax = plt.subplots(1, constrained_layout=True)
		# plotting
		plt.spy(
			A,
			markersize=0.01,
			color='k',
			precision=1e-4)
		# compute percentage of non zero entries
		density = non_zero_count/(rows * columns) * 100
		# rounding
		density = float('%.2g' % density)
		ax.set_xticks([0, columns])
		ax.set_yticks([0, rows])
		ax.set_title(f'density {density}%')
		plt.savefig(f'sparse_with_density_{density}%.pdf')

		return 0

if __name__ == '__main__':
	level = logging.INFO
	fmt = '[%(levelname)s] %(asctime)s - %(message)s'
	logging.basicConfig(level=level, format=fmt)
	
	
	



	

	
