from fenics import *
from mshr import * 
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import math
import logging

class Linear_PDE_t:

	"""

	Attributes
	----------
	T : float
		Final time. T = num_steps * dt
	num_steps : int
		Number of timesteps.
	dt : float
		Timestep tau.
	mesh :

	mesh_size : int
		Number of nodes in the mesh.
	degree : int
		Degree of FEM space (1 is linear, 2 quadratic...).
	V : dolfin.function.functionspace.FunctionSpace
		FEM space.
	dirichlet_boundary : function
		Returns true if argument is in the Dirichlet boundary.
	bc : dolfin.fem.dirichletbc.DirichletBC
		Dirichlet boundary condition of the state space.
	u_D : dolfin.function.expression.Expression
		RHS of dirichlet boundary condition of the state space.
	neumann_boundary : function
		Returns true if argument is in the Neumann boundary.
	u_N : dolfin.function.expression.Expression
		RHS of Neumann boundary condition of the state space.
		Lambda too small might make the control solution to explode.
	f : dolfin.function.expression.Expression
		Summand in the RHS of state equation defined on the domain, useful to construct test examples.
	g : dolfin.function.expression.Expression
		Summand in the RHS of state equation defined on the Neumann boundary, useful to construct test examples.
	u_0 : dolfin.function.expression.Expression
		Initial condition.
	u_exact : dolfin.function.expression.Expression
		Exact solution, needed to compute errors in constructed test examples.

	Methods
	-------
	__init__(T, num_steps)
	set_space(mesh, degree)
	set_dirichlet_boundary_conditions(u_D,, dirichlet_boundary)
	set_neumann_boundary_conditions(u_N,, neumann_boundary)
	set_equation(f, g, u_0)
	set_exact_solution(u_exact)
	compute_errors()
	compute_relative_errors()
	to_dict_of_functions(vec)
	to_vec(D)
	visualize_1D(lower, upper, n, path)
	visualize_1D_exact(lower, upper, n, path)
	visualize_2D(path)
	spy_sparse(A)

	"""

	def __init__(self, T, num_steps):

		"""

		Parameters
		----------
		T : float
			Final time. T = num_steps * dt
		num_steps : int
			Number of timesteps.

		"""

		self.T = T
		self.num_steps = num_steps
		self.dt = T/num_steps

	def set_space(self, mesh, degree):

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

	def set_dirichlet_boundary_conditions(self, u_D, dirichlet_boundary):

		"""

		Parameters
		----------
		u_D : dolfin.function.expression.Expression
			RHS of dirichlet boundary condition.		
		dirichlet_boundary : function
			Returns true if argument is in the Dirichlet boundary.

		"""

		self.dirichlet_boundary = dirichlet_boundary
		if dirichlet_boundary == None:
			pass
		else:
			self.u_D = u_D

	def set_neumann_boundary_conditions(self, u_N, neumann_boundary):

		"""
	    Parameters
	    ----------
		u_N : dolfin.function.expression.Expression
			RHS of Neumann boundary condition.
		neumann_boundary : function
			Returns true if argument is in the Neumann boundary.

	    """
		
		self.neumann_boundary = neumann_boundary
		if neumann_boundary == None:
			pass
		else:
			self.u_N = u_N

	# def set_control_space(self, mesh, degree):
	# 	self.mesh_control = mesh
	# 	self.U = FunctionSpace(self.mesh_control, 'P', degree)

	def set_equation(self, f, g, u_0):

		"""

		Parameters
		----------
		f : dolfin.function.expression.Expression
			Summand in the RHS of state equation, useful to construct test examples.
		g : dolfin.function.expression.Expression
			Neumann boundary summand in the RHS of state equation, useful to construct test examples.
		u_0 : dolfin.function.expression.Expression
			Initial condition.
			
		"""

		self.f = f
		self.g = g
		self.u_0 = u_0

	def set_exact_solution(self, u_exact):

		"""

		Parameters
		----------
		
		u_exact : dolfin.function.expression.Expression
			Exact solution, needed to compute errors in constructed test examples.	

		"""

		self.u_exact = u_exact

	def compute_errors(self):

		"""

		Returns
		-------
		int
			0 if successful, 1 otherwise.

		"""

		print('computing errors...')

		# L2 error
		L2_error_norm_sq = 0
		H1_error_norm_sq = 0
		self.u_exact.t = 0
		
		for n in range(self.num_steps + 1 ):

				L2_error_norm_sq += self.dt * errornorm(self.u_exact, self.u[n], mesh=self.mesh, norm_type='L2')**2
				H1_error_norm_sq += self.dt * errornorm(self.u_exact, self.u[n], mesh=self.mesh, norm_type='H1')**2

				self.u_exact.t += self.dt
				
		self.L2_error_norm = np.sqrt(L2_error_norm_sq)
		self.H1_error_norm = np.sqrt(H1_error_norm_sq)

		logging.info( f'L2-errors w.r.t. number of timesteps {self.num_steps} and number of meshpoints {self.mesh_size}: {self.L2_error_norm}' )
		logging.info( f'H1 relative errors w.r.t. number of timesteps {self.num_steps} and number of meshpoints {self.mesh_size}: {self.H1_error_norm}')
			
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
		L2_error_norm_sq = 0
		# H1_error_norm_sq = 0
		exact_norm_sq = 0
		# exact_H1_norm_sq = 0
		self.u_exact.t = 0
		
		for n in range(self.num_steps + 1):

			exact_norm_sq += self.dt * norm(self.u_exact, 'L2', mesh=self.mesh)**2
			# exact_H1_norm_sq += self.dt * norm(self.u_exact, 'H1', mesh=self.mesh)**2
			L2_error_norm_sq += self.dt * errornorm(self.u_exact, self.u[n], mesh=self.mesh, norm_type='L2')**2
			# H1_error_norm_sq += self.dt * errornorm(self.u_exact, self.u[n], mesh=self.mesh, norm_type='H1')**2

			self.u_exact.t += self.dt
				
		self.L2_error_norm = np.sqrt(L2_error_norm_sq)/np.sqrt(exact_norm_sq)
		# self.H1_error_norm = np.sqrt(H1_error_norm_sq)/np.sqrt(exact_H1_norm_sq)

		logging.info( f'L2 relative errors w.r.t. number of timesteps {self.num_steps} and number of meshpoints {self.mesh_size}: {self.L2_error_norm}')
		# logging.info( f'H1 relative errors w.r.t. number of timesteps {self.num_steps} and number of meshpoints {self.mesh_size}: {self.H1_error_norm}')

		print('\n')

		return 0

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

		colors = np.zeros((n + 1, n + 1))

		# assemble arrays of colors
		for t in range(n + 1):
			time  = int(self.num_steps * t/n)
			# print(time)
			for x in range(n + 1):
				colors[t, x] = self.u[time](space[x])
		
		# plotting
		plt.figure()
		fig, ax = plt.subplots(1)

		# control
		plot_control = ax.scatter(
			X,
			new_dt * Y,
			s=1,
			c=colors,
			cmap='jet')#plt.cm.get_cmap('CMRmap', 5))

		# plot_solution = ax1.scatter(X, new_dt * Y, s=1, c=colors, vmin=vmin, vmax=vmax, cmap='jet')#plt.cm.get_cmap('CMRmap', 5))
		ax.set_title('solution')
		ax.set(xlabel='x', ylabel='t')
		ax.set(
			xlim=(lower, upper),
			ylim=(0, self.T))
		plt.colorbar(
			plot_control,
			ax=ax,
			ticks=[
				np.min(colors),
				0,
				np.max(colors)])

		plt.setp(
			ax,
			xticks=[lower, 0.5*(lower + upper), upper],
			yticks=[0, self.T])
		# adjust dimension and distance
		fig.subplots_adjust(hspace=0.3)
		fig.set_size_inches(5, 4)

		# plt.savefig(f'visualization_coupled/graphs_1D/graphs_{lam}.pdf')
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

		self.u_exact.t

		space = np.linspace( lower, upper, num=n + 1 )
	
		X, Y = np.meshgrid( space, np.array([range(n + 1)]) )

		colors = np.zeros((n + 1, n + 1))

		# assemble arrays of colors
		for t in range(n + 1):
			time  = int(self.num_steps * t/n)
			# print(time)
			for x in range(n + 1):
		
				colors[t, x] = self.u_exact(space[x])

			self.u_exact.t += new_dt

		# plotting
		plt.figure()
		fig, ax = plt.subplots(1)

		# control
		plot_control = ax.scatter(
			X,
			new_dt * Y,
			s=1,
			c=colors,
			vmin=np.min(colors),
			vmax=np.max(colors),
			cmap='jet')#plt.cm.get_cmap('CMRmap', 5))
		ax.set_title('exact solution')
		ax.set(xlabel='x', ylabel='t')
		ax.set(
			xlim=(lower, upper),
			ylim=(0, self.T))
		plt.colorbar(
			plot_control,
			ax=ax,
			ticks=[
				np.min(colors),
				0,
				np.max(colors)])

		# ax4.imshow(colors, vmin=vmin, vmax=vmax, cmap='jet')

		# fig.colorbar(plot_adjoint, ax=[ax1, ax2, ax3])
		plt.setp(
			ax,
			xticks=[lower, 0.5*(lower + upper), upper],
			yticks=[0, self.T])
		# adjust dimension and distance
		fig.subplots_adjust(hspace=0.3)
		fig.set_size_inches(5, 4)

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

		xdmffile_solution = XDMFFile(path + '/solution.xdmf')

		xdmffile_exact_solution = XDMFFile(path + '/exact_solution.xdmf')

		t = 0

		add_solution = Function(self.V)
		add_exact_solution = Function(self.V)
		
		for i in range(self.num_steps + 1):

			add_solution.assign(self.u[i])
			# Save all solution functions to file
			xdmffile_solution.write(add_solution, t)

			add_exact_solution.assign(project(self.u_exact, self.V))
			# Save all exact solution functions to file
			xdmffile_exact_solution.write(add_exact_solution, t)

			# update time
			t += self.dt
			self.u_exact.t = t

		return 0

if __name__ == '__main__':

	pass


