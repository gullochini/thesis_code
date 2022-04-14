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
import orders
import sqp_quasi_linear
import semi_smooth_quasi_linear

######################### MESH INDEPENDENCE ########################

def plot_1D_t_increments_vary_mesh_sqp(lam):
	## python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	g = Constant(0.0)
	T = 1.0
	# beta = Constant(1.0)
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
	
	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-08, 1e-8, 20, 20

	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	mesh_list = [ [5, 25],[10, 100], [20, 400], [40, 1600] ]
	colors = ['orange','darkorange', 'orangered', 'red']
	increment_list = []

	total_time_start = time.perf_counter()

	# initialize old error listtotal_time_start = time.perf_counter()
	old_errors = [100, 100, 100]

	for pair in mesh_list:
		# build mesh
		# 1D
		new_mesh = IntervalMesh(pair[0], 0.0, 1.0)
		# initialize instance
		P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, pair[1], 'time')
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
		# call solver
		P.sqp_solve()
		
		# store increments
		increment_list.append(np.array(P.incr_list))

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP AND MESHSIZE: {total_time_end - total_time_start} s')

	n_plots = np.shape(np.array(increment_list))[0]

	# plotting
	fig = plt.figure(figsize=(6,8))
	fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)
	for i in range(2):
		for j in range(2):

			its = np.arange(1, 1 + len(increment_list[2*i + j])) 

			ax[i, j].semilogy(
				its, 
				increment_list[2*i + j], 
				marker='o', 
				color=colors[2*i + j],
				linewidth=0.7, 
				label=fr'incr$_n$, $N_h = {mesh_list[2*i + j][0] + 1}, N_T = {mesh_list[2*i + j][1]}$', 
				markerfacecolor='none', 
				markeredgecolor=colors[2*i + j])

			if i == 1:
				ax[i, j].set(xlabel=r'SQP iteration $n$')

			ax[i, j].legend(loc="lower left")

	plt.savefig(f'visualization_sqp/increments_1D_vary_mesh.pdf')

	return 0


def plot_1D_t_increments_vary_mesh_semismooth(lam):
	## python functions 
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

	mu, tol_nm, tol_sm, max_it_nm, max_it_sm = 1, 1e-10, 1e-7, 20, 20

	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	mesh_list = [ [5, 25], [10, 100], [20, 400], [40, 1600] ]
	# mesh_list = [ [6, 25],[11, 100], [6, 25],[11, 100] ]
	colors = ['orange','darkorange', 'orangered', 'red']
	# mesh_size_list = [5, 10, 20, 40, 80]
	increment_list = []

	total_time_start = time.perf_counter()

	# initialize old error listtotal_time_start = time.perf_counter()
	old_errors = [100, 100, 100]

	for pair in mesh_list:
		# build mesh
		# 1D
		new_mesh = IntervalMesh(pair[0], 0.0, 1.0)
		# initialize instance
		P = semi_smooth_quasi_linear.Quasi_Linear_Problem_Box(T, pair[1], 'time')
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
		
		# store increments
		increment_list.append(np.array(P.incr_list))

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP AND MESHSIZE: {total_time_end - total_time_start} s')

	n_plots = np.shape(np.array(increment_list))[0]

	# plotting
	fig = plt.figure(figsize=(6,8))
	fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)

	for i in range(2):
		for j in range(2):

			its = np.arange(1, 1 + len(increment_list[2*i + j])) 

			ax[i, j].semilogy(
				its, 
				increment_list[2*i + j], 
				marker='o', 
				color=colors[2*i + j],
				linewidth=0.7, 
				label=fr'incr$_n$, $N_h = {mesh_list[2*i + j][0] + 1}, N_T = {mesh_list[2*i + j][1]}$', 
				markerfacecolor='none', 
				markeredgecolor=colors[2*i + j])

			if i == 1:
				ax[i, j].set(xlabel=r'Semi-smooth iteration $n$')

			ax[i, j].legend(loc="lower left")

	plt.savefig(f'visualization_semi_smooth_quasi_linear/increments_1D_vary_mesh.pdf')

	return 0

######################### ERROR ESTIMATES ##########################

########################## distributed #########################


def plot_1D_vary_dt(lam):
	## python functions 
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
	y_0   	= Expression(' sin(pi*x[0])', degree=5)
	p_end 	= Constant(0.0) 

	y_target= Expression(' cos(2*pi*t) * sin(pi*x[0]) * (1 - 2*lam*pi) - lam*pow(pi,2) * sin(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) - sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) + lam * pow(pi,2)*sin(2*pi*t)*cos(2*pi*t)*pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) * (   pow(cos(pi*x[0]),2)* ( sin(pi*x[0])*cos(2*pi*t)* ( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) - 1 ) - pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) - 1 ) + pow(sin(pi*x[0]), 2)*( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) + 1 )	) * pow( pow( exp(1),cos(2*pi*t)*sin(pi*x[0]) ) + 1, -3) ', t=0, degree=5, lam=lam)

	y_exact = Expression(' 		 		cos(2*pi*t) * sin(pi*x[0])', t=0, degree=5)
	p_exact = Expression('		 - lam* sin(2*pi*t) * sin(pi*x[0])', t=0, degree=5, lam=lam)
	
	# space and time dependant constraints
	u_b = Expression(' 2 * (0.2 + 0.5 * x[0]) * t', t=0, degree=2)
	u_a = Expression(' - 2 * (0.2 + 0.5 * x[0]) * (1 - t)', t=0, degree=2)

	u_exact = Expression('( sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ? sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ) < 2 * t * (0.2 + 0.5 * x[0]) ? (sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ? sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t)) : 2 * t * (0.2 + 0.5 * x[0])', t=0, degree=5)
	
	f 		= Expression('- pow(pi,2) * cos(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1), cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) - sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) - 2*pi*sin(2*pi*t) * sin(pi*x[0]) - ((sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ? sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t)) < 2 * t * (0.2 + 0.5 * x[0]) ? (sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ? sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t)) : 2 * t * (0.2 + 0.5 * x[0]))', t=0, degree=5)
	g = Constant(0.0)
	
	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-08, 1e-8, 20, 20

	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	num_steps_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]#, 8192]
	
	L2_same_mesh_list = []

	total_time_start = time.perf_counter()
	# build mesh
	# 1D
	mesh = IntervalMesh(64, 0.0, 1.0)
	
	# initialize old error list
	old_errors = [100, 100, 100]
	
	for N_steps in num_steps_list:
		# initialize instance
		P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, N_steps, 'distributed')
		# set up problem by callig class attrubutes
		P.set_state_space(mesh, 1)
		P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary)
		P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), None)
		# P.set_control_space(mesh, 1)
		P.set_control_constraints(u_a, u_b)
		P.set_cost(lam, y_target)
		P.set_state_equation(beta, f, g, y_0, p_end)
		P.set_exact_solution(y_exact, u_exact, p_exact)
		# set quasi-linear parameters
		P.set_non_linearity(mu, csi, csi_p, csi_pp)
		P.set_maxit_and_tol(tol, tol_pd, max_it_sqp, max_it_pd)
		# call solver
		P.sqp_solve()
		# compute errors
		P.compute_errors()
				
		# check if errors are stagnating
		if P.L2_error_norm[0] >= old_errors[0]:
			logging.error('state error is stagnating or increasing!')
		if P.L2_error_norm[1] >= old_errors[1]:
			logging.error('control error is stagnating or increasing!')
		if P.L2_error_norm[2] >= old_errors[2]:
			logging.error('adjoint error is stagnating or increasing!')

		if  P.L2_error_norm[0] >= old_errors[0] and P.L2_error_norm[1] >= old_errors[1] and P.L2_error_norm[2] >= old_errors[2]:
			logging.error('all errors stagnating or increasing!!')
			# store errors
			L2_same_mesh_list.append(P.L2_error_norm)
			# exit the loop
			break

		# store old errors
		old_errors = P.L2_error_norm.copy()
		# store errors
		L2_same_mesh_list.append(P.L2_error_norm)
	
	L2_same_mesh_array = np.array(L2_same_mesh_list)

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP AND MESHSIZE: {total_time_end - total_time_start} s')

	print(L2_same_mesh_array)
	print(orders.compute_orders(L2_same_mesh_array))

	# diameter = 2.0

	diameter = 1

	L = np.shape(L2_same_mesh_array)[0]

	x_time = np.linspace(num_steps_list[0], num_steps_list[ L-1 ], 	100)

	# plotting
	plt.figure()
	fig, ax1 = plt.subplots(1)

	ax1.loglog(
		num_steps_list[:L],
		L2_same_mesh_array[:,0],
		label=r'$||\, y-y_h \,||_{L^2}$',
		marker='o',
		color='r',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='r')
	ax1.loglog(
		num_steps_list[:L],
		L2_same_mesh_array[:,1],
		label=r'$||\, u-u_h \,||_{L^2}$',
		marker='s',
		color='b',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='b')
	ax1.loglog(
		num_steps_list[:L],
		L2_same_mesh_array[:,2],
		label=r'$||\, p-p_h \,||_{L^2}$',
		marker='D',
		color='lime',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='lime')
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
	plt.savefig(f'visualization_sqp/vary_dt/L2_relative_errors_1D_vary_dt.pdf')

	return 0


def plot_1D_vary_size(lam):
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
	p_exact = Expression('		 - lam* sin(2*pi*t) * sin(pi*x[0])', t=0, degree=5, lam=lam)
	
	# space and time dependant constraints
	u_b = Expression(' 2 * (0.2 + 0.5 * x[0]) * t', t=0, degree=2)
	u_a = Expression(' - 2 * (0.2 + 0.5 * x[0]) * (1 - t)', t=0, degree=2)

	u_exact = Expression('( sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ? sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ) < 2 * t * (0.2 + 0.5 * x[0]) ? (sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ? sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t)) : 2 * t * (0.2 + 0.5 * x[0])', t=0, degree=5)
	
	f 		= Expression('- pow(pi,2) * cos(2*pi*t) * ( cos(2*pi*t) * pow(cos(pi*x[0]),2) * pow(exp(1), cos(2*pi*t) * sin(pi*x[0])) * pow(pow(exp(1),cos(2*pi*t) * sin(pi*x[0])) + 1, -2) - sin(pi*x[0]) * ( pow( pow( exp(1), -cos(2*pi*t) * sin(pi*x[0]) ) + 1, -1) + 1 )) - 2*pi*sin(2*pi*t) * sin(pi*x[0]) - ((sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ? sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t)) < 2 * t * (0.2 + 0.5 * x[0]) ? (sin(2*pi*t) * sin(pi*x[0]) > - 2 * (0.2 + 0.5 * x[0]) * (1 - t) ? sin(2*pi*t) * sin(pi*x[0]) : - 2 * (0.2 + 0.5 * x[0]) * (1 - t)) : 2 * t * (0.2 + 0.5 * x[0]))', t=0, degree=5)
	g = Constant(0.0)
	
	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-08, 1e-8, 20, 20

	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	mesh_size_list = [2, 4, 8, 16, 32, 64]#, 128]#, 256]
	# mesh_size_list = [5, 10, 20, 40, 80]
	L2_same_timestep_list = []

	total_time_start = time.perf_counter()

	# initialize old error listtotal_time_start = time.perf_counter()
	old_errors = [100, 100, 100]

	for N_h in mesh_size_list:
		# build mesh
		# 1D
		new_mesh = IntervalMesh(N_h, 0.0, 1.0)
		# initialize instance
		if N_h <= 8:
			P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, 1024, 'distributed')
		elif N_h <= 32:
			P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, 2048, 'distributed')
		else:
			P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, 4096, 'distributed')
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
		# call solver
		P.sqp_solve()
		# compute errors
		P.compute_errors()
		
		# check if errors are stagnating
		if P.L2_error_norm[0] >= old_errors[0]:
			logging.error('state error is stagnating or increasing!')
		if P.L2_error_norm[1] >= old_errors[1]:
			logging.error('control error is stagnating or increasing!')
		if P.L2_error_norm[2] >= old_errors[2]:
			logging.error('adjoint error is stagnating or increasing!')

		if  P.L2_error_norm[0] >= old_errors[0] and P.L2_error_norm[1] >= old_errors[1] and P.L2_error_norm[2] >= old_errors[2]:
			logging.error('all errors stagnating or increasing!!')
			# store errors
			L2_same_mesh_list.append(P.L2_error_norm)
			# exit the loop
			break

		# store old errors
		old_errors = P.L2_error_norm.copy()
		# store errors
		L2_same_timestep_list.append(P.L2_error_norm)
		
	L2_same_timestep_array = np.array(L2_same_timestep_list)

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP AND MESHSIZE: {total_time_end - total_time_start} s')

	print(L2_same_timestep_array)
	print(orders.compute_orders(L2_same_timestep_array))

	diameter = 1

	L = np.shape(L2_same_timestep_array)[0]

	x_mesh = np.linspace(mesh_size_list[0], mesh_size_list[ L-1 ], 100)

	# plotting
	plt.figure()
	fig, ax2 = plt.subplots(1)

	ax2.loglog(
		mesh_size_list[:L],
		L2_same_timestep_array[:,0],
		label=r'$||\, y-y_h \,||_{L^2}$',
		marker='o',
		color='r',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='r')
	ax2.loglog(
		mesh_size_list[:L],
		L2_same_timestep_array[:,1],
		label=r'$||\, u-u_h \,||_{L^2}$',
		marker='s',
		color='b',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='b')
	ax2.loglog(
		mesh_size_list[:L],
		L2_same_timestep_array[:,2],
		label=r'$||\, p-p_h \,||_{L^2}$',
		marker='D',
		color='lime',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='lime')

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
	plt.savefig(f'visualization_sqp/vary_size/L2_relative_errors_1D_vary_size.pdf')

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

	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-08, 1e-8, 20, 20

	num_steps_list = [4, 8, 16, 32, 64, 128, 256]#, 512]#, 1024]
	
	L2_same_mesh_list = []
	# Linf_same_mesh_list = []

	total_time_start = time.perf_counter()

	# build mesh
	# 2D
	mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), 16, 16)
	
	# initialize old error list
	old_errors = [100, 100, 100]
	
	for N_steps in num_steps_list:
		# initialize instance
		P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, N_steps, 'distributed')
		# set up problem by callig class attrubutes
		P.set_state_space(mesh, 1)
		P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary)
		P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), None)
		# P.set_control_space(mesh, 1)
		P.set_control_constraints(u_a, u_b)
		P.set_cost(lam, y_target)
		P.set_state_equation(beta, f, g, y_0, p_end)
		P.set_exact_solution(y_exact, u_exact, p_exact)
		# set quasi-linear parameters
		P.set_non_linearity(mu, csi, csi_p, csi_pp)
		P.set_maxit_and_tol(tol, tol_pd, max_it_sqp, max_it_pd)
		# call solver
		P.sqp_solve()
		# compute errors
		P.compute_relative_errors()
		# store errors
		# Linf_same_mesh_list.append(P.Linf_error_norm)
				
		# check if errors are stagnating
		if P.L2_error_norm[0] >= old_errors[0]:
			logging.error('state error is stagnating or increasing!')
		if P.L2_error_norm[1] >= old_errors[1]:
			logging.error('control error is stagnating or increasing!')
		if P.L2_error_norm[2] >= old_errors[2]:
			logging.error('adjoint error is stagnating or increasing!')

		if  P.L2_error_norm[0] >= old_errors[0] and P.L2_error_norm[1] >= old_errors[1] and P.L2_error_norm[2] >= old_errors[2]:
			logging.error('all errors stagnating or increasing!!')
			# store errors
			L2_same_mesh_list.append(P.L2_error_norm)
			# exit the loop
			break

		# store old errors
		old_errors = P.L2_error_norm.copy()
		# store errors
		L2_same_mesh_list.append(P.L2_error_norm)
	
	L2_same_mesh_array = np.array(L2_same_mesh_list)

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP AND MESHSIZE: {total_time_end - total_time_start} s')

	print(L2_same_mesh_array)
	print(orders.compute_orders(L2_same_mesh_array))

	# diameter = 2.0

	diameter = 1

	L = np.shape(L2_same_mesh_array)[0]

	x_time = np.linspace(num_steps_list[0], num_steps_list[ L-1 ], 	100)

	# plotting
	plt.figure()
	fig, ax1 = plt.subplots(1)

	ax1.loglog( 
		num_steps_list[:L],
		L2_same_mesh_array[:,0],
		label=r'$||\, y-y_h \,||_{L^2}$',
		marker='o',
		color='r',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='r')
	ax1.loglog( 
		num_steps_list[:L],
		L2_same_mesh_array[:,1],
		label=r'$||\, u-u_h \,||_{L^2}$',
		marker='s',
		color='b',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='b')
	ax1.loglog( 
		num_steps_list[:L],
		L2_same_mesh_array[:,2],
		label=r'$||\, p-p_h \,||_{L^2}$',
		marker='D',
		color='lime',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='lime')
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

	fig.set_size_inches(6, 4 )
	plt.savefig(f'visualization_sqp/2D/L2_relative_errors_2D_vary_dt_distributed.pdf')

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

	
	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-8, 1e-8, 10, 10

	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	mesh_size_list = [2, 4, 8, 16]
	L2_same_timestep_list = []

	total_time_start = time.perf_counter()

	# initialize old error listtotal_time_start = time.perf_counter()
	old_errors = [100, 100, 100]

	for N_h in mesh_size_list:
			# build mesh
		# 2D
		new_mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), N_h, N_h)
		# initialize instance
		P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, 256, 'distributed')
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
		# call solver
		P.sqp_solve()
		# compute errors
		P.compute_relative_errors()
		
		# check if errors are stagnating
		if P.L2_error_norm[0] >= old_errors[0]:
			logging.error('state error is stagnating or increasing!')
		if P.L2_error_norm[1] >= old_errors[1]:
			logging.error('control error is stagnating or increasing!')
		if P.L2_error_norm[2] >= old_errors[2]:
			logging.error('adjoint error is stagnating or increasing!')

		if  P.L2_error_norm[0] >= old_errors[0] and P.L2_error_norm[1] >= old_errors[1] and P.L2_error_norm[2] >= old_errors[2]:
			logging.error('all errors stagnating or increasing!!')
			# store errors
			L2_same_mesh_list.append(P.L2_error_norm)
			# exit the loop
			break

		# store old errors
		old_errors = P.L2_error_norm.copy()
		# store errors
		L2_same_timestep_list.append(P.L2_error_norm)
		
	L2_same_timestep_array = np.array(L2_same_timestep_list)

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP AND MESHSIZE: {total_time_end - total_time_start} s')

	print(L2_same_timestep_array)
	print(orders.compute_orders(L2_same_timestep_array))

	# diameter = 2.0

	diameter = 1

	L = np.shape(L2_same_timestep_array)[0]

	x_mesh = np.linspace(mesh_size_list[0], mesh_size_list[ L-1 ], 100)

	# plotting
	plt.figure()
	fig, ax2 = plt.subplots(1)

	ax2.loglog(
		mesh_size_list[:L],
		L2_same_timestep_array[:,0],
		label=r'$||\, y-y_h \,||_{L^2}$',
		marker='o',
		color='r',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='r')
	ax2.loglog(
		mesh_size_list[:L],
		L2_same_timestep_array[:,1],
		label=r'$||\, u-u_h \,||_{L^2}$',
		marker='s',
		color='b',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='b')
	ax2.loglog(
		mesh_size_list[:L],
		L2_same_timestep_array[:,2],
		label=r'$||\, p-p_h \,||_{L^2}$',
		marker='D',
		color='lime',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='lime')

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
	ax2.set(xlabel=f'{diameter}/h', ylabel='error')
	ax2.legend(loc="lower left")

	fig.set_size_inches(6, 4)
	plt.savefig(f'visualization_sqp/2D/L2_relative_errors_2D_vary_size_distributed.pdf')

################################## neumann ###########################


def plot_2D_N_vary_dt(lam):
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

	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-08, 1e-8, 20, 20

	num_steps_list = [4, 8, 16, 32, 64, 128, 256]#, 1024]
	
	L2_same_mesh_list = []
	# Linf_same_mesh_list = []

	total_time_start = time.perf_counter()

	# build mesh
	# 2D
	mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), 16, 16)
	
	# initialize old error list
	old_errors = [100, 100, 100]
	
	for N_steps in num_steps_list:
		# initialize instance
		P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, N_steps, 'neumann boundary')
		# set up problem by callig class attrubutes
		P.set_state_space(mesh, 1)
		P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), None)
		P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), boundary)
		# P.set_control_space(mesh, 1)
		P.set_control_constraints(u_a, u_b)
		P.set_cost(lam, y_target)
		P.set_state_equation(beta, f, g, y_0, p_end)
		P.set_exact_solution(y_exact, u_exact, p_exact)
		# set quasi-linear parameters
		P.set_non_linearity(mu, csi, csi_p, csi_pp)
		P.set_maxit_and_tol(tol, tol_pd, max_it_sqp, max_it_pd)
		# call solver
		P.sqp_solve()
		# compute errors
		P.compute_relative_errors()

		# check if errors are stagnating
		if P.L2_error_norm[0] >= old_errors[0]:
			logging.error('state error is stagnating or increasing!')
		if P.L2_error_norm[1] >= old_errors[1]:
			logging.error('control error is stagnating or increasing!')
		if P.L2_error_norm[2] >= old_errors[2]:
			logging.error('adjoint error is stagnating or increasing!')

		if  P.L2_error_norm[0] >= old_errors[0] and P.L2_error_norm[1] >= old_errors[1] and P.L2_error_norm[2] >= old_errors[2]:
			logging.error('all errors stagnating or increasing!!')
			# store errors
			L2_same_mesh_list.append(P.L2_error_norm)
			# exit the loop
			break

		# store old errors
		old_errors = P.L2_error_norm.copy()
		# store errors
		L2_same_mesh_list.append(P.L2_error_norm)
	
	L2_same_mesh_array = np.array(L2_same_mesh_list)

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP AND MESHSIZE: {total_time_end - total_time_start} s')

	print(L2_same_mesh_array)
	print(orders.compute_orders(L2_same_mesh_array))

	diameter = 1

	L = np.shape(L2_same_mesh_array)[0]

	x_time = np.linspace(num_steps_list[0], num_steps_list[ L-1 ], 	100)

	# plotting
	plt.figure()
	fig, ax1 = plt.subplots(1)

	ax1.loglog( 
		num_steps_list[:L],
		L2_same_mesh_array[:,0],
		label=r'$||\, y-y_h \,||_{L^2}$',
		marker='o',
		color='r',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='r')
	ax1.loglog( 
		num_steps_list[:L],
		L2_same_mesh_array[:,1],
		label=r'$||\, u-u_h \,||_{L^2}$',
		marker='s',
		color='b',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='b')
	ax1.loglog( 
		num_steps_list[:L],
		L2_same_mesh_array[:,2],
		label=r'$||\, p-p_h \,||_{L^2}$',
		marker='D',
		color='lime',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='lime')
	ax1.loglog(
		x_time,
		np.power(x_time, -1) * T ,
		label=r'$\tau$',
		linestyle='--',
		linewidth=0.7,
		color='c')
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

	fig.set_size_inches(6, 4 )
	plt.savefig(f'visualization_sqp/2D/L2_relative_errors_2D_vary_dt_neumann.pdf')

	return 0


def plot_2D_N_vary_size(lam):
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


	
	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-08, 1e-8, 20, 20

	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	mesh_size_list = [2, 4, 8, 16]
	L2_same_timestep_list = []

	total_time_start = time.perf_counter()

	# initialize old error listtotal_time_start = time.perf_counter()
	old_errors = [100, 100, 100]

	for N_h in mesh_size_list:
			# build mesh
		# 2D
		new_mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), N_h, N_h)
		# initialize instance
		P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, 512, 'neumann boundary')
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
		# call solver
		P.sqp_solve()
		# compute errors
		P.compute_relative_errors()
		
		# check if errors are stagnating
		if P.L2_error_norm[0] >= old_errors[0]:
			logging.error('state error is stagnating or increasing!')
		if P.L2_error_norm[1] >= old_errors[1]:
			logging.error('control error is stagnating or increasing!')
		if P.L2_error_norm[2] >= old_errors[2]:
			logging.error('adjoint error is stagnating or increasing!')

		if  P.L2_error_norm[0] >= old_errors[0] and P.L2_error_norm[1] >= old_errors[1] and P.L2_error_norm[2] >= old_errors[2]:
			logging.error('all errors stagnating or increasing!!')
			# store errors
			L2_same_mesh_list.append(P.L2_error_norm)
			# exit the loop
			break

		# store old errors
		old_errors = P.L2_error_norm.copy()
		# store errors
		L2_same_timestep_list.append(P.L2_error_norm)
		
	L2_same_timestep_array = np.array(L2_same_timestep_list)

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME OF LOOPING OVER TIMESTEP AND MESHSIZE: {total_time_end - total_time_start} s')

	print(L2_same_timestep_array)
	print(orders.compute_orders(L2_same_timestep_array))

	# diameter = 2.0

	diameter = 1

	L = np.shape(L2_same_timestep_array)[0]

	x_mesh = np.linspace(mesh_size_list[0], mesh_size_list[ L-1 ], 100)

	# plotting
	plt.figure()
	fig, ax2 = plt.subplots(1)

	ax2.loglog(
		mesh_size_list[:L],
		L2_same_timestep_array[:,0],
		label=r'$||\, y-y_h \,||_{L^2}$',
		marker='o',
		color='r',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='r')
	ax2.loglog(
		mesh_size_list[:L],
		L2_same_timestep_array[:,1],
		label=r'$||\, u-u_h \,||_{L^2}$',
		marker='s',
		color='b',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='b')
	ax2.loglog(
		mesh_size_list[:L],
		L2_same_timestep_array[:,2],
		label=r'$||\, p-p_h \,||_{L^2}$',
		marker='D',
		color='lime',
		linewidth=0.7,
		markerfacecolor='none',
		markeredgecolor='lime')

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
	ax2.set(xlabel=f'{diameter}/h', ylabel='error')
	ax2.legend(loc="lower left")

	fig.set_size_inches(6, 4)
	plt.savefig(f'visualization_sqp/2D/L2_relative_errors_2D_vary_size_neumann.pdf')

	return 0

#############################################

if __name__ == '__main__':
	level = logging.INFO
	fmt = '[%(levelname)s] %(asctime)s - %(message)s'
	logging.basicConfig(level=level, format=fmt)

	# plot_1D_t_increments_vary_mesh_sqp(1e-2)
	# plot_1D_t_increments_vary_mesh_semismooth(1e-2)

	# plot_1D_vary_dt(1e-2)
	plot_1D_vary_size(1e-2)

	# plot_2D_vary_dt(1e-2)
	# plot_2D_vary_size(1e-2)

	# plot_2D_N_vary_dt(1e-2)
	# plot_2D_N_vary_size(1e-2)

	logging.info('FINISHED')



