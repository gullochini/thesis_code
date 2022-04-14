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

import sqp_quasi_linear, semi_smooth_quasi_linear, orders

######################### 1D ##########################

def example_1D(lam, mesh_size, num_steps):
	## python functions 
	def csi(w):
		return Constant(1.0) + 1/(1 + math.e**(-w))
	def csi_p(w):
		return math.e**(-w)/(1 + math.e**(-w))**2
	def csi_pp(w):
		return (math.e**(-2*w) - math.e**(-w))/(1 + math.e**(-w))**3

	# independent of choice of constraints
	T = 1.0 
	beta = Constant(1.0)
	y_0 = Constant(0.0)
	p_end = Constant(0.0) 
	g = Constant(0.0)
	f = Constant(0.0)
	y_exact, u_exact, p_exact = Constant(0.0), Constant(0.0), Constant(0.0) 

	y_target= Expression(
		'''
			t * ( x[0] > .75 ? 1 : 0 )
		''',
		t=0, degree=5, lam=lam)

	u_a	= Constant( 0.0 )
	# space and time dependant constraint
	u_b = Expression('  (4 + .5 * x[0]) * exp(t - 1)', t=0, degree=2)

	mu, tol_nm, tol_sm, max_it_nm, max_it_sm = 1, 1e-10, 1e-4, 20, 20

	tol, tol_pd, max_it_sqp, max_it_pd = 1e-8, 1e-8, 10, 10

	# dirichlet boundary is boundary
	def boundary(x, on_boundary):
		return on_boundary

	def boundary_D(x, on_boundary):
		if on_boundary:
			if near(x[0], 0, 1e-8):
				return True
			else:
				return False
		else:
			return False

	def boundary_N(x, on_boundary):
		if on_boundary:
			if near(x[0], 1, 1e-8):
				return True
			else:
				return False
		else:
			return False

	total_time_start = time.perf_counter()

	# build mesh
	# 1D
	new_mesh = IntervalMesh(mesh_size, 0.0, 1.0)

	########################################## SEMISMOOTH ########################################

	# # initialize instance
	# P = semi_smooth_quasi_linear.Quasi_Linear_Problem_Box(T, num_steps, 'distributed')
	# # set up problem by callig class attrubutes
	# P.set_state_space(new_mesh, 1)
	# P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary_D)
	# P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), boundary_N)
	# # P.set_control_space(new_mesh, 1)
	# P.set_control_constraints(u_a, u_b)
	# P.set_cost(lam, y_target)
	# P.set_state_equation(beta, f, g, y_0, p_end)
	# P.set_exact_solution(y_exact, u_exact, p_exact)
	# # set quasi-linear parameters
	# P.set_non_linearity(mu, csi, csi_p, csi_pp)
	# P.set_maxit_and_tol(tol_nm, tol_sm, max_it_nm, max_it_sm)
	# # call solver
	# P.semi_smooth_solve()
	
	# P.visualize_1D(0, 1, 128, 'visualization_semi_smooth_quasi_linear/other')
	# # P.visualize_paraview('visualization_semi_smooth_quasi_linear/other')
	# print(P.evaluate_cost_functional(P.y, P.u))

	# P.plot_residuals('visualization_semi_smooth_quasi_linear/other')

	# print('inf_errors\n',P.error_sequence_list)
	# print('increments\n',P.incr_list)
	# print('orders\n', orders.compute_EOC_orders(P.incr_list))

	# total_time_end = time.perf_counter()

	# logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')


	################################# SQP ################################
	# initialize instance
	P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, num_steps, 'distributed')
	# set up problem by callig class attrubutes
	P.set_state_space(new_mesh, 1)
	P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary_D)
	P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), boundary_N)
	# P.set_control_space(new_mesh, 1)
	P.set_control_constraints(u_a, u_b)
	P.set_cost(lam, y_target)
	P.set_state_equation(beta, f, g, y_0, p_end)
	# P.set_exact_solution(y_exact, u_exact, p_exact)
	# set quasi-linear parameters
	P.set_non_linearity(mu, csi, csi_p, csi_pp)

	P.set_maxit_and_tol(tol, tol_pd, max_it_sqp, max_it_pd)
	# call solver
	P.sqp_solve()
	
	P.visualize_1D(0, 1, 128, 'visualization_other')
	P.visualize_paraview('visualization_other/paraview/1D/distributed')
	print(P.evaluate_cost_functional(P.y, P.u))

	J_y_u, J_y, J_u = P.evaluate_cost_functional(P.y, P.u)
	print(f'optimal cost: {J_y_u}, contributions J_y: {J_y}, J_u: {J_u} ')

	print('inf_errors\n',P.error_sequence_list)
	print('increments\n',P.incr_list)
	print('orders\n', orders.compute_EOC_orders(P.incr_list))
	logging.info(fr'computed order of convergence is $q =${math.log(P.incr_list[-1]/P.incr_list[-2]) / math.log(P.incr_list[-2]/P.incr_list[-3])}')

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0

############################## 2D unit circle ############################

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
			if x[1] <= 0 or x[0] <= 0:
				return True
			else:
				return False
		else:
			return False

	def boundary_N(x, on_boundary):
		if on_boundary:
			if x[1] > 0 and x[0] > 0:
				return True
			else:
				return False
		else:
			return False
	
	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-08, 1e-10, 20, 20

	T 		= 1.0
	beta 	= Constant( 1.0 )

	y_exact, u_exact, p_exact = Constant(0.0), Constant(0.0), Constant(0.0) 

	y_0 = Constant(0.0)
	f = Constant(0.0)
	g = Constant(0.0)

	p_end 	= Constant(0.0) 

	y_target= Expression(
		'''	
			10 * ( x[0] > 0.5 ? 1 : 0 )*( x[1] > 0.5 ? 1 : 0 )
		''',
		t=0,
		degree=5,
		lam=1e-2)
	
	# space and time dependant constraints
	u_b = Expression(
		't*(1 + 0.5*x[0])*(1 + 0.5*x[1])',
		t=0,
		degree=2)
	u_a = Constant(0.0)
	

	total_time_start = time.perf_counter()

	# build mesh
	# 2D
	center = Point(0.0, 0.0)
	# new_mesh = generate_mesh(Circle(center, 1, mesh_size**2), mesh_size)
	new_mesh = generate_mesh(Circle(center, 1), mesh_size)
	# initialize instance
	P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, num_steps, 'distributed')
	# set up problem by callig class attrubutes
	P.set_state_space(new_mesh, 1)
	print(P.mesh_size)
	P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary_D)
	P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), boundary_N)
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
	# visualization
	P.visualize_paraview('visualization_other/paraview/2D/distributed')

	J_y_u, J_y, J_u = P.evaluate_cost_functional(P.y, P.u)
	print(f'optimal cost: {J_y_u}, contributions J_y: {J_y}, J_u: {J_u} ')

	if P.compute_proj_residuals_flag:
		P.plot_residuals('visualization_other/2D/distributed')

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0

def example_2D_t(lam, mesh_size, num_steps):

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
			if x[1] <= 0 or x[0] <= 0:
				return True
			else:
				return False
		else:
			return False

	def boundary_N(x, on_boundary):
		if on_boundary:
			if x[1] > 0 and x[0] > 0:
				return True
			else:
				return False
		else:
			return False
	
	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-08, 1e-10, 20, 20

	T 		= 1.0
	beta 	= Expression(
		'(pow(x[0], 2) + pow(x[1], 2) <= pow(2,-1) ? 1 : 0)*(x[0] >= 0 ? 1 : 0)*(x[1] >= 0 ? 1 : 0)',
		degree=5 )

	y_exact, u_exact, p_exact = Constant(0.0), Constant(0.0), Constant(0.0) 

	y_0 = Constant(0.0)
	f = Constant(0.0)
	g = Constant(0.0)

	p_end = Constant(0.0)

	y_target= Expression(
		'''	
			( x[0] > 0 ? 1 : 0 )*( x[1] > 0 ? 1 : 0 )*(t < 0.5 ? 1 : 0)
			-( x[0] > 0 ? 1 : 0 )*( x[1] > 0 ? 1 : 0 )*(t >= 0.5 ? 1 : 0 )
		''',
		t=0,
		degree=5,
		lam=1e-2)
	
	# time dependant constraints
	u_b = Expression(
		'5*(1 + t)',
		t=0,
		degree=5)
	u_a = Expression(
		'-5*(1 + (1-t))',
		t=0,
		degree=5)

	total_time_start = time.perf_counter()

	# build mesh
	# 2D
	center = Point(0.0, 0.0)
	new_mesh = generate_mesh(Circle(center, 1), mesh_size)

	# build mesh
	# 2D
	# initialize instance
	P = sqp_quasi_linear.Quasi_Linear_Problem_Box(T, num_steps, 'time')
	# set up problem by callig class attrubutes
	P.set_state_space(new_mesh, 1)
	print(P.mesh_size)
	P.set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary_D)
	P.set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), boundary_N)
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
	# visualization
	P.visualize_paraview('visualization_other/paraview/2D/time')

	P.visualize_purely_time_dep('visualization_other/2D/time')

	J_y_u, J_y, J_u = P.evaluate_cost_functional(P.y, P.u)
	print(f'optimal cost: {J_y_u}, contributions J_y: {J_y}, J_u: {J_u} ')

	if P.compute_proj_residuals_flag:
		P.plot_residuals('visualization_other/2D/time')

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0

def example_2D_t_vary_lam(mesh_size, num_steps):

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
			if x[1] <= 0 or x[0] <= 0:
				return True
			else:
				return False
		else:
			return False

	def boundary_N(x, on_boundary):
		if on_boundary:
			if x[1] > 0 and x[0] > 0:
				return True
			else:
				return False
		else:
			return False

	mu, tol, tol_pd, max_it_sqp, max_it_pd = 1, 1e-08, 1e-10, 20, 20

	T 		= 1.0
	beta 	= Expression(
		'(pow(x[0], 2) + pow(x[1], 2) <= pow(2,-1) ? 1 : 0)*(x[0] >= 0 ? 1 : 0)*(x[1] >= 0 ? 1 : 0)',
		degree=5 )

	y_exact, u_exact, p_exact = Constant(0.0), Constant(0.0), Constant(0.0) 

	y_0 = Constant(0.0)
	f = Constant(0.0)
	g = Constant(0.0)

	p_end = Constant(0.0)

	y_target= Expression(
		'''	
			.5*( x[0] > 0 ? 1 : 0 )*( x[1] > 0 ? 1 : 0 )*(t < 0.5 ? 1 : 0)
			-.5*( x[0] > 0 ? 1 : 0 )*( x[1] > 0 ? 1 : 0 )*(t >= 0.5 ? 1 : 0 )
		''',
		t=0,
		degree=5,
		lam=1e-2)
	
	u_b = Expression(
		'1 + t',
		t=0,
		degree=5)
	u_a = Expression(
		'-2 + t',
		t=0,
		degree=5)

	total_time_start = time.perf_counter()

	# build mesh
	# 2D
	center = Point(0.0, 0.0)
	# new_mesh = generate_mesh(Circle(center, 1, mesh_size**2), mesh_size)
	new_mesh = generate_mesh(Circle(center, 1), mesh_size)

	# build mesh
	# 2D
	instances = []
	# plotting
	plt.figure()
	fig, ax = plt.subplots(1, constrained_layout=True)

	for lam, color in zip([.05, .01, .002, .0004, .00008], ['orange','darkorange', 'red', 'magenta', 'blue']):
		# initialize instance
		instances.append(sqp_quasi_linear.Quasi_Linear_Problem_Box(T, num_steps, 'time'))
		# set up problem by callig class attrubutes
		instances[-1].set_state_space(new_mesh, 1)
		print(instances[-1].mesh_size)
		instances[-1].set_dirichlet_boundary_conditions(Constant(0.0), Constant(0.0), boundary_D)
		instances[-1].set_neumann_boundary_conditions(Constant(0.0), Constant(0.0), boundary_N)
		# P.set_control_space(new_mesh, 1)
		instances[-1].set_control_constraints(u_a, u_b)
		instances[-1].set_cost(lam, y_target)
		instances[-1].set_state_equation(beta, f, g, y_0, p_end)
		instances[-1].set_exact_solution(y_exact, u_exact, p_exact)
		# set quasi-linear parameters
		instances[-1].set_non_linearity(mu, csi, csi_p, csi_pp)
		instances[-1].set_maxit_and_tol(tol, tol_pd, max_it_sqp, max_it_pd)
		# P.compute_proj_residuals_flag = True
		# call solver
		instances[-1].sqp_solve()
		
		# visualization
		control_list = []
		time_list = []
		u_a_list = []
		u_b_list = []
		t=0

		for i in range(instances[-1].num_steps + 1):
			# extrapolate constant values
			control_list.append(interpolate(instances[-1].u[i], instances[-1].V).vector().get_local()[0])

			if color == 'orange': 
				# extrapolate constant values
				instances[-1].u_b.t, instances[-1].u_a.t = t, t
				u_a_list.append(interpolate(instances[-1].u_a, instances[-1].V).vector().get_local()[2])
				u_b_list.append(interpolate(instances[-1].u_b, instances[-1].V).vector().get_local()[2])

			time_list.append(t)
			t += instances[-1].dt

		if color == 'orange': 
			# # constraints
			plot_u_a = ax.plot(
				time_list,
				u_a_list,
				label=fr'$u_a(t)$',
				# marker='D',
				linewidth=0.5,
				linestyle='--',
				color='0.5')
			plot_u_b = ax.plot(
				time_list,
				u_b_list,
				label=fr'$u_b(t)$',
				# marker='D',
				linewidth=0.5,
				linestyle='-.',
				color='0.5')

		# control
		plot_control = ax.plot(
			time_list,
			control_list,
			label=fr'$u(t), \lambda=${lam}',
			# marker='D',
			linewidth=0.7,
			color=color)

	# ax.set_title('purely time dep. control')
	ax.set(ylabel='control', xlabel='t')
	ax.set(
		#2ylim=(min(control_list)-0.2, max(control_list)+0.2),
		xlim=(0, instances[-1].T))

	ax.legend(loc="lower left")

	plt.savefig('visualization_other/2D/time/control_vary_lam.pdf')

	total_time_end = time.perf_counter()

	logging.info(f'TOTAL TIME: {total_time_end - total_time_start} s')

	return 0


################################################################
		
if __name__ == '__main__':
	level = logging.INFO
	fmt = '[%(levelname)s] %(asctime)s - %(message)s'
	logging.basicConfig(level=level, format=fmt)

	# example_1D(1e-2, 40, 1600)
	# example_2D(1e-2, 10, 100)
	# example_2D_t(1e-2, 10, 100)
	# example_2D_t_vary_lam(12, 144)

	logging.info('FINISHED')
