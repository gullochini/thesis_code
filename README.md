# On the numerical solution of optimal control problems governed by quasilinear parabolic PDEs
## Master thesis - Guglielmo Chini

Numerical solver of optimal control problems governed by quasilinear parabolic partial differential equations.

### documentation
the command
```bash
pydoc filename
```
shows the documentation for the file ```filename.py```

### run an example
choose and uncomment the example in the ```__main__``` section of one of the following ```.py``` files:

```
quasi_linear_solver_t.py
sqp_quasi_linear.py
semi_smooth_quasi_linear.py
other_examples.py
error_estimates.py
```

In order for them to run, an istallation of FEniCS on Anaconda is necessary. For instance, if FEniCS is located in the venv fenics-env

```bash
conda activate fenics-env
python3 filename.py
```

The ```.xdmf``` files will be written in the following directories: visualization/paraview, visualization_sqp/paraview, visualization_sqp/other/paraview, visualization_semismooth/paraview, visualization_other/paraview.

They can be loaded directly in the paraview app to visualize the computed solutions.
