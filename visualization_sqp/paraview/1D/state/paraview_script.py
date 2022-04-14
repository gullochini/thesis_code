# state file generated using paraview version 5.9.1

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [950, 795]
renderView1.InteractionMode = '2D'
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.5, 0.0, 0.49887123703956604]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [0.5043000451935676, -2.72896871223842, 0.07531678547313493]
renderView1.CameraFocalPoint = [0.5043000451935676, 0.0, 0.07531678547313493]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 1.251269613370654
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(950, 795)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Xdmf3ReaderS'
solution_statexdmf = Xdmf3ReaderS(registrationName='solution_state.xdmf', FileName=['/home/gullo/repo/quasi-linear-PDE-optimal-control/visualization_sqp/paraview/1D/state/solution_state.xdmf'])
solution_statexdmf.PointArrays = ['f_107926']

# create a new 'Warp By Scalar'
warpByScalar1 = WarpByScalar(registrationName='WarpByScalar1', Input=solution_statexdmf)
warpByScalar1.Scalars = ['POINTS', 'f_107926']

# create a new 'Xdmf3ReaderS'
solution_statexdmf_1 = Xdmf3ReaderS(registrationName='solution_state.xdmf', FileName=['/home/gullo/repo/quasi-linear-PDE-optimal-control/visualization_sqp/paraview/1D/state/solution_state.xdmf'])
solution_statexdmf_1.PointArrays = ['f_107926']

# create a new 'Warp By Scalar'
warpByScalar2 = WarpByScalar(registrationName='WarpByScalar2', Input=solution_statexdmf_1)
warpByScalar2.Scalars = ['POINTS', 'f_107926']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from solution_statexdmf
solution_statexdmfDisplay = Show(solution_statexdmf, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'f_107926'
f_107926LUT = GetColorTransferFunction('f_107926')
f_107926LUT.AutomaticRescaleRangeMode = 'Never'
f_107926LUT.RGBPoints = [-0.9985164999961853, 0.231373, 0.298039, 0.752941, 0.0006209015846252441, 0.865003, 0.865003, 0.865003, 0.9997583031654358, 0.705882, 0.0156863, 0.14902]
f_107926LUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'f_107926'
f_107926PWF = GetOpacityTransferFunction('f_107926')
f_107926PWF.Points = [-0.9985164999961853, 0.0, 0.5, 0.0, 0.9997583031654358, 1.0, 0.5, 0.0]
f_107926PWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
solution_statexdmfDisplay.Representation = 'Point Gaussian'
solution_statexdmfDisplay.ColorArrayName = ['POINTS', 'f_107926']
solution_statexdmfDisplay.LookupTable = f_107926LUT
solution_statexdmfDisplay.SelectTCoordArray = 'None'
solution_statexdmfDisplay.SelectNormalArray = 'None'
solution_statexdmfDisplay.SelectTangentArray = 'None'
solution_statexdmfDisplay.OSPRayScaleArray = 'f_107926'
solution_statexdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
solution_statexdmfDisplay.SelectOrientationVectors = 'None'
solution_statexdmfDisplay.ScaleFactor = 0.1
solution_statexdmfDisplay.SelectScaleArray = 'f_107926'
solution_statexdmfDisplay.GlyphType = 'Arrow'
solution_statexdmfDisplay.GlyphTableIndexArray = 'f_107926'
solution_statexdmfDisplay.GaussianRadius = 0.005
solution_statexdmfDisplay.SetScaleArray = ['POINTS', 'f_107926']
solution_statexdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
solution_statexdmfDisplay.OpacityArray = ['POINTS', 'f_107926']
solution_statexdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
solution_statexdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
solution_statexdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
solution_statexdmfDisplay.ScalarOpacityFunction = f_107926PWF
solution_statexdmfDisplay.ScalarOpacityUnitDistance = 0.3684031498640387
solution_statexdmfDisplay.OpacityArrayName = ['POINTS', 'f_107926']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
solution_statexdmfDisplay.ScaleTransferFunction.Points = [-8.151511779460038e-23, 0.0, 0.5, 0.0, 0.9977424740791321, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
solution_statexdmfDisplay.OpacityTransferFunction.Points = [-8.151511779460038e-23, 0.0, 0.5, 0.0, 0.9977424740791321, 1.0, 0.5, 0.0]

# show data from warpByScalar1
warpByScalar1Display = Show(warpByScalar1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
warpByScalar1Display.Representation = 'Point Gaussian'
warpByScalar1Display.ColorArrayName = ['POINTS', 'f_107926']
warpByScalar1Display.LookupTable = f_107926LUT
warpByScalar1Display.SelectTCoordArray = 'None'
warpByScalar1Display.SelectNormalArray = 'None'
warpByScalar1Display.SelectTangentArray = 'None'
warpByScalar1Display.OSPRayScaleArray = 'f_107926'
warpByScalar1Display.OSPRayScaleFunction = 'PiecewiseFunction'
warpByScalar1Display.SelectOrientationVectors = 'None'
warpByScalar1Display.ScaleFactor = 0.1
warpByScalar1Display.SelectScaleArray = 'f_107926'
warpByScalar1Display.GlyphType = 'Arrow'
warpByScalar1Display.GlyphTableIndexArray = 'f_107926'
warpByScalar1Display.GaussianRadius = 0.005
warpByScalar1Display.SetScaleArray = ['POINTS', 'f_107926']
warpByScalar1Display.ScaleTransferFunction = 'PiecewiseFunction'
warpByScalar1Display.OpacityArray = ['POINTS', 'f_107926']
warpByScalar1Display.OpacityTransferFunction = 'PiecewiseFunction'
warpByScalar1Display.DataAxesGrid = 'GridAxesRepresentation'
warpByScalar1Display.PolarAxes = 'PolarAxesRepresentation'
warpByScalar1Display.ScalarOpacityFunction = f_107926PWF
warpByScalar1Display.ScalarOpacityUnitDistance = 0.5204129769112176
warpByScalar1Display.OpacityArrayName = ['POINTS', 'f_107926']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
warpByScalar1Display.ScaleTransferFunction.Points = [-8.151511779460038e-23, 0.0, 0.5, 0.0, 0.9977424740791321, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
warpByScalar1Display.OpacityTransferFunction.Points = [-8.151511779460038e-23, 0.0, 0.5, 0.0, 0.9977424740791321, 1.0, 0.5, 0.0]

# show data from solution_statexdmf_1
solution_statexdmf_1Display = Show(solution_statexdmf_1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
solution_statexdmf_1Display.Representation = 'Surface'
solution_statexdmf_1Display.ColorArrayName = ['POINTS', 'f_107926']
solution_statexdmf_1Display.LookupTable = f_107926LUT
solution_statexdmf_1Display.SelectTCoordArray = 'None'
solution_statexdmf_1Display.SelectNormalArray = 'None'
solution_statexdmf_1Display.SelectTangentArray = 'None'
solution_statexdmf_1Display.OSPRayScaleArray = 'f_107926'
solution_statexdmf_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
solution_statexdmf_1Display.SelectOrientationVectors = 'None'
solution_statexdmf_1Display.ScaleFactor = 0.1
solution_statexdmf_1Display.SelectScaleArray = 'f_107926'
solution_statexdmf_1Display.GlyphType = 'Arrow'
solution_statexdmf_1Display.GlyphTableIndexArray = 'f_107926'
solution_statexdmf_1Display.GaussianRadius = 0.005
solution_statexdmf_1Display.SetScaleArray = ['POINTS', 'f_107926']
solution_statexdmf_1Display.ScaleTransferFunction = 'PiecewiseFunction'
solution_statexdmf_1Display.OpacityArray = ['POINTS', 'f_107926']
solution_statexdmf_1Display.OpacityTransferFunction = 'PiecewiseFunction'
solution_statexdmf_1Display.DataAxesGrid = 'GridAxesRepresentation'
solution_statexdmf_1Display.PolarAxes = 'PolarAxesRepresentation'
solution_statexdmf_1Display.ScalarOpacityFunction = f_107926PWF
solution_statexdmf_1Display.ScalarOpacityUnitDistance = 0.3684031498640387
solution_statexdmf_1Display.OpacityArrayName = ['POINTS', 'f_107926']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
solution_statexdmf_1Display.ScaleTransferFunction.Points = [-8.151511779460038e-23, 0.0, 0.5, 0.0, 0.9977424740791321, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
solution_statexdmf_1Display.OpacityTransferFunction.Points = [-8.151511779460038e-23, 0.0, 0.5, 0.0, 0.9977424740791321, 1.0, 0.5, 0.0]

# show data from warpByScalar2
warpByScalar2Display = Show(warpByScalar2, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
warpByScalar2Display.Representation = 'Surface'
warpByScalar2Display.ColorArrayName = ['POINTS', 'f_107926']
warpByScalar2Display.LookupTable = f_107926LUT
warpByScalar2Display.SelectTCoordArray = 'None'
warpByScalar2Display.SelectNormalArray = 'None'
warpByScalar2Display.SelectTangentArray = 'None'
warpByScalar2Display.OSPRayScaleArray = 'f_107926'
warpByScalar2Display.OSPRayScaleFunction = 'PiecewiseFunction'
warpByScalar2Display.SelectOrientationVectors = 'None'
warpByScalar2Display.ScaleFactor = 0.1
warpByScalar2Display.SelectScaleArray = 'f_107926'
warpByScalar2Display.GlyphType = 'Arrow'
warpByScalar2Display.GlyphTableIndexArray = 'f_107926'
warpByScalar2Display.GaussianRadius = 0.005
warpByScalar2Display.SetScaleArray = ['POINTS', 'f_107926']
warpByScalar2Display.ScaleTransferFunction = 'PiecewiseFunction'
warpByScalar2Display.OpacityArray = ['POINTS', 'f_107926']
warpByScalar2Display.OpacityTransferFunction = 'PiecewiseFunction'
warpByScalar2Display.DataAxesGrid = 'GridAxesRepresentation'
warpByScalar2Display.PolarAxes = 'PolarAxesRepresentation'
warpByScalar2Display.ScalarOpacityFunction = f_107926PWF
warpByScalar2Display.ScalarOpacityUnitDistance = 0.5204129769112176
warpByScalar2Display.OpacityArrayName = ['POINTS', 'f_107926']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
warpByScalar2Display.ScaleTransferFunction.Points = [-8.151511779460038e-23, 0.0, 0.5, 0.0, 0.9977424740791321, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
warpByScalar2Display.OpacityTransferFunction.Points = [-8.151511779460038e-23, 0.0, 0.5, 0.0, 0.9977424740791321, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for f_107926LUT in view renderView1
f_107926LUTColorBar = GetScalarBar(f_107926LUT, renderView1)
f_107926LUTColorBar.Orientation = 'Horizontal'
f_107926LUTColorBar.WindowLocation = 'AnyLocation'
f_107926LUTColorBar.Position = [0.6189473684210528, 0.9020754716981132]
f_107926LUTColorBar.Title = 'f_107926'
f_107926LUTColorBar.ComponentTitle = ''
f_107926LUTColorBar.ScalarBarLength = 0.3299999999999993

# set color bar visibility
f_107926LUTColorBar.Visibility = 1

# show color legend
solution_statexdmfDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
warpByScalar1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
solution_statexdmf_1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
warpByScalar2Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(None)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')