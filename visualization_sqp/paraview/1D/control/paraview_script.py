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
renderView1.CenterOfRotation = [0.5, 0.0, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [0.5, -1.9318516525781368, 0.0]
renderView1.CameraFocalPoint = [0.5, 0.0, 0.0]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.605
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
solution_controlxdmf = Xdmf3ReaderS(registrationName='solution_control.xdmf', FileName=['/home/gullo/repo/quasi-linear-PDE-optimal-control/visualization_sqp/paraview/1D/control/solution_control.xdmf'])
solution_controlxdmf.PointArrays = ['f_107929']

# create a new 'Warp By Scalar'
warpByScalar1 = WarpByScalar(registrationName='WarpByScalar1', Input=solution_controlxdmf)
warpByScalar1.Scalars = ['POINTS', 'f_107929']

# create a new 'Xdmf3ReaderS'
solution_controlxdmf_1 = Xdmf3ReaderS(registrationName='solution_control.xdmf', FileName=['/home/gullo/repo/quasi-linear-PDE-optimal-control/visualization_sqp/paraview/1D/control/solution_control.xdmf'])
solution_controlxdmf_1.PointArrays = ['f_107929']

# create a new 'Warp By Scalar'
warpByScalar2 = WarpByScalar(registrationName='WarpByScalar2', Input=solution_controlxdmf_1)
warpByScalar2.Scalars = ['POINTS', 'f_107929']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from solution_controlxdmf
solution_controlxdmfDisplay = Show(solution_controlxdmf, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'f_107929'
f_107929LUT = GetColorTransferFunction('f_107929')
f_107929LUT.AutomaticRescaleRangeMode = 'Never'
f_107929LUT.RGBPoints = [-0.45137500762939453, 0.231373, 0.298039, 0.752941, -0.004105821251869202, 0.865003, 0.865003, 0.865003, 0.44316336512565613, 0.705882, 0.0156863, 0.14902]
f_107929LUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'f_107929'
f_107929PWF = GetOpacityTransferFunction('f_107929')
f_107929PWF.Points = [-0.45137500762939453, 0.0, 0.5, 0.0, 0.44316336512565613, 1.0, 0.5, 0.0]
f_107929PWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
solution_controlxdmfDisplay.Representation = 'Points'
solution_controlxdmfDisplay.ColorArrayName = ['POINTS', 'f_107929']
solution_controlxdmfDisplay.LookupTable = f_107929LUT
solution_controlxdmfDisplay.SelectTCoordArray = 'None'
solution_controlxdmfDisplay.SelectNormalArray = 'None'
solution_controlxdmfDisplay.SelectTangentArray = 'None'
solution_controlxdmfDisplay.OSPRayScaleArray = 'f_107929'
solution_controlxdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
solution_controlxdmfDisplay.SelectOrientationVectors = 'None'
solution_controlxdmfDisplay.ScaleFactor = 0.1
solution_controlxdmfDisplay.SelectScaleArray = 'f_107929'
solution_controlxdmfDisplay.GlyphType = 'Arrow'
solution_controlxdmfDisplay.GlyphTableIndexArray = 'f_107929'
solution_controlxdmfDisplay.GaussianRadius = 0.005
solution_controlxdmfDisplay.SetScaleArray = ['POINTS', 'f_107929']
solution_controlxdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
solution_controlxdmfDisplay.OpacityArray = ['POINTS', 'f_107929']
solution_controlxdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
solution_controlxdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
solution_controlxdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
solution_controlxdmfDisplay.ScalarOpacityFunction = f_107929PWF
solution_controlxdmfDisplay.ScalarOpacityUnitDistance = 0.3684031498640387
solution_controlxdmfDisplay.OpacityArrayName = ['POINTS', 'f_107929']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
solution_controlxdmfDisplay.ScaleTransferFunction.Points = [-1.2763877278192988e-16, 0.0, 0.5, 0.0, 0.003375000087544322, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
solution_controlxdmfDisplay.OpacityTransferFunction.Points = [-1.2763877278192988e-16, 0.0, 0.5, 0.0, 0.003375000087544322, 1.0, 0.5, 0.0]

# show data from warpByScalar1
warpByScalar1Display = Show(warpByScalar1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
warpByScalar1Display.Representation = 'Surface'
warpByScalar1Display.ColorArrayName = ['POINTS', 'f_107929']
warpByScalar1Display.LookupTable = f_107929LUT
warpByScalar1Display.SelectTCoordArray = 'None'
warpByScalar1Display.SelectNormalArray = 'None'
warpByScalar1Display.SelectTangentArray = 'None'
warpByScalar1Display.OSPRayScaleArray = 'f_107929'
warpByScalar1Display.OSPRayScaleFunction = 'PiecewiseFunction'
warpByScalar1Display.SelectOrientationVectors = 'None'
warpByScalar1Display.ScaleFactor = 0.1
warpByScalar1Display.SelectScaleArray = 'f_107929'
warpByScalar1Display.GlyphType = 'Arrow'
warpByScalar1Display.GlyphTableIndexArray = 'f_107929'
warpByScalar1Display.GaussianRadius = 0.005
warpByScalar1Display.SetScaleArray = ['POINTS', 'f_107929']
warpByScalar1Display.ScaleTransferFunction = 'PiecewiseFunction'
warpByScalar1Display.OpacityArray = ['POINTS', 'f_107929']
warpByScalar1Display.OpacityTransferFunction = 'PiecewiseFunction'
warpByScalar1Display.DataAxesGrid = 'GridAxesRepresentation'
warpByScalar1Display.PolarAxes = 'PolarAxesRepresentation'
warpByScalar1Display.ScalarOpacityFunction = f_107929PWF
warpByScalar1Display.ScalarOpacityUnitDistance = 0.40168191466364916
warpByScalar1Display.OpacityArrayName = ['POINTS', 'f_107929']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
warpByScalar1Display.ScaleTransferFunction.Points = [-2.642320222462676e-20, 0.0, 0.5, 0.0, 0.4345398545265198, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
warpByScalar1Display.OpacityTransferFunction.Points = [-2.642320222462676e-20, 0.0, 0.5, 0.0, 0.4345398545265198, 1.0, 0.5, 0.0]

# show data from warpByScalar2
warpByScalar2Display = Show(warpByScalar2, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
warpByScalar2Display.Representation = 'Point Gaussian'
warpByScalar2Display.ColorArrayName = ['POINTS', 'f_107929']
warpByScalar2Display.LookupTable = f_107929LUT
warpByScalar2Display.SelectTCoordArray = 'None'
warpByScalar2Display.SelectNormalArray = 'None'
warpByScalar2Display.SelectTangentArray = 'None'
warpByScalar2Display.OSPRayScaleArray = 'f_107929'
warpByScalar2Display.OSPRayScaleFunction = 'PiecewiseFunction'
warpByScalar2Display.SelectOrientationVectors = 'None'
warpByScalar2Display.ScaleFactor = 0.1
warpByScalar2Display.SelectScaleArray = 'f_107929'
warpByScalar2Display.GlyphType = 'Arrow'
warpByScalar2Display.GlyphTableIndexArray = 'f_107929'
warpByScalar2Display.GaussianRadius = 0.005
warpByScalar2Display.SetScaleArray = ['POINTS', 'f_107929']
warpByScalar2Display.ScaleTransferFunction = 'PiecewiseFunction'
warpByScalar2Display.OpacityArray = ['POINTS', 'f_107929']
warpByScalar2Display.OpacityTransferFunction = 'PiecewiseFunction'
warpByScalar2Display.DataAxesGrid = 'GridAxesRepresentation'
warpByScalar2Display.PolarAxes = 'PolarAxesRepresentation'
warpByScalar2Display.ScalarOpacityFunction = f_107929PWF
warpByScalar2Display.ScalarOpacityUnitDistance = 0.38845254026327786
warpByScalar2Display.OpacityArrayName = ['POINTS', 'f_107929']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
warpByScalar2Display.ScaleTransferFunction.Points = [-0.3343749940395355, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
warpByScalar2Display.OpacityTransferFunction.Points = [-0.3343749940395355, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for f_107929LUT in view renderView1
f_107929LUTColorBar = GetScalarBar(f_107929LUT, renderView1)
f_107929LUTColorBar.Orientation = 'Horizontal'
f_107929LUTColorBar.WindowLocation = 'AnyLocation'
f_107929LUTColorBar.Position = [0.6284210526315785, 0.9033333333333333]
f_107929LUTColorBar.Title = 'f_107929'
f_107929LUTColorBar.ComponentTitle = ''
f_107929LUTColorBar.ScalarBarLength = 0.33000000000000074

# set color bar visibility
f_107929LUTColorBar.Visibility = 1

# show color legend
solution_controlxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
warpByScalar1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
warpByScalar2Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(solution_controlxdmf)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')