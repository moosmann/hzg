# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


import os
import numpy as np
import odl
import io
import reader
import scipy.ndimage

# --- DATA --- #

# Projection folder and file names
# par_folder = '/asap3/petra3/gpfs/p05/2016/data'
par_folder = '/home/moosmanj'
# scan = '11002518/processed/mouse_joint_1/mouse_joint_1_d/sino'
scan_name = 'data'
# '11001464/processed/pnl_16_petrosia_a/sino'
# filename = 'mouse_joint_1_d00001.sin'
filename = 'test.sin'
# 'pnl_16_petrosia_a00001.sin'
path = os.path.join(par_folder, scan_name, filename)

print(path)

# Read projections
sino = reader.ReadDat(path)
print(sino.fullpath)
sino()
print('dim:', sino.dimensions, 'size: ', sino.dimsize, 'dtype: ', sino.dtype)

print('sino: shape = {}'.format(sino.data.shape))
dimHor, num_proj = sino.data.shape
print('dimHor: ', dimHor, '\nnum_proj: ', num_proj)
pix_size_eff = 0.0024597921
sino_width = pix_size_eff * dimHor
print('sino width: ', sino_width)
half_width = sino_width / 2
print('half_width: ', half_width)


# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_corner=[-half_width, -half_width],
    max_corner=[half_width, half_width],
    nsamples=[dimHor, dimHor],
    dtype=sino.dtype)


# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced
angle_partition = odl.uniform_partition(0, 1 * np.pi, num_proj)
# Detector: uniformly sampled
detector_partition = odl.uniform_partition(-half_width, half_width, dimHor)
print(detector_partition)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# ray transform aka forward projection. We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Projection data
# proj_data = ray_trafo.range.element(np.rot90(sino.data))
#print('proj_data: ', proj_data.space)
# proj_data.show(title='Sinogram')


proj_data = ray_trafo.range.element(
scipy.ndimage.median_filter(np.rot90(sino.data), [7, 1])
)
proj_data.show(title='Sinogram filtered', show=True)


# Optionally pass partial to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

# Choose a starting point
# x = ray_trafo.domain.zero()
#x = reco_space.zero()

# CG Run the algorithm
#odl.solvers.conjugate_gradient_normal(
#    ray_trafo, x, proj_data, niter=10, callback=callback)


# --- Set up the inverse problem --- #


# Initialize gradient operator
gradient = odl.Gradient(reco_space, method='forward')

# Column vector of two operators
op = odl.BroadcastOperator(ray_trafo, gradient)

# Create the proximal operator for unconstrained primal variable
proximal_primal = odl.solvers.proximal_zero(op.domain)

# Create proximal operators for the dual variable

# l2-data matching
prox_convconj_l2 = odl.solvers.proximal_cconj_l2_squared(ray_trafo.range,
                                                         g=proj_data)

# Isotropic TV-regularization i.e. the l1-norm
prox_convconj_l1 = odl.solvers.proximal_cconj_l1(gradient.range, lam=0.1,
                                                 isotropic=False)

# Combine proximal operators, order must correspond to the operator K
proximal_dual = odl.solvers.combine_proximals(prox_convconj_l2,
                                              prox_convconj_l1)


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.01 * odl.power_method_opnorm(op, 20)

niter = 20  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackPrintNorm() &
            odl.solvers.CallbackShow())

# Choose a starting point
x = op.domain.zero()

# Run the algorithm
odl.solvers.chambolle_pock_solver(
    op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
    proximal_dual=proximal_dual, niter=niter, callback=callback)






# Back-projection can be done by simply calling the adjoint operator on the
# projection data (or any element in the projection space).
# backproj = ray_trafo.adjoint(proj_data)

# Shows a slice of the phantom, projections, and reconstruction
# backproj.show(title='Back-projected data', show=True)
x.show('reco final', show=True)

print('FINISHED')
