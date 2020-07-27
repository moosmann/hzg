# From *.pca file
# [Geometry]
# FDD=399.99900000
# FOD=16.60243750
# Magnification=24.09278758
# VoxelSizeX=0.00207531
# VoxelSizeY=0.00207531
# CalibValue=-1.097
# cx=1141.50000000
# cy=1151.50000000
# DetectorRot=0.00000000
# Tilt=0.00000000
# Old_CalibValue=0.00000000
#
# [CT]
# Type=0
# NumberImages=2400
# StartImg=2394
# RotationSector=360.00000000
# NoRotation=0
# EstimatedTime=0
# RemainingTime=2400
# ScanTimeCmpl=2400
# NrImgDone=2393
# NrImgCmplScan=2401
# RefDriveEnabled=0
# SkipForNewInterval=25
# UpdateView=1
# SkipAcc=1
# FreeRayFactor=1.00003000
# Wnd_L=2062
# Wnd_T=10
# Wnd_R=2283
# Wnd_B=2303
# Level=3048.00000000
#
# [VSensor]
# EnableTiles=3
# Start=0
# NumTiles=1
# Interval=60
# Overlap=9
# AdjustImg=1
# SingleImgX=2304
#
# [Trajectory]
# Active=0
#
# [CalibValue]
# NumberImages=18
# Averaging=2
# Skip=3
#
# [FastCT]
# Active=0
#
# [Image]
# Top=0
# Left=0
# Bottom=2303
# Right=2303
# DimX=2284
# DimY=2304
# Rotation=0
# MirrorX=0
# MirrorY=0
# BitPP=16
# FreeRay=3401
#
# [ImgProc]
# SwBin=0
# AddSwBin=0
#
# [Warmup]
# Enable=1
# Counter=0
# MaxTimes=10
# TimeTrigOn=0
# kV=185
# Time=60
#
# [Multiscan]
# Active=0
#
# [Multiline]
# Installed=0
#
# [CalibImages]
# MGainMode=0
# MGainPoints=3
# Avg=20
# Skip=10
# EnableAutoAcq=1
# MGainVoltage=60:60:60:
# MGainCurrent=20:50:170:
# GainImg=\\192.168.10.2\VTX_REC_S\syncroload\Mg-5Gd-50-PEEK-holder\Mg-5Gd-50-PEEK-holder_bright_60kV_170uA_3000ms_1Det.tif
# MGainImg=\\192.168.10.2\VTX_REC_S\syncroload\Mg-5Gd-50-PEEK-holder\Mg-5Gd-50-PEEK-holder_bright_60kV_170uA_3000ms_1Det.tif
# OffsetImg=\\192.168.10.2\VTX_REC_S\syncroload\Mg-5Gd-50-PEEK-holder\Mg-5Gd-50-PEEK-holder_Dark_3000.tif
# DefPixelImg=C:\Program Files\phoenix x-ray\datosx 2 acq\CalibrationImages\pixmask_B1x1_x2304_y2304.tif
#
# [SectorScan]
# Active=0
#
# [DetectorShift]
# Enable=1
# Mode=0
# Amplitude=10
# Interval=1
# Step=1
#
# [Detector]
# InitTimeOut=20000
# Name=ham-c7942
# PixelsizeX=0.05000000
# PixelsizeY=0.05000000
# NrPixelsX=2304
# NrPixelsY=2304
# Timing=7
# TimingVal=3000.000
# Avg=3
# Skip=1
# Binning=0
# BitPP=12
# CameraGain=0
# SatValue=3890
# SatPixNrLimit=5308
#
# [Xray]
# ComPort=0
# Name=xs|180 hpnf
# ID=1240
# InitTimeout=15000
# Voltage=60
# Current=170
# Mode=0
# Filter=Unknown
# Collimation=-1
# WaitTime=5000
# WaitForStable=20000
# FocDistX=0.00000000
# FocDistY=0.00000000
# SpinStepkV=10
# SpinStepuA=10
# Macro=0
# RestrictNumSpots=0
# PreWarning=0
# MinGainCurrent=20
#
# [Cnc]
# InitTimeout=8000
# CollisionDetection=0
# JoyDriveDoorOpen=0
# SecPosSample=85.00000000
# MinSampleDetPos=180.00000000
# EnableKeyboardJoy=0
# KeyJoyVelocityFactor=0.25000000
#

#
# [Axis]
# XSample=0.000000
# YSample=74.999958
# ZSample=16.602438
# RSample=358.950300
# TSample=0.000000
# YTube=0.000000
# ZTube=0.000000
# XDetector=-0.150500
# YDetector=0.000000
# ZDetector=399.999000
# WDetector=0.000000
# ColSlave=0.000000


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import scipy
import odl
import os

# --- DATA --- #

# Projection folder and file names
par_folder = '/asap3/petra3/gpfs/p05/2016/data/11002518/processed/synchroload'
folder = 'Mg-5Gd-50-PEEK-holder_p2400_t3000ms_a3_s1_shift_z16.6'

filename ='Mg-5Gd-50-PEEK-holder00001.tif'
prefix = 'Mg-5Gd-50-PEEK-holder'
name = os.path.join(par_folder, folder, filename)

proj_range = np.arange(1, 2401, 20)
proj_num = proj_range.size
im = np.rot90(scipy.misc.imread(name, mode='F'), k=-1)
DimX, DimY = im.shape # 2284, 2304
print('proj: shape = {}, {}'.format(DimX, DimY))
roiDimY = 1000
roi = np.arange(DimY / 2 - roiDimY / 2,DimY / 2 - roiDimY / 2 + roiDimY, dtype=int)
DimY = roiDimY
proj_shape = [proj_range.size, DimX, DimY]
full_angle_deg = 360
full_angle_rad = full_angle_deg * np.pi / 180


# Read projections
data = np.zeros(proj_shape)
for nn in range(proj_num):
    filename = '{}{:05d}.tif'.format(prefix, proj_range[nn])
    name = os.path.join(par_folder, folder, filename)
    data[nn, :, :] = np.rot90(scipy.misc.imread(name, mode='F'), k=-1)[:, roi]
print('sino: shape = {}'.format(proj_shape))


# --- Domain, range, forward operator ---#

# Discrete reconstruction space: discretized functions on the
dhalf = 3
nsamples = 100
reco_space = odl.uniform_discr(
    min_corner=[-dhalf, -dhalf, -dhalf], max_corner=[dhalf, dhalf, dhalf],
    nsamples=[nsamples, nsamples, nsamples], dtype='float32')

# Make a circular cone beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, full_angle_rad, proj_num)
# Detector
pixelsize = 0.05
lenXhalf = DimX * pixelsize / 2
lenYhalf = DimY * pixelsize / 2
detector_partition = odl.uniform_partition([-lenXhalf, -lenYhalf], [lenXhalf, lenYhalf], [DimX, DimY])
ZSample = 16.602438
ZDetector = 399.999000
print('proj: (width,height) = ({},{})'.format(2 * lenXhalf, 2 * lenYhalf))
# Geometry
geometry = odl.tomo.CircularConeFlatGeometry(
    angle_partition, detector_partition, src_radius=ZSample, det_radius=ZDetector-ZSample, axis=[1, 0, 0])

# Ray transform aka forward projection using ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Create element in projection space
proj_data = ray_trafo.range.element(data)


# Back-projection: adjoint operator on the projection data
# x = ray_trafo.adjoint(proj_data)


# Optionally pass partial to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

# Choose a starting point
# x = ray_trafo.domain.zero()
x = reco_space.zero()

# Run the algorithm
odl.solvers.conjugate_gradient_normal(
    ray_trafo, x, proj_data, niter=20, callback=callback)

# Display images

proj_data.show(coords=[0, None, None], title='first projection')
proj_data.show(coords=[None, None, 0.5], title='sinogram at half height')
proj_data.show(show=False, coords=[None, 0.5, None], title='sinogram at half width')

x.show(coords=[0.5, None, None], title='dim 0 at half width')
x.show(coords=[None, 0.5, None], title='dim 1 at half width')
x.show(show=True, coords=[None, None, 0.5], title='dim2 at half width')

print('FINISHED')

