# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:43:48 2020

@author: moosmanj
"""

#import mayavi.mlab as mm
#import numpy as np
from plotly.offline import plot
from tifffile import imread
import plotly.express as px
import plotly.graph_objects as go

print('go.Figure')
fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
# print('show')
# fig.show()
# plot(fig, auto_open=True)

p = '/asap3/petra3/gpfs/p05/2020/data/11008823/processed/nova004_pyrochroa_coccinea_a/reco/float_rawBin2/nova004_pyrochroa_coccinea_a01248.tif'

im = imread(p)
fig = px.imshow(im)
fig = go.Figure(go.Image(z=im))
fig.show()
plot(fig, auto_open=True)

#mm.imshow(im, colormap='gray')
#mm.show()

