import numpy as np
from filter import filter_pixel
from metadata import MetaDataSets

im = 100 * np.ones([7, 7], dtype='uint16')
iinfo = np.iinfo(im.dtype)
im[2, 2] = iinfo.min  # dead pixel
im[2, -2] = iinfo.min + 1  # dark pixel
im[-2, -2] = iinfo.max  # overexposed pixel
im[-2, 2] = iinfo.max - 1  # hot pixel


class TestPixelFilter:
    def test_filter_pixel_do_nothing(self):
        imf = filter_pixel(im, thresh_hot=0, thresh_dark=0, filt_over=False, filt_dead=False)
        assert np.array_equal(imf, im)
        assert imf is im

    def test_filter_pixel_dead(self):
        imf = filter_pixel(im, thresh_hot=0, thresh_dark=0, filt_over=False, filt_dead=True)
        assert imf.min() != 0
