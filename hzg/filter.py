"""Filter functions."""
# import numba
import numpy as np
import scipy.ndimage
from time import time
from metadata import MetaDataSets
from skimage.filters.rank import median
# from skimage.filters import median
from skimage.morphology import disk
# import statistics


# @numba.njit(fastmath=True)
# @jit(uint16(uint16))
# @numba.guvectorize(
#     [numba.void(numba.uint16[:, :], numba.uint16[:, :])],
#     '(n,m) -> (n,m)',
#     # nopython=True,
#     # cache=True,
#     target='cuda'
# )
# @numba.jit(nopython=True)
def medfilt2d(im, out):
    # nx, ny = im.shape
    # for x in numba.prange(1, nx):
    #     for y in numba.prange(1, ny):
    #         k = im[x - 1:x + 1, y - 1:y + 1]
    #         k.flatten()
    #         out[x, y] = 1
    #         # k_med = np.median(k)
    #         # out[x, y] = k_med
    # scipy.ndimage.median_filter(im, size=(3, 3), mode='reflect', output=out)
    # median(im, out=out, mode='reflect', behavior='ndimage')
    # median(im, out=out, behavior='rank')
    median(im, np.ones([3, 3], dtype=np.uint16), out=out)


def filter_pixel(im: np.ndarray, thresh_hot: float, thresh_dark: float,
                 filt_over: bool = 1, filt_dead: bool = 1,
                 median_filter_size: tuple = (3, 3), median_filter_mode: str = 'reflect',
                 verbose: bool = 0):
    """Filter pixel outliers.

    :type im: numpy.ndarray
    :param verbose:
    :param median_filter_mode:
    :param median_filter_size:
    :param filt_dead:
    :param filt_over:
    :param thresh_dark:
    :param thresh_hot:
    """

    if not np.any([thresh_hot, thresh_dark, filt_over, filt_dead]):
        return im

    dtype = im.dtype

    if verbose:
        tsort = 0
        im_min = im.min()
        im_max = im.max()
        im_mean = im.mean()
        im_std = im.std()
        num_dead = 0
        num_over = 0
        num_hot = 0
        num_dark = 0
        start = time()
        print('Start')

    mask = np.zeros(im.shape, dtype=bool)
    if verbose:
        print(f'mask initialized: {time() - start}')

    if filt_dead:
        mask = im == np.iinfo(dtype).min
        if verbose:
            print(f'dead pixel mask: {time() - start}')
            num_dead = np.sum(mask)

    if filt_over:
        mask += im == np.iinfo(dtype).max
        if verbose:
            print(f'overexposed pixel mask: {time() - start}')
            num_over = np.sum(mask) - num_dead

    # mean value without dead and overexposed pixels
    im_mean_masked = im[~mask].mean()
    if verbose:
        print(f'median of pre-filtered mask: {time() - start}')

    # pre-filter image to avoid issues with clustered overexposed or dead pixels
    im[mask] = im_mean_masked
    if verbose:
        print(f'pre-filtering: {time() - start}')

    # im_med = median_filter(im, size=median_filter_size, mode=median_filter_mode)
    im_med = np.ones_like(im)
    print(f'im type: {im.dtype}')
    print(f'im_med type: {im_med.dtype}')
    medfilt2d(im, im_med)
    # size=median_filter_size, mode=median_filter_mode)

    if verbose:
        print(f'median filtered image: {time() - start}')
        print(f'median filtered image dtype: {im_med.dtype}')
    if thresh_dark > 0 or thresh_hot > 0:
        r = np.ndarray(im.shape, dtype=np.float32)
        np.divide(im, im_med, out=r)
        # r = im / im_med
        if verbose:
            print(f'ratio image: {time() - start}')
            print(f'ratio image dtype: {r.dtype}')
        if thresh_dark < 0.5 and thresh_hot < 0.5:
            if verbose:
                tsort_start = time()
            r_sorted = np.sort(r, axis=None)
            if verbose:
                print(f'ratio sorted: {time() - start}')
                tsort = time() - tsort_start
            if thresh_dark < 0.5:
                ind = int(np.ceil(thresh_dark * im.size))
                print(f'dark ind: {ind}')
                thresh_dark = r_sorted[ind]
            if thresh_hot < 0.5:
                ind = int(np.floor((1 - thresh_hot) * im.size))
                print(f'hot ind: {ind}')
                thresh_hot = r_sorted[ind]

    if thresh_dark > 0:
        mask += r < thresh_dark
        if verbose:
            print(f'dark mask: {time() - start}')
            num_dark = np.sum(mask) - num_dead - num_over

    if thresh_hot > 0:
        mask += r > thresh_hot
        if verbose:
            print(f'hot mask: {time() - start}')
            num_hot = np.sum(mask) - num_dead - num_over - num_dark

    # filter image
    im[mask] = im_med[mask]
    if verbose:
        print(f'image filtered: {time() - start}')

        im_min_filt = im.min()
        im_max_filt = im.max()
        im_mean_filt = im.mean()
        im_std_filt = im.std()
        print(f'\nPARAMETERS and STATISTICS')
        print(f'thresh_hot: {thresh_hot}')
        print(f'thresh_dark: {thresh_dark}')
        print(f'filt_over: {filt_over}')
        print(f'filt_dead: {filt_dead}')
        print(f'median_filter_size: {median_filter_size}')
        print(f'median_filter_mode: {median_filter_mode}')
        print('BEFORE / AFTER FILTERING:')
        print(f'im_min: {im_min} / {im_min_filt}')
        print(f'im_max: {im_max} / {im_max_filt}')
        print(f'im_mean: {im_mean} / {im_mean_filt}')
        print(f'im_std: {im_std} / {im_std_filt}')
        print(f'im_mean_masked: {im_mean_masked}')
        print(f'num_dead: {num_dead} ({100 * num_dead / im.size}%)')
        print(f'num_over: {num_over} ({100 * num_over / im.size}%)')
        print(f'num_dark: {num_dark} ({100 * num_dark / im.size}%)')
        print(f'num_hot: {num_hot} ({100 * num_hot / im.size}%)')
        print(f'sort time: {tsort}')
        print(f'elapsed time: {time() - start}')

    return im


if __name__ == '__main__':
    im0 = 100 * np.ones([7, 7], dtype='uint16')
    im0 += np.random.randint(10, size=im0.shape, dtype=im0.dtype)
    im_limits = np.iinfo(im0.dtype)
    im0[2, 2] = im_limits.min
    im0[2, -2] = im_limits.min + 1
    im0[4, 4] = im_limits.max
    im0[-2, 2] = im_limits.max - 1
    # print(im0)
    # imf = filter_pixel(im0, thresh_dark=0.9, thresh_hot=1.1, filt_dead=True, filt_over=True, verbose=True)
    # print(imf)

    print('\nFILTER XIMEA IMAGE')
    im = MetaDataSets.p07_ximea50mpix_imlog().imread_dark()
    imf = filter_pixel(im, thresh_hot=0.05, thresh_dark=0.02, filt_over=True, filt_dead=True, verbose=True)
    print('END')
