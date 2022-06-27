import gc
import numpy as np
import time
from tifffile import imread
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
matplotlib.use('tkagg')

if __name__ == '__main__':

    scan_path = '/asap3/petra3/gpfs/p07/2022/data/11014410/processed/hereon04_mhh_ts02/trans05'
    print('scan path: {}'.format(scan_path))

    num_sino = 2669
    num_proj = 3001
    num_pixel = 4601
    vol_shape = [num_sino, num_proj, num_pixel]

    gc.enable()
    print("Pre-allocate memory", end='')
    start = time.time()
    vol = np.empty(vol_shape, dtype=np.float32)
    end = time.time()
    print(" in {} s".format(end - start))

    print("Read data", end='')
    start = time.time()
    for nn in range(num_sino):
        name = scan_path + '/hereon04_mhh_ts02_{:>05}.tif'.format(nn)
        # print(name)
        vol[nn, :, :] = imread(name)
    end = time.time()
    print(" in {} s".format(end - start))

    fig, ax = plt.subplots()
    im = np.squeeze(vol[:, 1, :])
    im = ax.imshow(im, interpolation='none', cmap=cm.gray)

    plt.show()
