import numpy as np
from tifffile import imread


   if __name__ == '__main__':

    num_proc = round(0.5 * multiprocessing.cpu_count())
    num_proj_used = 200
    # Metadata
    # scan_path = '/asap3/petra3/gpfs/p05/2020/data/11008476/raw/hzb_108_F6-32900wh/'
    scan_path = '/asap3/petra3/gpfs/p07/2020/data/11010172/raw/swerim_21_12_oh_a/'
    md = MetaData(scan_path)
    num_proj_found = md.num_proj_found
    im_shape_raw = md.im_shape
    im_type = md.im_dtype
    print('number of projections found: %u' % num_proj_found)
    print('projection image shape: {}'.format(im_shape_raw))
    print('projection image type: {}'.format(im_type))

    # Sub-range
    proj_range = [i for i in range(num_proj_used)]
    full_proj_name_proc = md.full_proj_name[proj_range]

    # Allocate memory
    proj_shape1 = [num_proj_used, im_shape_raw[0], im_shape_raw[1]]
    proj_shape2 = [im_shape_raw[0], num_proj_used, im_shape_raw[1]]
    proj_shape3 = [im_shape_raw[0], im_shape_raw[1], num_proj_used]
    proj_shape4 = proj_shape1

    gc.enable()
    print("Pre-allocate memory", end='')
    start = time.time()
    proj1 = np.empty(proj_shape1, dtype=np.uint16)
    proj2 = np.empty(proj_shape2, dtype=np.uint16)
    proj3 = np.empty(proj_shape3, dtype=np.uint16)
    proj4 = np.empty(proj_shape4, dtype=np.uint16)
    end = time.time()
    print(" in {} s".format(end - start))
    print('proj shape1: {}'.format(proj1.shape))
    print('proj shape2: {}'.format(proj2.shape))
    print('proj shape3: {}'.format(proj3.shape))
    print('proj shape4: {}'.format(proj4.shape))
    print('memory per volume: {} GiB'.format(proj1.nbytes / 1024 ** 3))

    def read_proj_into_dim1():
        for nn in range(num_proj_used):
            num = proj_range[nn]
            fn = full_proj_name_proc[num]
            proj1[nn, :, :] = imread(fn)
            # name = os.path.join(par_folder, folder, filename)
            # data[nn, :, :] = np.rot90(scipy.misc.imread(name, mode='F'), k=-1)[:, roi]


