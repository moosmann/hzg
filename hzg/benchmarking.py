"""Benchmark i/o operations with and without concurrency."""

import gc
import numpy as np
from tifffile import imread
from metadata import MetaData
import timeit
import time
import threading
import multiprocessing
import concurrent

# TODO: parallel computing test of: threading, joblib, numba, asyncio, ...
# TODO: reading/writing/caching: numpy.memmap, pil, ...
if __name__ == '__main__':

    num_proc = round(0.5 * multiprocessing.cpu_count())
    num_proj_used = 200
    # Metadata
    scan_path = '/asap3/petra3/gpfs/p05/2020/data/11008476/raw/hzb_108_F6-32900wh/'
    # scan_path = '/asap3/petra3/gpfs/p07/2020/data/11010172/raw/swerim_21_12_oh_a/'
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


    def read_proj_into_dim2():
        for nn in range(num_proj_used):
            num = proj_range[nn]
            fn = full_proj_name_proc[num]
            proj2[:, nn, :] = imread(fn)


    def read_proj_into_dim3():
        for nn in range(num_proj_used):
            num = proj_range[nn]
            fn = full_proj_name_proc[num]
            proj3[:, :, nn] = imread(fn)


    def read_proj_multiprocessing_map():
        res = pool_mp.map(imread, full_proj_name_proc)
        # print(np.shape(res))
        return res


    def proj_reader(counter):
        filename = full_proj_name_proc[counter]
        im = imread(filename)
        return counter, im


    def read_proj_multiprocessing_imap():
        for _counter, _tmp in pool_mp.imap(proj_reader, range(num_proj_used)):
            # print('counter: {}'.format(counter))
            proj4[_counter, :, :] = _tmp


    def proj_reader2(counter):
        filename = full_proj_name_proc[counter]
        im = imread(filename)
        return im


    def read_proj_concurrent_thread(max_workers=36, chunksize=1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            res = []
            for im in executor.map(imread, full_proj_name_proc, chunksize=chunksize):
                res.append(im)
            # print(np.shape(res))
            return res


    def read_proj_threading():
        for fn in full_proj_name_proc:
            t = threading.Thread(target=imread, args=(fn,))
            t.start()


    def time_it(func, number=1, repeat=2, setup='gc.enable()'):
        name = func.__name__
        stmt = '{}()'.format(name)
        t = timeit.repeat(stmt, setup=setup, globals=globals(), number=number, repeat=repeat)
        print('{}, number: {}, repeat: {}, time: {} s'.format(name, number, repeat, t))


    def time_it2(func, max_workers, chunksize, number=1, repeat=2, setup='gc.enable()'):
        name = func.__name__
        stmt = '{}(max_workers={},chunksize={})'.format(name, max_workers, chunksize)
        t = timeit.repeat(stmt, setup=setup, globals=globals(), number=number, repeat=repeat)
        print('{}, number: {}, repeat: {}, worker: {}, chunk: {}, time: {} s'.format(
            name, number, repeat, max_workers, chunksize, t))


    def time_it3(func, repeat=1, *args, **kwargs):
        name = func.__name__
        string = ''
        for s in kwargs:
            string += f'{s}: {kwargs[s]}'
            if len(string) > 0:
                string += ', '
        tstart = time.time()
        for r in range(repeat):
            res = func(*args, **kwargs)
        tend = time.time()
        print(f'{name}, repeat: {repeat}, {string}time: {tend - tstart} s')
        rtype = type(res)
        rlen = len(res)

        rshape = np.shape(res)
        print(f'type: {rtype}, length: {rlen}, shape: {rshape}')


    if 0:
        print("Non-concurrent reading of images into different dimensions of pre-allocated arrays")
        time_it(read_proj_into_dim1, repeat=3)
        time_it(read_proj_into_dim2, repeat=3)
        time_it(read_proj_into_dim3, repeat=1)

        # Concurrent reading
        print('Open pool')
        start = time.time()
        pool_mp = multiprocessing.Pool(num_proc)
        end = time.time()
        print('Pool opened: {} s'.format(end - start))

        print("Concurrent reading using multiprocessing")
        time_it(read_proj_multiprocessing_map)
        time_it(read_proj_multiprocessing_imap)
        pool_mp.close()

    print("Concurrent reading using concurrent.threading")
    # time_it(read_proj_concurrent_thread)
    # time_it2(read_proj_concurrent_thread, 18, 1)
    time_it3(read_proj_concurrent_thread, repeat=1, max_workers=18, chunksize=1)
    # time_it2(read_proj_concurrent_thread, 36, 1)
    # time_it2(read_proj_concurrent_thread, 72, 1)
    # time_it2(read_proj_concurrent_thread, 18, 1)
    # time_it2(read_proj_concurrent_thread, 18, 10)
    # time_it2(read_proj_concurrent_thread, 18, 20)
    # time_it2(read_proj_concurrent_thread, 18, 40)
    # time_it2(read_proj_concurrent_thread, 36, 10)
    # time_it2(read_proj_concurrent_thread, 36, 20)
    # time_it2(read_proj_concurrent_thread, 36, 40)

    print("Finished")
