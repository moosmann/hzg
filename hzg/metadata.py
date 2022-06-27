"""Extract metadata from scan."""

import os
from warnings import warn
import glob
import h5py
import numpy as np
from tifffile import TiffFile, imread


class MetaData:
    """Retrieve metadata from scan path."""

    def __init__(self, scan_path: str):
        if not scan_path:
            raise ValueError('scan_path must not be empty')
        if scan_path.endswith(os.sep):
            scan_path = scan_path[0:-1]
        self.scan_path = scan_path
        self.raw_path, self.scan_name = os.path.split(scan_path)
        self.beamtime_path, tmp = os.path.split(self.raw_path)
        if tmp != 'raw':
            warn("Parent folder name ({}) is not 'raw'".format(tmp))
        tmp, self.beamtime_id = os.path.split(self.beamtime_path)
        self.processed_path = os.path.join(self.beamtime_path, 'processed')
        self.scratch_path = os.path.join(self.beamtime_path, 'scratch_cc')

        # log files
        tmp = glob.glob(os.path.join(self.scan_path, '*_nexus.h5'))
        self.h5_log = None if not tmp else tmp[0]
        tmp = glob.glob(os.path.join(self.scan_path, '*scan.log'))
        self.scan_log = None if not tmp else tmp[0]
        tmp = glob.glob(os.path.join(self.scan_path, '*image.log'))
        self.image_log = None if not tmp else tmp[0]

        self.im_name = None
        self.im_time = None
        self.im_key = None
        self.piezo = None

        # read h5 log
        if self.h5_log is not None:
            with h5py.File(self.h5_log, "r") as f:
                tmp = f['/entry/scan/data/image_file/value'][:]
                if tmp.size != 0:
                    self.im_name = tmp
                    self.im_time = f['/entry/scan/data/image_file/time'][:]
                    self.im_key = f['/entry/scan/data/image_key/value'][:]
                    self.im_key_time = f['/entry/scan/data/image_key/time'][:]

                    if self.im_name.size != self.im_time.size:
                        raise Exception('H5 log entry mismatch in image file')
                    if self.im_key.size != self.im_key_time.size:
                        raise Exception('H5 log entry mismatch in image key')
                    if self.im_name.size != self.im_key_time.size:
                        raise Exception('H5 log entry mismatch in image key and image file')
                    if not np.array_equal(self.im_time, self.im_key_time):
                        raise Exception('H5 log entry image file times and image key times differ')

        # read image log
        if self.image_log is not None:
            dt = [('im_name', np.object),
                  ('im_time', 'f4'),
                  ('im_key', 'u2'),
                  ('angle', 'f4'),
                  ('s_stage_x', 'f4'),
                  ('piezo', 'f4'),
                  ('petra', 'f4')]
            im_name, im_time, im_key, im_angle, im_s_stage_x, im_piezo, im_petra = np.loadtxt(
                self.image_log, dtype=dt, encoding='utf8', unpack=True)

            if self.im_name is None:
                self.im_name = im_name
            else:
                if not np.array_equal(self.im_name, im_name):
                    raise Exception('Image names from H5 log and image log differ')

            if self.im_time is None:
                self.im_time = im_time

            if self.im_key is None:
                self.im_key = im_key
            else:
                if not np.array_equal(self.im_key, im_key):
                    raise Exception('Image keys from H5 log and image log differ')

        # image name
        self.proj_name = self.im_name[self.im_key == 0]
        self.ref_name = self.im_name[self.im_key == 1]
        self.dark_name = self.im_name[self.im_key == 2]
        if self.proj_name[0].startswith(os.sep):
            tmp = self.scan_path
        else:
            tmp = self.scan_path + os.sep

        # Number of images found
        self.num_proj_found = self.proj_name.size
        self.num_ref_found = self.ref_name.size
        self.num_dark_found = self.dark_name.size

        # Consistency checks
        if self.num_dark_found > self.num_ref_found:
            warn('Number of dark fields ({}) greater than number of flat fields ({})'.format(
                self.num_dark_found, self.num_ref_found))
        if self.num_ref_found > self.num_proj_found:
            warn('Number of flat fields ({}) greater than number of projections ({})'.format(
                self.num_ref_found, self.num_proj_found))

        # full image path
        self.full_proj_name = tmp + self.proj_name
        self.full_ref_name = tmp + self.ref_name
        self.full_dark_name = tmp + self.dark_name

        # Read  image metadata
        fn = self.full_proj_name[0]
        tif = TiffFile(fn)
        page = tif.pages[0]
        self.im_shape = page.shape
        self.im_size = page.size
        self.im_nbytes = np.dtype(page).itemsize * page.size
        self.page = page
        self.im_dtype = page.dtype
        if self.im_dtype is not np.dtype('uint16'):
            raise TypeError(f'image data type ({self.im_dtype}) is not uint16')

        if self.piezo is None:
            self.proj_shape = (self.num_proj_found, *self.im_shape)
        else:
            self.proj_shape = (self.num_proj_found, len(self.piezo[0]), *self.im_shape)
        # ROI, move to parameters
        self.roi_vert = None
        self.roi_hor = None
        self.roi_angles = None

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.scan_path)

    def imread_dark(self, num=0):
        return imread(self.full_dark_name[num])

    def imread_ref(self, num=0):
        return imread(self.full_ref_name[num])

    def imread_proj(self, num=0):
        return imread(self.full_proj_name[num])

    def info(self):
        """Print general scan information."""
        k0 = ['scan_name',
              'beamtime_id',
              'scan_path',
              'raw_path',
              'beamtime_path',
              'processed_path',
              'scratch_path',
              'h5_log',
              'scan_log',
              'image_log',
              'num_proj_found',
              'num_ref_found',
              'num_dark_found',
              'im_shape',
              'im_size',
              'im_dtype',
              'im_nbytes',
              'proj_shape',
              ]
        for name in k0:
            print(f'{name}: {self.__dict__[name]}')

    def info_images(self):
        """Print additional image file related info."""
        k1 = ['proj_name', 'ref_name', 'dark_name',
              'full_proj_name', 'full_ref_name', 'full_dark_name']
        k2 = ['im_name',
              'im_key',
              'im_time',
              'im_key_time']
        for name in k1:
            print(f'{name}[0]: {self.__dict__[name][0]}')
            print(f'{name}[-1]: {self.__dict__[name][-1]}')
        for name in k2:
            print(f'{name}: {self.__dict__[name]}')
        for key in [0, 1, 2]:
            name = self.im_name[self.im_key == key][0]
            time = self.im_time[self.im_key == key][0]
            print(f'image: key, name, time = {key}, {name}, {time}')


class MetaDataSets(MetaData):
    """Default constructors for exemplary metadata sets."""

    @classmethod
    def p05_kit_h5log(cls):
        return cls('/asap3/petra3/gpfs/p05/2020/data/11008476/raw/hzb_108_F6-32900wh/')

    @classmethod
    def p05_ccd_h5log(cls):
        return cls('/asap3/petra3/gpfs/p05/2020/data/11008672/raw/bkk_006_goldfish_inoculate')

    @classmethod
    def p07_ximea50mpix_imlog(cls):
        return cls('/asap3/petra3/gpfs/p07/2020/data/11010172/raw/swerim_21_12_oh_a/')

    @classmethod
    def hnee19(cls):
        return cls('/asap3/petra3/gpfs/p05/2018/data/11004450/raw/hnee19_pappel_oppositeWood_000')

    @classmethod
    def hnee21(cls):
        return cls('/asap3/petra3/gpfs/p05/2018/data/11004450/raw/hnee21_pappel_oppositeWood_000')


if __name__ == '__main__':
    print('Instantiate class methods:')
    for key, val in MetaDataSets.__dict__.items():
        if type(val) is classmethod:
            print(getattr(MetaDataSets, key)())
    md = MetaDataSets.p07_ximea50mpix_imlog()
    print('End')
