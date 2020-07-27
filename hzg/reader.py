# # Imports for common Python 2/3 codebase
# from __future__ import print_function, division, absolute_import
# from future import standard_library
#
# standard_library.install_aliases()

import numpy
import io


class ReadDat:
    """
    The read_dat class reads tomo files written with the write_dat routine of
    the IDL tomo package (idlprogs).
    """

    def __init__(self, path):
        self.fullpath = path
        self.dtype = None
        self.dimensions = None
        self.dimsize = None
        self.header = None
        self.data = None

    def __call__(self, path=None):
        if path is not None:
            self.fullpath = path
        self.check_header()
        self.read_file()

    def read_file(self):
        """Load binary data to python array."""
        with io.open(self.fullpath, 'r', encoding='utf8', errors='ignore') as f:
            # Now seek forward until beginning of file or we get a \n
            while True:
                ch = f.read(1)
                if ch == '\n':
                    break
            # load data
            self.data = numpy.fromfile(f, numpy.dtype(self.dtype))
            # shape new array on Fortran column major order (used by IDL)
            self.data = numpy.reshape(self.data, self.dimsize, order='F')

    def check_header(self):
        """
        Reads header and returns file info to class variables.
        """
        with io.open(self.fullpath, mode='r', encoding='utf8', errors='ignore') as f:
            # read first 100 characters of the file (header).
            f_100 = f.read(100)  # TAG
            # read first 100 characters, split EOL
            first100char = f_100.split('\r\n')
            self.header = first100char[0]
            # split header at underscores
            header_list = self.header.split('_')
            # get header information
            self.dimensions = int(header_list[1])
            idldtype = header_list[2]
            # get data dimensions
            self.dimsize = [0] * self.dimensions
            for i in range(self.dimensions):
                self.dimsize[i] = int(header_list[3 + i])
            # convert idl data type to python dtype
            dtype_idl2numpy = {'B': 'uint8', 'I': 'int16', 'U': 'uint16', 'L': 'int32', 'F': 'float32', 'D': 'float64',
                               'C': 'complex64'}
            self.dtype = dtype_idl2numpy[idldtype]
