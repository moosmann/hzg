#
# def read_img(filename):
#     """Read Beckmannsches data format.
#
#     Header of 19 bytes following little-endian binary image data.
#
#     Parameters
#     ----------
#     filename : 'string'
#
#     Returns
#     -------
#     image : array-like
#
#     Examples
#     --------
#     >>> filename = '/asap3/petra3/gpfs/p05/2016/data/11002519/processed/bwi_01/bwi_01_c/sino/bwi_01_c01389.sis'
#     >>> header, image = read_img(filename)
#     >>> print(header)
#     PC_2_F_1529_2400_
#     >>> print(image.size, image.shape)
#
#
# """
#     import numpy as np
#     with open(filename, 'rb') as f:
#         # Read 19 byte header
#         header = f.read(17)
#         # Assign remaining binary data
#         image = np.fromfile(f)
#
#
#         # Read dimension from header. TODO: User regexp
#         dim1 = int(header[7:11])
#         dim2 = int(header[12:16])
#
#         np.reshape(image, (dim1, dim2))
#
#     #str = '*uint16';
#     print(image.shape, dim1, dim2)
#     f.close()
#
#     return header, image