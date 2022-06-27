from metadata import MetaData
import matplotlib as mpl
import matplotlib.pyplot as plt
from tifffile import imread
import time


print(f'backend: {plt.get_backend()}')
# mpl.use('qt5agg')
plt.ion()
# plt.ioff()
# plt.figure()
# mpl.style.use('fast')

scan_path = '/asap3/petra3/gpfs/p05/2020/data/11008476/raw/hzb_108_F6-32900wh/'
md = MetaData(scan_path)
fn = md.full_proj_name[0]
im = imread(fn)

cmap = 'gray'
# plt.style.use('dark_background')
start = time.time()
# fig = plt.figure('window title', tight_layout=True)
fig, ax = plt.subplots(num=1)
fig.canvas.set_window_title('Window title')
# ax = fig.add_subplot(1, 1, 1)
ax = ax.imshow(im, cmap=cmap, interpolation='nearest')
ax.set_xlabel('horizontal')
ax.set_ylabel('vertical')
ax.set_title('image')
fig.suptitle('suptitle')
sm = mpl.cm.ScalarMappable(norm=None, cmap=cmap)
fig.colorbar(sm, ax=ax)
# p = plt.imshow(im, cmap='gray', interpolation='nearest')
plt.show()
# plt.colorbar(ax=ax)
end = time.time()

print(f'FINISHED: {end - start}')
