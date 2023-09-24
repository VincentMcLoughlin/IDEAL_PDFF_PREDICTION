import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

#np_path = "/home/mclougv/IDEAL_PDFF_prediction/X20254_micro_dark_files/1767750_dark.npy"
np_path = "/genetics3/mclougv/IDEAL_x20254_dark_files/1000015_dark.npy"
np_data = np.load(np_path)
print(np_data.shape)
#display_data = np_data[:,:,0]
#print(display_data.shape)
#display_data *= 4095

channel=0
plt.imshow(np_data[:,:,channel], cmap=plt.cm.gray)
plt.savefig("tmp_dark.png")

# print(display_data.shape)
# plt.imshow(display_data, interpolation='nearest')
# plt.gray()
# plt.show()
# plt.savefig("test.png")

