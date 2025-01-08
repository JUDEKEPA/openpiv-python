import sys
sys.path.append('/Users/zliu0236/code/DefDAP/')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import defdap.hrdic as dic
import numpy as np
import matplotlib.pyplot as plt

import defdap.hrdic as hrdic
import defdap.ebsd as ebsd
import defdap.manual_dic as manual_dic

import h5py


def eliminate_zero_border(displacement_array):
    # Find the indices of the non-zero elements
    non_zero_indices = np.argwhere(displacement_array != 0)

    # Determine the minimum and maximum indices for each dimension
    min_indices = non_zero_indices.min(axis=0)
    max_indices = non_zero_indices.max(axis=0) + 1  # +1 to include the max index

    # Slice the array to exclude the border
    displacement_array_trimmed = displacement_array[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1]]

    return displacement_array_trimmed

with h5py.File('/Users/zliu0236/code/openpiv-python/software test/Ncorr vs openPIV/data/Ncorr strain radius1 exx test image.h5', 'r') as f:
    ncorr_exx1 = f['data'][:]
    ncorr_exx1 = ncorr_exx1.T

with h5py.File(
        '/Users/zliu0236/code/openpiv-python/software test/Ncorr vs openPIV/data/Ncorr strain radius1 exy test image.h5',
        'r') as f:
    ncorr_exy1 = f['data'][:]
    ncorr_exy1 = ncorr_exy1.T

with h5py.File(
        '/Users/zliu0236/code/openpiv-python/software test/Ncorr vs openPIV/data/Ncorr strain radius1 eyy test image.h5',
        'r') as f:
    ncorr_eyy1 = f['data'][:]
    ncorr_eyy1 = ncorr_eyy1.T


ncorr_exx1 = eliminate_zero_border(ncorr_exx1)
ncorr_exy1 = eliminate_zero_border(ncorr_exy1)
ncorr_eyy1 = eliminate_zero_border(ncorr_eyy1)

# Load the HRDIC data
with h5py.File('/Users/zliu0236/code/openpiv-python/software test/Ncorr vs openPIV/data/Ncorr r16 spacing2 u test image.h5', 'r') as f:
    u_displc = f['data'][:]
    u_displc = u_displc.T

with h5py.File('/Users/zliu0236/code/openpiv-python/software test/Ncorr vs openPIV/data/Ncorr r16 spacing2 v test image.h5', 'r') as f:
    v_displc = f['data'][:]
    v_displc = v_displc.T

u_displc_trimmed = eliminate_zero_border(u_displc)
v_displc_trimmed = eliminate_zero_border(v_displc)

a = manual_dic.Map(32, 411, 305, np.array([2, 5, 8]), np.array([2, 2, 2]), u_displc_trimmed, v_displc_trimmed)

e11 = a.e11
e22 = a.e22
e12 = a.e12

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].imshow(e11, cmap='jet')
ax[1].imshow(ncorr_exx1, cmap='jet')

plt.show()

# b = hrdic.Map('/Users/zliu0236/Downloads', 'export.txt')
# fieldWidth = 20 # microns
# numPixels = 2048
# pixelSize = fieldWidth / numPixels
# a.binning = 3
#
# a.setScale(pixelSize)
# a.plotMaxShear(vmin=0, vmax=0.10, plotScaleBar=True)
# # a.plotMaxShear()
# # a.setScale()
# plt.show()

