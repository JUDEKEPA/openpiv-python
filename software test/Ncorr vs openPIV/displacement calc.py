import matplotlib.pyplot as plt
from openpiv import tools, pyprocess, validation, filters
from openpiv.smoothn import smoothn
import numpy as np
import imageio
import h5py

import time

import matplotlib
matplotlib.use('TkAgg')

def eliminate_zero_border(displacement_array):
    # Find the indices of the non-zero elements
    non_zero_indices = np.argwhere(displacement_array != 0)

    # Determine the minimum and maximum indices for each dimension
    min_indices = non_zero_indices.min(axis=0)
    max_indices = non_zero_indices.max(axis=0) + 1  # +1 to include the max index

    # Slice the array to exclude the border
    displacement_array_trimmed = displacement_array[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1]]

    return displacement_array_trimmed

# Load the images
frame_a = tools.imread('/Users/zliu0236/Downloads/Ncorr-data analysis/HFW15.1DT1.33s_1.tif')
frame_b = tools.imread('/Users/zliu0236/Downloads/Ncorr-data analysis/HFW15.1DT1.33s_2.tif')

# Adjust frame_b dimensions
frame_b = frame_b[:, :1270]

# fig,ax = plt.subplots(1,2,figsize=(12,10))
# ax[0].imshow(frame_a,cmap=plt.cm.gray)
# ax[1].imshow(frame_b,cmap=plt.cm.gray)
#
# plt.show()

start_time = time.time()
# Perform PIV analysis
piv_u, piv_v, sig2noise = pyprocess.extended_search_area_piv(frame_a, frame_b, window_size=32, overlap=29, search_area_size=32,
                                                     sig2noise_method='peak2peak')
end_time1 = time.time()
print('Time for PIV analysis: ', end_time1 - start_time)

mask = validation.sig2noise_val( sig2noise, threshold = 1.01 )
#
# end_time2 = time.time()
#
piv_u, piv_v = filters.replace_outliers(piv_u, piv_v, mask, method='localmean', max_iter=10, kernel_size=2)
#
# end_time3 = time.time()
#
# print('Time for PIV analysis: ', end_time1 - start_time)
# print('Time for sig2noise_val: ', end_time2 - end_time1)
# print('Time for replace_outliers: ', end_time3 - end_time2)
# print('Total time: ', end_time3 - start_time)
#
# piv_u, piv_v, sig2noise = pyprocess.extended_search_area_piv(frame_a, frame_b, window_size=32, overlap=29, search_area_size=32,
#                                                     sig2noise_method='peak2peak')
# piv_u, *_ = smoothn(piv_u, s=0.5, isrobust=True)
# piv_v, *_ = smoothn(piv_v, s=0.5, isrobust=True)

# Load the HRDIC data
# with h5py.File('/Users/zliu0236/code/openpiv-python/software test/Ncorr vs openPIV/data/Ncorr r16 spacing2 u test image.h5', 'r') as f:
#     ncorr_u = f['data'][:]
#     ncorr_u = ncorr_u.T
#
# with h5py.File('/Users/zliu0236/code/openpiv-python/software test/Ncorr vs openPIV/data/Ncorr r16 spacing2 v test image.h5', 'r') as f:
#     ncorr_v = f['data'][:]
#     ncorr_v = ncorr_v.T
#
# ncorr_u = eliminate_zero_border(ncorr_u)
# ncorr_v = eliminate_zero_border(ncorr_v)
#
#
# # Smooth the displacement field
# # u2, s, exitflag, Wtot = smoothn(u, s=0.5, isrobust=True)
#
#
# fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
#
# im1 = ax[0].imshow(ncorr_v, cmap='jet')
# fig.colorbar(im1, ax=ax[0])
# ax[0].set_title('Ncorr')
#
# im2 = ax[1].imshow(piv_v, cmap='jet')
# fig.colorbar(im2, ax=ax[1])
# ax[1].set_title('PIV')
#
# plt.show()


# fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
#
# # Histogram
# ax[0].hist(np.reshape(ncorr_u, (125355, )), bins=50, color='blue', alpha=0.7, label='Ncorr')
# ax[0].hist(np.reshape(piv_u, (127204, )), bins=50, color='red', alpha=0.7, label='PIV')
# ax[0].set_title('Displacement-U')
# ax[0].set_xlabel('Value')
# ax[0].set_ylabel('Frequency')
# ax[0].legend()
#
# ax[1].hist(np.reshape(ncorr_v, (125355, )), bins=50, color='blue', alpha=0.7, label='Ncorr')
# ax[1].hist(np.reshape(piv_v, (127204, )), bins=50, color='red', alpha=0.7, label='PIV')
# ax[1].set_title('Displacement-V')
# ax[1].set_xlabel('Value')
# ax[1].set_ylabel('Frequency')
# ax[1].legend()
#
#
# # Show the plots
# plt.show()