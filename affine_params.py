import DMD_utilities as dmd
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import sys
from math import cos, sin
import scipy


sys.path.append('C:\\Users\\Light Sheet User\\Desktop\\dmd_2')
dmd_rows = 1200; dmd_cols = 1920
camera_rows = 2304; camera_cols = 2304

R = dmd.create_target_for_affine_transform()
points = np.load('affine_target_points.npy') # theoretical

# get XP coords
mask = np.zeros((camera_rows, camera_cols))
mask[500:1920, 650:1750] = 1 # xp checked
Rxp = imageio.imread('R_camera.tiff') * mask # avoid DMD bright borders

Rxp = Rxp[:, ::-1]
peaks = np.zeros((5, 2))
cut_range = 20
for i in range(5):
    peaks[i, 0], peaks[i, 1] = np.unravel_index(np.argmax(Rxp), Rxp.shape)
    Rxp[int(peaks[i, 0]-cut_range):int(peaks[i, 0]+cut_range),
        int(peaks[i, 1]-cut_range):int(peaks[i, 1]+cut_range)] = 0
theta = np.pi / 4
rotation_matrix = np.array([[cos(theta), -sin(theta)],
                                [sin(theta), cos(theta)]])
peaks = (rotation_matrix@peaks.T).T




peaks_2 = np.asarray(((1152, 1152), (1152, 1152)))
peaks_2[:, :] = peaks_2[:, ::-1]
peaks_2 = (rotation_matrix@peaks_2.T).T





peaks, ref_row, ref_col = dmd.center(peaks)
points, ref_center_row, ref_center_col = dmd.center(points)
ref_norm = np.linalg.norm(points)
norm = np.linalg.norm(peaks)
peaks = ref_norm/norm * peaks




for i in range(peaks_2.shape[0]):
    peaks_2[i, 0] -= ref_row[0]; peaks_2[i, 1] -= ref_col[0]
peaks_2 = ref_norm/norm * peaks_2





res = scipy.optimize.minimize(dmd.letter_rotation, 0,
                              args=(peaks, points),
                              method="L-BFGS-B")
theta = res.x[0] * np.pi / 180 / 2
rotation_matrix = np.array([[cos(theta), -sin(theta)],
                                [sin(theta), cos(theta)]])
peaks = (rotation_matrix@peaks.T).T







matrix = np.ones((dmd_rows, dmd_cols), dtype=np.uint8)
for i in range(peaks_2.shape[0]):
    peaks_2[i, 0] += ref_center_row[0]; peaks_2[i, 1] += ref_center_col[0]
    matrix[int(peaks_2[i, 0]-5):int(peaks_2[i, 0]+5), 
    int(peaks_2[i, 1]-5):int(peaks_2[i, 1]+5)] = 0
imageio.imwrite('T2.png', matrix)






# plt.scatter(peaks[:, 1] , peaks[:, 0])
# plt.scatter(points[:, 1], points[:, 0])
# plt.show()

params = [theta + np.pi/4, ref_norm/norm, ref_row[0], ref_col[0], ref_center_row[0], ref_center_col[0]]
np.savez(r'D:\DMD_transform_parameters\params_3', parameters_list=params)
np.savez(r'C:\Users\Light Sheet User\Desktop\dmd_2\params_3', parameters_list=params)