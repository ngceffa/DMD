import DMD_utilities as dmd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import napari as nap

sys.path.append('C:\\Users\\Light Sheet User\\Desktop\\dmd_2')
params = 'D:\DMD_transform_parameters\params_3.npz'
theta, scaling, ref_row, ref_col, ref_center_row, ref_center_col = np.load(params, allow_pickle=True)['parameters_list']

# Usually this is the folder with the maker showing the position of the optogeetic activator
xp_folder = r'D:\Data\OR83b_Cronos-mVenus__KCs_GFP-rGECO_\s_2_488_c_20221018_135439'
# the actual volume to be used a target
target_stack = xp_folder + '/SPC00_TM00004_ANG000_CM0_CHN00_PH0.stack'
 # where to save stimulation ROIs location
masks_folder = 'D:\Data\OR83b_Cronos-mVenus__KCs_GFP-rGECO_\\targets'
# seed for naming the masks
xp_name = 's_2_2'
# where the calculated DMD masks are stored
excitations_folder = masks_folder + '\\' + xp_name
if not os.path.exists(excitations_folder):
    os.makedirs(excitations_folder)

stack = dmd.open_binary_stack(
    target_stack,
    size_cols=1024,
    size_rows=512,
    file_type=np.uint16
    )
target_proj = np.amax(stack, axis=0)

viewer = nap.Viewer()
viewer.add_image(
    target_proj,
    name='Select two stimulation areas',
    colormap='magma'
    )
viewer.layers[0].contrast_limits = [
    np.amin(target_proj), np.amax(target_proj)]
nap.run()
proceed = input('"Enter" to proceed --> ') 
# make sure user selected two stilation patterns
if ('Shapes' not in viewer.layers):
    print('You must draw a shaape to proceed.\n')
    exit()

masks = []
for i in range(1, len(viewer.layers)):
    mask = np.zeros((target_proj.shape), dtype=np.uint16)
    for shape in (viewer.layers[i].data):
        mask[int(shape[0][0]):int(shape[2][0]), int(shape[0][1]):int(shape[2][1])] = 1
    masks.append(mask)

# transform the excitation points into DMD-space loadable images


# IN FOR CYCLE!!!

DMD_mask_points = dmd.camera_to_DMD(
    xp_folder,
    params['params_folder'] + '\\params.npz',
    mask,
    params['save_images_folder'] + '\\' + xp_name + '_odour_1.png'
    )