from venv import create
import tifffile as tif
import numpy as np
import re
import imageio
import pickle
import matplotlib.pyplot as plt
from math import cos, sin, sqrt
from sklearn.metrics import mean_squared_error
import scipy.optimize as opt
import math_utils as matut

# Takes a reference <ref> of (nn) vectors and compare it to another set <cam>
# rotated by an angle <theta>. Returns the RMSE between the two sets of
# vectors. This is used to optimize the angle <theta> which minize the RMSE
# between the two models
def letter_rotation(theta, ref, cam):
    # Number of vectors composing the reference model
    nn = ref.shape[0]
    
    # Initialize an array for the rotated <cam> model
    rotated_cam = np.zeros((nn,2))
    
    # Rotation matrix
    rotation_matrix = np.array([[cos(theta), -sin(theta)],
                                [sin(theta), cos(theta)]])
    
    # Apply rotation (there might be a better algebraic way of computing
    # the dot-product for each vector...)
    for ii in range(0, nn):
        rotated_cam[ii,:] = np.dot(rotation_matrix, cam[ii,:])
    
    # Calculate the mean squared error from sklearn.metrics
    MSE = mean_squared_error(ref, rotated_cam)
    
    # Calculate the square root of the  MSE
    # (smoother function, better for optimization)
    RMSE = sqrt(MSE)
    
    # Return the RMSE
    return RMSE

def open_binary_stack(
    stack_path,
    size_cols=1024,
    size_rows=1024,
    file_type=np.uint16
    ):
    with open(stack_path, 'rb') as file:
        stack_original = np.fromfile(file, dtype=file_type)
    # Determine Z size automatically based on the array size
    size_z = int(stack_original.size / size_cols / size_rows)
    # Reshape the stack based on known dimensions
    stack = np.reshape(stack_original, (size_z, size_rows, size_cols))
    type_max = np.iinfo(stack.dtype).max
    type_min = np.iinfo(stack.dtype).min
    # hotpixels correction
    stack[stack == type_max] = type_min
    return stack

def flip_horizontal(vectors, grid_size_x):
    # Number of vectors
    number_of_vectors = vectors.shape[0]
    
    # Apply the flip in place!
    for ii in range(number_of_vectors):
        vectors[ii, 0] = abs(vectors[ii, 0] - grid_size_x)
    
    # Return true if successful (no error trapping)
    return True

def center(points):
    # Calculate center of reference R
    ref_center_x = points[:, 0].mean()
    ref_center_y = points[:, 1].mean()
    print(ref_center_x, ref_center_y)
    ref_center_x = np.repeat(ref_center_x, points.shape[0])
    ref_center_y = np.repeat(ref_center_y, points.shape[0])

    # Center reference R from DMD to zero
    centered_points = points[:,0] - ref_center_x
    centered_points = np.stack((centered_points,
                            points[:,1] - ref_center_y),
                            axis=1)
    
    return centered_points, ref_center_x, ref_center_y

def tag(tag: str = 'name', input: str or Number = 'value'):
    start = '<' + tag + '>'
    end = '</' + tag + '>\n'
    return start + str(input) + end

def script(
    flashes=20,
    TM_start = 100,
    TM_step = 15,
    duration = .5,
    mask_name_seed = 'shifted_target_',
    script_name = 'script.dmdsc'
    ):
    """ Generate excitation protocol as a script for Labview
    It assumes the images are .png
    """

    TM_interval = np.ones(flashes)
    durations = duration * np.ones((flashes))
    script = open(script_name, 'w')   
    
    head = '<Cluster>\n'
    head += '<Name>DMD Script Config</Name>\n'
    head += '<NumElts>1</NumElts>\n'
    head += '<Array>\n'
    head += '<Name>[DMD script steps]</Name>\n'
    head += '<Dimsize>' + str(flashes) + '</Dimsize>\n'
    script.write(head)

    for i in range(flashes):
        flash = '<Cluster>\n'
        flash += tag('Name', 'DMD settings for 1 TM')
        flash += '<NumElts>10</NumElts>\n'
        flash += '<String>\n'
        flash += tag('Name', 'PNG name')
        # here is where the images are updated
        flash += tag('Val', mask_name_seed + str(i) + '.png')
        flash += '</String>\n'
        flash += '<I32>\n'
        flash += tag('Name', 'TM index')
        flash += tag('Val', int(TM_start + i * TM_step))
        flash += '</I32>\n'
        flash += '<SGL>\n'
        flash += tag('Name', 'Pre wait (s)') 
        flash += tag('Val','0.10000') 
        flash += '</SGL>\n'
        flash += '<SGL>\n'
        flash += tag('Name', 'Pulse (s)') 
        flash += tag('Val', float(durations[i]))
        flash += '</SGL>\n'
        flash += '<I32>\n'
        flash += tag('Name', '# triggers')
        flash += tag('Val', 1) # hardcoded for now
        flash += '</I32>\n'
        flash += '<SGL>\n'
        flash += tag('Name', 'Period (s)') 
        flash += tag('Val', durations[i]) # not really sure why
        flash += '</SGL>\n'
        flash += '<Cluster>\n'
        flash += tag('Name', 'Final Offset (pix)') 
        flash += tag('NumElts', 2) # again, it probably means something
        flash += '<I32>\n'
        flash += tag('Name', 'X')
        flash += tag('Val', int(0)) # (>.<) no discernable meaning
        flash += '</I32>\n'
        flash += '<I32>\n'
        flash += tag('Name', 'Y')
        flash += tag('Val', int(0)) # (>.<) no discernable meaning
        flash += '</I32>\n'
        flash += '</Cluster>\n'
        flash += '<Boolean>\n'
        flash += tag('Name', 'Motion?')
        flash += tag('Val', int(0))
        flash += '</Boolean>\n'
        flash += '<EW>\n'
        flash += tag('Name', 'Offset space')
        flash += tag('Choice', 'DMD')
        flash += tag('Choice', 'Camera 1')
        flash += tag('Choice', 'Camera 2')
        flash += tag('Val', 0)
        flash += '</EW>\n'
        flash += '<SGL>\n'
        flash += tag('Name', 'Power Multiplier')
        flash += tag('Val', 1.00000)
        flash += '</SGL>\n'
        flash += '</Cluster>\n'
        
        script.write(flash)
    
    tail = '</Array>\n'
    tail += '</Cluster>\n'

    script.write(tail)
    script.close()
    
    return None

def script_randomized(
    flashes=20,
    TM_start = 100,
    TM_step = 15,
    duration = .5,
    mask_name_seed = 'shifted_target_',
    script_name = 'script.dmdsc'
    ):
    """ Generate excitation protocol as a script for Labview
    It assumes the images are .png
    """

    TM_interval = np.ones(flashes)
    durations = duration * np.ones((flashes))
    script = open(script_name, 'w')   
    
    head = '<Cluster>\n'
    head += '<Name>DMD Script Config</Name>\n'
    head += '<NumElts>1</NumElts>\n'
    head += '<Array>\n'
    head += '<Name>[DMD script steps]</Name>\n'
    head += '<Dimsize>' + str(flashes) + '</Dimsize>\n'
    script.write(head)

    for i in range(flashes):
        flash = '<Cluster>\n'
        flash += tag('Name', 'DMD settings for 1 TM')
        flash += '<NumElts>10</NumElts>\n'
        flash += '<String>\n'
        flash += tag('Name', 'PNG name')
        # here is where the images are updated
        flash += tag('Val', mask_name_seed + str(i) + '.png')
        flash += '</String>\n'
        flash += '<I32>\n'
        flash += tag('Name', 'TM index')
        flash += tag('Val', int(TM_start + i * TM_step))
        flash += '</I32>\n'
        flash += '<SGL>\n'
        flash += tag('Name', 'Pre wait (s)') 
        flash += tag('Val','0.10000') 
        flash += '</SGL>\n'
        flash += '<SGL>\n'
        flash += tag('Name', 'Pulse (s)') 
        flash += tag('Val', float(durations[i]))
        flash += '</SGL>\n'
        flash += '<I32>\n'
        flash += tag('Name', '# triggers')
        flash += tag('Val', 1) # hardcoded for now
        flash += '</I32>\n'
        flash += '<SGL>\n'
        flash += tag('Name', 'Period (s)') 
        flash += tag('Val', durations[i]) # not really sure why
        flash += '</SGL>\n'
        flash += '<Cluster>\n'
        flash += tag('Name', 'Final Offset (pix)') 
        flash += tag('NumElts', 2) # again, it probably means something
        flash += '<I32>\n'
        flash += tag('Name', 'X')
        flash += tag('Val', int(0)) # (>.<) no discernable meaning
        flash += '</I32>\n'
        flash += '<I32>\n'
        flash += tag('Name', 'Y')
        flash += tag('Val', int(0)) # (>.<) no discernable meaning
        flash += '</I32>\n'
        flash += '</Cluster>\n'
        flash += '<Boolean>\n'
        flash += tag('Name', 'Motion?')
        flash += tag('Val', int(0))
        flash += '</Boolean>\n'
        flash += '<EW>\n'
        flash += tag('Name', 'Offset space')
        flash += tag('Choice', 'DMD')
        flash += tag('Choice', 'Camera 1')
        flash += tag('Choice', 'Camera 2')
        flash += tag('Val', 0)
        flash += '</EW>\n'
        flash += '<SGL>\n'
        flash += tag('Name', 'Power Multiplier')
        flash += tag('Val', 1.00000)
        flash += '</SGL>\n'
        flash += '</Cluster>\n'
        
        script.write(flash)
    
    tail = '</Array>\n'
    tail += '</Cluster>\n'

    script.write(tail)
    script.close()
    
    return None

def find_all_max(mask):
    # Find the size of the mask (limited to 2304 x 2304 currently)
    (size_x, size_y) = mask.shape
    
    # Determine the maximum value in the mask
    mask_value = np.amax(mask) 
    # Initialize the output 2D vector list 
    vector_count = 0
    mask_vectors = np.zeros((np.count_nonzero(mask == mask_value), 2))
    
    # Find, and store all vectors corresponding to a maximum in the mask
    for ii in range(0, size_x):
        for jj in range(0, size_y):
            if mask[ii, jj] == mask_value:
                # Notice the reverse coordinates here for consistancy with
                # how Python treats the arrays
                mask_vectors[vector_count, 0] = jj
                mask_vectors[vector_count, 1] = ii
                vector_count += 1
    
    # Return a 2D vector array of the mask in camera space
    return mask_vectors

def find_all_max_2(mask):
    # Find the size of the mask (limited to 2304 x 2304 currently)
    (size_x, size_y) = mask.shape
    
    # Determine the maximum value in the mask
    mask_value = np.amax(mask) 
    # Initialize the output 2D vector list 
    vector_count = 0
    mask_vectors = np.zeros((np.count_nonzero(mask == mask_value), 2))
    
    # Find, and store all vectors corresponding to a maximum in the mask
    for ii in range(0, size_x):
        for jj in range(0, size_y):
            if mask[ii, jj] == mask_value:
                # Notice the reverse coordinates here for consistancy with
                # how Python treats the arrays
                mask_vectors[vector_count, 0] = ii
                mask_vectors[vector_count, 1] = jj
                vector_count += 1
    
    # Return a 2D vector array of the mask in camera space
    return mask_vectors


def camera_to_DMD(xp_folder, params_list, target, output_name):
    # arrays are organized in this arbitrary way when computing the transform.
    # ref_norm,\
    # norm,\
    # rotation_matrix,\
    # center_x,\
    # center_y,\
    # ref_center_x,\
    # ref_center_y = np.load(folder + '/' + params_file,
    #     allow_pickle=True)
    ref_norm,\
    norm,\
    rotation_matrix,\
    center_x,\
    center_y,\
    ref_center_x,\
    ref_center_y = np.load(params_list, allow_pickle=True)['parameters_list']
    dmd_size_x = 1920
    dmd_size_y = 1200
    # Find coordinates of landmarks and store in Nx2 vector
    mask_vectors = find_all_max(target)

    # find the top-left corner of the ROI
    with (open(xp_folder + '/ch0.xml')) as file:
        for line in file:
            dims = re.match('^<info camera_roi', line)
            if dims != None:
                replaced = line[18:].replace('_',',')
                roi_values = np.asarray(re.split(',', replaced)[:4]).\
                    astype(np.uint16)
    offsets = np.zeros((1, 2))
    offsets[0, 0] = roi_values[2] #- 1
    offsets[0, 1] = roi_values[0] #- 1

    print(offsets)

    # Apply ROI if necessary (by default, offsets is a zero-vector)

    mask_vectors = np.add(mask_vectors, offsets)

    # Remove distance to center in Camera
    mask_vectors[:,0] = mask_vectors[:,0] \
                    - np.repeat(center_x[0], mask_vectors.shape[0])
    mask_vectors[:,1] = mask_vectors[:,1] \
                    - np.repeat(center_y[0], mask_vectors.shape[0])

    # Scale vectors
    mask_vectors = ref_norm/norm * mask_vectors

    # Rotate vectors
    rotated_vectors = (rotation_matrix@mask_vectors.T).T

    # Move vectors to center of DMD
    rotated_vectors[:,0] = rotated_vectors[:,0] + np.repeat(ref_center_x[0],
                                                        mask_vectors.shape[0])
    rotated_vectors[:,1] = rotated_vectors[:,1] + np.repeat(ref_center_y[0],
                                                        mask_vectors.shape[0])

    # Apply flip
    if not flip_horizontal(rotated_vectors, dmd_size_x):
        print('Error: something went wrong when applying the horizontal flip.')

    # Create the DMD mask
    DMD_mask = 255 * np.ones((dmd_size_y, dmd_size_x))

    for ii in range(0, rotated_vectors.shape[0]):
        if rotated_vectors[ii, 0] > -1 \
            and rotated_vectors[ii, 0] < dmd_size_x-1:
            if rotated_vectors[ii, 1] > -1 \
                and rotated_vectors[ii, 1] < dmd_size_y-1:
                DMD_mask[int(round(rotated_vectors[ii, 1])),
                        int(round(rotated_vectors[ii, 0]))] = 0
    DMD_mask_save = DMD_mask.astype(np.uint8)
    imageio.imwrite(
        output_name,
        DMD_mask_save
        )
    return DMD_mask

def gaus_2D(
    rows: int,
    cols: int,
    center=[0, 0],
    den=[1, 1], 
    A=1, 
    offset=0
    ):
    """Two-dimensional Gaussian, on a rectagula domain."""
    x = np.linspace(0, rows, rows)
    y = np.linspace(0, cols, cols)
    x, y = np.meshgrid(x, y, indexing='ij')
    top = ((x - center[1]) / den[0])**2 + ((y - center[0]) / den[1])**2
    return A * np.exp(-np.pi * top) + offset

def camera_to_DMD_point(params_list, center, sigma_x, sigma_y, output_name='gaus.tiff'):
    ref_norm,\
    norm,\
    rotation_matrix,\
    center_x,\
    center_y,\
    ref_center_x,\
    ref_center_y = np.load(params_list, allow_pickle=True)['parameters_list']
    dmd_size_x = 1920
    dmd_size_y = 1200
    # Find coordinates of landmarks and store in Nx2 vector
    mask_vectors = np.asarray((center, center))
    

    # Remove distance to center in Camera
    mask_vectors[:,0] = mask_vectors[:,0] \
                    - np.repeat(center_x[0], mask_vectors.shape[0])
    mask_vectors[:,1] = mask_vectors[:,1] \
                    - np.repeat(center_y[0], mask_vectors.shape[0])

    # Scale vectors
    mask_vectors = ref_norm/norm * mask_vectors

    rescaled_sigma_x = ref_norm/norm * sigma_x
    rescaled_sigma_y = ref_norm/norm * sigma_y

    # Rotate vectors
    rotated_vectors = (rotation_matrix@mask_vectors.T).T

    # Move vectors to center of DMD
    rotated_vectors[:,0] = rotated_vectors[:,0] + np.repeat(ref_center_x[0],
                                                        mask_vectors.shape[0])
    rotated_vectors[:,1] = rotated_vectors[:,1] + np.repeat(ref_center_y[0],
                                                        mask_vectors.shape[0])

    # Apply flip
    if not flip_horizontal(rotated_vectors, dmd_size_x):
        print('Error: something went wrong when applying the horizontal flip.')

    DMD_mask_save = gaus_2D(1200, 1920, [rotated_vectors[0, 0], rotated_vectors[0, 1]], [rescaled_sigma_x, rescaled_sigma_y])

    # DMD_mask_save = mask.astype(np.uint8)

    imageio.imwrite(
        output_name,
        DMD_mask_save
        )
    return None

def sample_random_in_ellipse(corners, sets, points_per_set):
    """From an elliptical selection within Napari,
    it will extract N sets of M ('points_per_set') points.
    The points are randomly selected within the ROI.
    
    Returns matrix [2, N, M] of coordinates. 
    First entry of axis 0 is the row, the second is the column.
    """
    
    points = int(points_per_set * sets)
    print("%d points per set, %d sets will be created"\
          %(int(points_per_set), sets))
    
    center_row = int(corners[0, 0] + (corners[2, 0] - corners[0, 0]) / 2)
    center_col = int(corners[0, 1] + (corners[1, 1] - corners[0, 1]) / 2)
    radius_vertical = int((corners[2, 0] - corners[0, 0]) / 2)
    radius_horizontal = int((corners[1, 1] - corners[0, 1]) / 2)
    rows = np.arange(int(corners[0, 0]), int(corners[2, 0]), 1).astype(np.uint16)
    selected_rows = np.random.choice(rows, size=points)
    selected_cols = np.zeros((selected_rows.shape))
    
    rows = np.arange(-radius_vertical, radius_vertical, 1)
    selected_rows = np.random.choice(rows, size=points)
    for i in range(len(selected_rows)):
        y_max = radius_horizontal / radius_vertical * \
            np.sqrt(((radius_vertical)**2 - (selected_rows[i])**2))
        if y_max != 0:
            y_min = - y_max
            interval = np.arange(y_min, y_max, 1)
            selected_cols[i] = np.random.choice(interval)
        else: selected_cols[i] = 0
    selected_rows += center_row
    selected_cols = (selected_cols + center_col).astype(np.uint16)
    
    reshaped_rows = np.resize(selected_rows, (sets, points_per_set))
    reshaped_cols = np.resize(selected_cols, (sets, points_per_set))
    
    matrix = np.stack((reshaped_rows, reshaped_cols), axis=0)
    
    return matrix

def camera_to_DMD_gaus(xp_folder, params_list, target, output_name, compensation_image='gaus.tiff'):

    ref_norm, norm, rotation_matrix, center_x, center_y, ref_center_x, ref_center_y = \
        np.load(params_list, allow_pickle=True)['parameters_list']
    dmd_size_x = 1920
    dmd_size_y = 1200
    # Find coordinates of landmarks and store in Nx2 vector
    mask_vectors = find_all_max(target)

    # find the top-left corner of the ROI
    with (open(xp_folder + '/ch0.xml')) as file:
        for line in file:
            dims = re.match('^<info camera_roi', line)
            if dims != None:
                replaced = line[18:].replace('_',',')
                roi_values = np.asarray(re.split(',', replaced)[:4]).\
                    astype(np.uint16)
    offsets = np.zeros((1, 2))
    offsets[0, 0] = roi_values[2] #- 1
    offsets[0, 1] = roi_values[0] #- 1

    # Apply ROI if necessary (by default, offsets is a zero-vector)
    mask_vectors = np.add(mask_vectors, offsets)

    # Remove distance to center in Camera
    mask_vectors[:,0] = mask_vectors[:,0] \
                    - np.repeat(center_x[0], mask_vectors.shape[0])
    mask_vectors[:,1] = mask_vectors[:,1] \
                    - np.repeat(center_y[0], mask_vectors.shape[0])

    # Scale vectors
    mask_vectors = ref_norm/norm * mask_vectors

    # Rotate vectors
    rotated_vectors = (rotation_matrix@mask_vectors.T).T

    # Move vectors to center of DMD
    rotated_vectors[:,0] = rotated_vectors[:,0] + np.repeat(ref_center_x[0],
                                                        mask_vectors.shape[0])
    rotated_vectors[:,1] = rotated_vectors[:,1] + np.repeat(ref_center_y[0],
                                                        mask_vectors.shape[0])

    # Apply flip
    if not flip_horizontal(rotated_vectors, dmd_size_x):
        print('Error: something went wrong when applying the horizontal flip.')

    # Create the DMD mask
    DMD_mask = 255 * np.ones((dmd_size_y, dmd_size_x))

    for ii in range(0, rotated_vectors.shape[0]):
        if rotated_vectors[ii, 0] > -1 \
            and rotated_vectors[ii, 0] < dmd_size_x-1:
            if rotated_vectors[ii, 1] > -1 \
                and rotated_vectors[ii, 1] < dmd_size_y-1:
                DMD_mask[int(round(rotated_vectors[ii, 1])),
                        int(round(rotated_vectors[ii, 0]))] = 0
    

    DMD_mask = ((255 - DMD_mask) * (imageio.imread(compensation_image)))
    DMD_mask[DMD_mask[:, :] == 0] = 255

    # DMD_mask = 255 * DMD_mask / (np.amax(DMD_mask) - np.amin(DMD_mask))

    DMD_mask = DMD_mask.astype(np.uint8)

    DMD_mask_save = DMD_mask.astype(np.uint8)
    imageio.imwrite(
        output_name,
        DMD_mask_save
        )
    return DMD_mask

def create_target_for_affine_transform():
    rows = 1200; cols = 1920
    cr = rows // 2; cc = cols // 2 # center row and col
    points = np.asarray(((cr - 200, cc + 100), (cr - 100, cc + 400),
                        (cr, cc + 50), (cr + 200, cc + 500),
                        (cr + 250, cc + 100)))
    mask = np.ones((rows, cols), dtype=np.uint16)
    for point in points:
        mask[point[0] - 2:point[0] + 3, point[1] - 2:point[1] + 3] = 0
    np.save('affine_target_points', points)
    imageio.imsave('R_new.png', mask)
    return mask

def find_affine_2(camera_letter_R, dmd_letter_R):

    flip_horizontal(dmd_letter_R, 1920)
    camera_letter_R, camera_center_x, camera_center_y = center(camera_letter_R)
    dmd_letter_R, dmd_center_x, dmd_center_y = center(dmd_letter_R)
    # Calculate norm of reference R
    ref_norm = np.linalg.norm(dmd_letter_R)
    norm = np.linalg.norm(camera_letter_R)
    camera_letter_R = ref_norm/norm * camera_letter_R
    res = opt.minimize(letter_rotation,0,
                              args=(dmd_letter_R, camera_letter_R),
                              method="L-BFGS-B")
    theta = res.x[0]
    rotation_matrix = np.array([[cos(theta), -sin(theta)],
                                [sin(theta), cos(theta)]])
    parameters_list = [
        ref_norm,
        norm,
        rotation_matrix,
        camera_center_x,
        camera_center_y,
        dmd_center_x,
        dmd_center_y
        ]
    paramters_list = np.asarray(parameters_list, dtype=object)
    np.savez(r'D:\DMD_transform_parameters\params_2', parameters_list=paramters_list)
    np.savez(r'C:\Users\Light Sheet User\Desktop\test_DMD\params_2', parameters_list=paramters_list)

    return None                

def center(points):
    # Calculate center of reference R
    ref_center_x = points[:, 0].mean()
    ref_center_y = points[:, 1].mean()
    ref_center_x = np.repeat(ref_center_x, points.shape[0])
    ref_center_y = np.repeat(ref_center_y, points.shape[0])

    # Center reference R from DMD to zero
    centered_points = points[:,0] - ref_center_x
    centered_points = np.stack((centered_points,
                            points[:,1] - ref_center_y),
                            axis=1)
    return centered_points, ref_center_x, ref_center_y