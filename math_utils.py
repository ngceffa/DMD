from dask.base import persist
import numpy as np
# from numba import jit
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack as ft
import tifffile as tif
from scipy.optimize import curve_fit
import re
import dask.array as da
from IPython.display import display, Math
import tifffile as tif
from sklearn.metrics import mean_squared_error
from math import cos, sin, sqrt


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    
def rect(x, b, c):
    """Rectangular signal:
        - x: dominion;
        - b: width.
        - c: centre;
    """
    y = np.zeros((x.shape))
    y[c - b / 2 : c + b / 2] = 1
    return y

def gaus_3D_for_fit(
    x_data,
    offset,
    amplitude, 
    z_0, 
    x_0, 
    y_0, 
    den_z, 
    den_x, 
    den_y):
    # xdata is a vstack of raveled meshgrids directions
    z, x, y = x_data
    return (offset + np.abs(amplitude)
         * np.exp(-1 * np.pi *
        (((z - z_0)**2 / den_z**2)
        + ((x - x_0)**2 / den_x**2)
        + ((y - y_0)**2 / den_y**2))
        )).flatten()

def gaus_3D(
    z,
    x,
    y,
    amplitude, 
    z_0, 
    x_0, 
    y_0, 
    den_z, 
    den_x, 
    den_y):
    z, x, y = np.meshgrid(z, x, y, indexing='ij')
    return (np.abs(amplitude)
         * np.exp(-1 * np.pi *
        (((z - z_0)**2 / den_z**2)
        + ((x - x_0)**2 / den_x**2)
        + ((y - y_0)**2 / den_y**2))
        ))

def FT2(f):
    """ 2D Inverse Fourier Transform, with proper shift
    """
    return (ft.fftshift(ft.fft2(ft.ifftshift(f))))

def FT(f):
    """ 1D Inverse Fourier Transform, with proper shift
    """
    return (ft.fftshift(ft.fft(ft.ifftshift(f))))

def IFT(f):
    """ 1D Inverse Fourier Transform, with proper shift
    """
    return (ft.fftshift(ft.ifft(ft.ifftshift(f))))

def IFT2(f):
    """ 2D Fourier Transform, with proper shift
    """
    return (ft.fftshift(ft.ifft2(ft.ifftshift(f))))

def FT3(f):
    """ 3D Fourier Transform, with proper shift
    """
    return ft.fftshift(ft.fftn(ft.ifftshift(f)))

def IFT3(F):
    """ 3D Inverse Fourier Transform, with proper shift
    """
    return ft.ifftshift(ft.ifftn(ft.fftshift(F)))

def convert_to_16_bit(array):
    """Take the (N-dim) array and return a 16 bit copy.
    """
    array_rescaled = np.zeros((array.shape), dtype=np.uint16)
    old_max, old_min = np.amax(array), np.amin(array)
    amplitude = (2**16 / (old_max - old_min))
    array_rescaled[:, :] = (array[:, :] - old_min) * amplitude
    return array_rescaled

def deconvolve_RL(stack, 
                  psf,
                  iterations: int,
                  tell_steps=False):
    """Simple 3D Lucy-Richarson deconvolution.
    N.B. TO DO: add renormalization.
    N.B.2 It returns a real-valued result, not int.
    """
    o = np.copy(stack).astype(complex)
    # Richardson-Lucy in the for loop
    for k in range (iterations):
        step_0 = stack / (IFT3(FT3(o) * psf))
        step_1 = IFT3(FT3(step_0) * np.conj(psf))
        o *= step_1
        if(tell_steps): print(k)
    return np.real(np.sqrt((o * np.conj(o))))

def gaussian_2D(dim, center=[0, 0], sigma=1):
    """ 2D gaussian, cetered at origin.
        - dim = input extent 
        (assumed square, e.g. for a 1024x1024 image it should be simply 1024, 
         and the extent would go [-512;512])
        - sigma = stdev !!! NO
    """
    x, y = np.meshgrid(np.arange(-dim/2, dim/2, 1), \
                np.arange(-dim/2, dim/2, 1))
    top = (x - center[1])**2 + (y - center[0])**2 # row major convention
    return np.exp(-(top / (2 * sigma)**2))

def gaus_2D(
    rows: int,
    cols: int,
    center=[0, 0],
    den=[1, 1], 
    A=1, 
    offset=0
    ):
    """Two-dimensional Gaussian, on a rectagula domain."""
    x = np.linspace(-(rows/2), rows/2, rows)
    y = np.linspace(-(cols/2), cols/2, cols)
    x, y = np.meshgrid(x, y, indexing='ij')
    top = ((x - center[1]) / den[0])**2 + ((y - center[0]) / den[1])**2
    return A * np.exp(-np.pi * top) + offset

def normalize_0_1(array):
    """ Normalize input (N-dim) array to 0-1 range.
    """
    maximum, minimum = np.amax(array), np.amin(array)
    normalized = np.zeros(array.shape)
    delta = maximum - minimum
    normalized = (array - minimum) / delta
    return normalized

def spatial_Xcorr_2D(f, g):
    """
    Cross-correlation between two 2D functions: (f**g).
    N.B. f can be considered as the moving input, g as the target.
    - inputs are padded to avoid artifacts (this makes it slower)
    """
    M, N = f.shape[0], f.shape[1]
    one, two = np.pad(f,
                      ((int(M/2), int(M/2)),
                       (int(N/2), int(N/2))),
                      mode = 'constant',
                      constant_values=(0,0)),\
               np.pad(g,
                      ((int(M/2), int(M/2)),
                      (int(N/2), int(N/2))),
                      mode = 'constant', 
                      constant_values=(0,0))                  
    ONE, TWO =   FT2(one), FT2(two)
    spatial_cross = ft.ifftshift(ft.ifft2(ft.ifftshift(ONE) \
                  * np.conj(ft.ifftshift(TWO)))) \
                    [int(M/2) :int(M/2+M), int(N/2) : int(N/2+N)]
    return np.real(spatial_cross)

def xcorr_peak_2D(cross):
    """Assuming a 2D cross-correlation shows a peak (global max),
    it returns its position (row, col).
    """
    row_shift, col_shift = np.unravel_index(np.argmax(cross), cross.shape)
    return int(row_shift - cross.shape[0]/2), int(col_shift - cross.shape[1]/2)

def spatial_xcorr_3D(f, g, pad: str = 'n'):
    """
    Cross-correlation between two 2D functions: (f**g).
    N.B. f can be considered as the moving input, g as the target.
    - inputs are not padded to avoid artifacts (this would make it slower)
    - The output is normalized to [0,1)
    """
    Z, M, N = f.shape[0], f.shape[1], f.shape[2]
    if pad == 'y':
        one, two = np.pad(np.copy(f),
                        ((int(Z/2), int(Z/2)),
                        (int(M/2), int(M/2)),
                        (int(N/2), int(N/2))),
                        mode = 'constant',
                        constant_values=(0, 0, 0)),\
                np.pad(np.copy(g),
                        ((int(z/2), int(z/2)),
                        (int(M/2), int(M/2)),
                        (int(N/2), int(N/2))),
                        mode = 'constant', 
                        constant_values=(0, 0, 0))                  
        ONE, TWO =   FT3(one), FT3(two)
    elif pad == 'n':
        ONE, TWO =   FT3(f), FT3(g)
    else: print('pad value not properly defined\n'); return False
    spatial_cross = ft.ifftshift(ft.ifftn(ft.ifftshift(ONE) \
                  * np.conj(ft.ifftshift(TWO))))
                    # [int(Z/2) :int(Z/2+Z),
                    # int(M/2) :int(M/2+M),
                    # int(N/2) : int(N/2+N)]
    spatial_cross = normalize_0_1(spatial_cross)
    return np.real(spatial_cross)

def xcorr_peak_3D(cross):
    """Assuming a 2D cross-correlation shows a peak (global max),
    it returns its position (row, col).
    """
    depth_shift, row_shift, col_shift = \
                                np.unravel_index(np.argmax(cross), cross.shape)
    return int(depth_shift - cross.shape[0]/2), \
           int(row_shift - cross.shape[0]/2), \
           int(col_shift - cross.shape[1]/2)

def shift_image(image, shift):                 
    H, W = image.shape
    shifted = image.copy()
    if shift[0] >= 0:
        shifted[int(shift[0]):, :] = \
                                    image[:int(H - shift[0]), :] # shift up
        # need to cancel the shifted region...
        shifted[:int(shift[0]), :] = np.amin(shifted)
    elif shift[0] < 0:
        shifted[:int(H + shift[0]), :] = \
                                     image[int(-shift[0]):, :] # shift down
        shifted[int(H + shift[0]):, :] = np.amin(shifted)
    if shift[1] > 0:
        # shift right
        shifted[:, int(shift[1]):] = shifted[:, :int(W - shift[1])]
        shifted[:, :int(shift[1])] = np.amin(shifted)
    elif shift[1] < 0:
        # shift right
        shifted[:, :int(W + shift[1])] = shifted[:, int(- shift[1]):]
        shifted[:, int(W + shift[1]):] = np.amin(shifted)
    return shifted
    
def shift_volume(volume, shift):
    D, H, W = volume.shape
    shifted = np.copy(volume) * 0
    if shift[0] >= 0: shifted[:,int(shift[0]):, :] = \
                                    volume[:,:int(H - shift[0]), :] # shift up
    elif shift[0] < 0: shifted[:,:int(H + shift[0]), :] = \
                                     volume[:,int(-shift[0]):, :] # shift down
    if shift[1] > 0:
        # shift right
        shifted[:,:, int(shift[1]):] = shifted[:,:, :int(W - shift[1])]
    elif shift[1] < 0:
        # shift right
        shifted[:, :, :int(W + shift[1])] = shifted[:,:, int(- shift[1]):]

    return shifted

def open_binary_volume_with_hotpixel_correction(name, 
                                                VOLUME_SLICES,
                                                IMAGES_DIMENSION,
                                                hotpixel_value=64000,
                                                format=np.uint16):
    """ It also performs hotpixel correction"""
    with open(name, 'rb') as file:
        raw_array = np.fromfile(file, dtype=format)
    raw_array[raw_array[:] > hotpixel_value] = 0
    volume_array =  np.reshape(raw_array, (VOLUME_SLICES,
                                           IMAGES_DIMENSION,
                                           IMAGES_DIMENSION))
    return volume_array

def files_names_list(total_volumes, seed_0='SPC00_TM', 
                                    seed_1='_ANG000_CM', 
                                    seed_2='_CHN00_PH0'):
    """ Basically used to list acquisition files so that I can parallelize.
    List single entry paradigm: 
                [volume_number (int), views_1 (array), views_2 (array)]
    """
    files_list = []
    j = 0
    for i in range(total_volumes):
        temp_list = [i]
        for k in range(0, 2):
            temp_list.append(seed_0 + f'{j:05}' + seed_1
                            + str(k) + seed_2 + ".stack")
        files_list.append(temp_list)
        j += 1
    return files_list

def files_names_list_with_path(path, total_volumes, seed_0='SPC00_TM', 
                                    seed_1='_ANG000_CM', 
                                    seed_2='_CHN00_PH0'):
    """ Basically used to list acquisition files so that I can parallelize.
    List single entry paradigm: 
                [volume_number (int), views_1 (array), views_2 (array)]
    """
    files_list = []
    j = 0
    for i in range(total_volumes):
        temp_list = []
        for k in range(0, 1):
            temp_list.append(path + seed_0 + f'{j:05}' + seed_1
                            + str(k) + seed_2 + ".stack")
        files_list.append(temp_list)
        j += 1
    return files_list

def open_binary_stack(
    stack_path,
    size_rows=1024,
    size_cols=1024,
    file_type=np.uint16
    ):
    stack_original = np.fromfile(stack_path, dtype=file_type)
    # Determine Z size automatically based on the array size
    size_z = int(stack_original.size / size_rows / size_cols)
    # Reshape the stack based on known dimensions
    stack = np.reshape(stack_original, (size_z, size_rows, size_cols))
    type_max = np.iinfo(stack.dtype).max
    type_min = np.iinfo(stack.dtype).min
    # hotpixels correction
    stack[stack == type_max] = type_min
    return stack

def fit_3D_gaussians(
    stack,
    timepoint=0,
    slices=41,
    num_beads=25,
    box_side=22,
    min_cutoff=1,
    max_cutoff=15,
    verbose='n',
    plot='y',
    plot_name=0
    ):
    stack = stack.compute()
    # volume for fitting a single bead
    box = np.asarray((box_side, box_side, box_side))
    # Setup for the fitting
    z_width, x_width, y_width = [], [], []
    x_val = np.arange(0, box[0], 1)
    y_val = np.arange(0, box[1], 1)
    z_val = np.arange(0, box[2], 1)
    z_grid, x_grid, y_grid = np.meshgrid(
        z_val,
        x_val,
        y_val,
        indexing='ij'
        )
    xdata = np.vstack((z_grid.ravel(), x_grid.ravel(), y_grid.ravel()))
    mean_z, mean_x, mean_y = 0, 0, 0
    beads = 0
    z_chosen = int(slices / 2) #be default, look around the center
    stack_shape = stack.shape

    if verbose == 'y': print('\nNUMBER OF BEADS FOUND:\n')

    while beads < num_beads:

        x_max, y_max = np.unravel_index(
            np.argmax(
                stack[timepoint, z_chosen, :, :]),
                stack[timepoint, z_chosen, :, :].shape)

        # if the box is all contained in the volume
        if (
            x_max - box[1] / 2 > 0
            and x_max + box[1] / 2 < stack_shape[2]
            and y_max - box[2] / 2 > 0
            and y_max + box[2] / 2 < stack_shape[3]):
            substack = stack[
                timepoint,
                int(z_chosen - box[0] / 2):int(z_chosen + box[0] / 2),
                int(x_max - box[1] / 2):int(x_max + box[1] / 2),
                int(y_max - box[2] / 2):int(y_max + box[2] / 2),
            ]
            # first guesses for fit parameters
            bg = 1.
            A_coeff = 1000.
            x_0 = box[1] / 2.
            y_0 = box[2] / 2.
            z_0 = box[0] / 2.
            x_sig = 3.
            y_sig = x_sig
            z_sig = 4.
            p0 = [bg, A_coeff, z_0, x_0, y_0, z_sig, x_sig, y_sig]
            # params = []
            popt, _ = curve_fit(
                gaus_3D_for_fit,
                xdata,
                substack.ravel(), 
                p0)

            stack[
                timepoint,
                int(z_chosen - box[0] / 2):int(z_chosen + box[0] / 2),
                int(x_max - box[1] / 2):int(x_max + box[1] / 2),
                int(y_max - box[2] / 2):int(y_max + box[2] / 2),
            ] = 0
            # avoid failure in the fitting: if any width is larger than
            # the max cutoff, then assume the fit was unsuccessful and discard:
            # same hold for the min_cutoff value
            if(
                np.abs(popt[5]) < max_cutoff
                and np.abs(popt[6]) < max_cutoff
                and np.abs(popt[7]) < max_cutoff
                and np.abs(popt[5]) > min_cutoff
                and np.abs(popt[6]) > min_cutoff
                and np.abs(popt[7]) > min_cutoff
                ):
                beads += 1  
                z_width.append(np.abs(popt[5]))
                x_width.append(np.abs(popt[6]))
                y_width.append(np.abs(popt[7]))
                mean_z += np.abs(popt[5])
                mean_x += np.abs(popt[6])
                mean_y += np.abs(popt[7])
        else:
            stack[
                timepoint,
                int(z_chosen - box[0] / 2):int(z_chosen + box[0] / 2),
                int(x_max - box[1] / 2):int(x_max + box[1] / 2),
                int(y_max - box[2] / 2):int(y_max + box[2] / 2),
            ] = 0
        if verbose == 'y': print(beads, end=', ', flush=True)
        if verbose == 'y': print('\n')
        
    mean_z = np.sum(np.asarray(z_width)) / num_beads
    mean_x = np.sum(np.asarray(x_width)) / num_beads
    mean_y = np.sum(np.asarray(y_width)) / num_beads
        
    if plot == 'y':
        plt.figure(plot_name, (12, 5))
        plt.title(plot_name)
        plt.plot(
            z_width,
            'o-',
            color='mediumorchid',
            lw=2,
            ms=7,
            alpha=.8,
            label='z')
        z_const = mean_z * np.ones((len(z_width)))
        plt.plot(
            z_const, 
            '--', 
            color='indigo', 
            lw=2
            )
        plt.plot(
            x_width,
            's-',
            color='indianred',
            lw=2,
            ms=9,
            alpha=.8,
            label='x'
            )
        plt.xlabel('bead')
        plt.ylabel(r'2 $\pi$ $\sigma^{2}$ [pixels]')
        x_const = mean_x * np.ones((len(x_width)))
        plt.plot(
            x_const, 
            '--', 
            color='red', 
            lw=2)
        plt.plot(
            y_width, 
            '^-', 
            color='dodgerblue', 
            lw=2, 
            ms=9, 
            alpha=.8, 
            label='y'
            )
        y_const = mean_y * np.ones((len(y_width)))
        plt.plot(y_const, '--', color='blue', lw=2)
        # # annotation = r"$B + A \,exp(-\frac{(x - x_{0})^{2}}{b^{2}})$"
        # plt.annotate(annotation, xy=(num_beads/2, np.amax(y_width)/1.1))
        plt.legend()
        plt.show()
    display(Math(r'f(x) = B + A\,e^\left(- \pi \frac{(x-x_{0})^{2}}{b^{2}}\right)'))
    display(Math(r'b = 2 \pi \sigma^{2}'))
    return mean_z, mean_x, mean_y

def extract_metadata(folder):
    """To be used with Light-sheet data, as collected by Isoview 5.
       In the saved folder, a file ch0.xml contains all the necessary metadata. 
       This function extracts few parameters that are useful for analysis:
       - x: number of cols;
       - y: number of rows; [check if not the opposite]
       - z: number of slices per volume;
       - z_step: distance between slices;
       - 
    """

    with (open(folder + '/ch0.xml')) as file:
        for line in file:
            dims = re.match('^<info dimensions', line)
            z_step_line = re.match('^<info z_step', line)
            if dims != None:
                found = line[18:] 
                res = found[:found.index(',')]
            if z_step_line != None:
                found = line[14:]
                z_step = found[:found.index('"')]

    x = res[:res.index('x')]
    res = res[res.index('x')+1:]
    y = res[:res.index('x')]
    z = res[res.index('x')+1:]
    return np.asarray((int(z), int(x), int(y))), float(z_step)

def merge_views(
    front,
    back, 
    method='local_variance',
    sigma=10
    ):
    """ Images are merged.
    """
    if method == 'average':
        merged = front.copy() * 0
        # Super simple average: fast but poor quality
        # merged = front #((front + back / 2))
        for i in range(front.shape[0]):
            merged[i, :, :] = np.real((front[i, :, :] + back[i, :, :])/2)
    elif method == 'original':
        merged = front.copy() * 0
        # Images are merged using the approximation of local variance
        # similar to the one defined  in Preibish et al. 2008
        # 'Mosaicing of Single Plane Illumination Microscopy Images
        # Using Groupwise Registration and Fast Content-Based Image Fusion'
        # shorturl.at/nHMY5
        gaus_1 = gaus_2D(
            merged.shape[1],
            merged.shape[2],
            den=(20, 20))
        Gaus_1 = FT2(gaus_1)
        gaus_2 = gaus_2D(
            merged.shape[1],
            merged.shape[2],
            den=(40, 40))
        Gaus_2 = FT2(gaus_2)
        for i in range(front.shape[0]):
            Front = FT2(front[i, :, :])
            Back = FT2(back[i, :, :])
            front_weight = (IFT2(Front - (Front * Gaus_1)))**2
            front_weight = IFT2((FT2(front_weight) * Gaus_2))
            front_weight = np.sqrt(front_weight * np.conj(front_weight))

            # front_weight = np.abs(np.real(front_weight))
            back_weight = (IFT2(Back - (Back * Gaus_1)))**2
            back_weight = IFT2((FT2(back_weight) * Gaus_2))
            back_weight = np.sqrt(back_weight * np.conj(back_weight))

            # back_weigth = np.abs(np.real(back_weigth))
            tot = front_weight + back_weight
            merged[i, :, :] = np.real((front_weight * front[i, :, :] 
                                     + back_weight * back[i, :, :])\
                                     / tot[:, :])
    elif method == 'local_variance':
        merged = front.copy() * 0
        # Images are merged using the approximation of local variance
        # similar to the one defined  in Preibish et al. 2008
        # 'Mosaicing of Single Plane Illumination Microscopy Images
        # Using Groupwise Registration and Fast Content-Based Image Fusion'
        # shorturl.at/nHMY5
        gaus_1 = gaus_2D(merged.shape[1], merged.shape[2], den=(sigma, sigma))
        Gaus_1 = FT2(gaus_1)
        for i in range(front.shape[0]):
            Front = FT2(front[i, :, :])
            Back = FT2(back[i, :, :])
            front_weight = np.abs(IFT2(Front - (Front * Gaus_1)))**2
            back_weight = np.abs(IFT2(Back - (Back * Gaus_1)))**2
            tot = front_weight + back_weight
            merged[i, :, :] = np.real((front_weight * front[i, :, :] 
                                     + back_weight * back[i, :, :])\
                                     / tot[:, :])
    elif method == 'and_linear':
        merged = front.copy() * 0
        
        gaus_1 = gaus_2D(merged.shape[1], merged.shape[2], den=(sigma, sigma))
        Gaus_1 = FT2(gaus_1)

        linear = np.linspace(0, 1, front.shape[0])
        counter = np.linspace(1, 0, front.shape[0])
        ff, bb = [],[]
        for i in range(front.shape[0]):
            # deriv_front = np.gradient(front[i, :, :])
            # grad_front = np.abs((deriv_front[0]**2 + deriv_front[1]**2))
            # deriv_back = np.gradient(back[i, :, :])
            # grad_back = np.abs((deriv_back[0]**2 + deriv_back[1]**2))
            Front = FT2(front[i, :, :])
            Back = FT2(back[i, :, :])
            front_weight = np.abs(IFT2(Front - (Front * Gaus_1)))**2
            # front_weight *= linear[i]
            # front_weight = np.abs(np.real(front_weight))
            back_weigth = np.abs(IFT2(Back - (Back * Gaus_1)))**2
            # back_weigth *= counter[i]
            # back_weigth = np.abs(np.real(back_weigth))
            ff.append(np.amax(front_weight))
            bb.append(np.amax(back_weigth))
            tot = front_weight + back_weigth
            merged[i, :, :] = np.real((front_weight * front[i, :, :]
                                     + back_weigth * back[i, :, :])\
                                     / tot[:, :])
        plt.figure('ww', figsize=(10,10))
        plt.plot(bb)
        plt.plot(ff)
        plt.show()
    
    elif method =='max':
        merged = front.copy() * 0
        for i in range(front.shape[0]):
            max_f = np.amax(front[i, :, :])
            max_b = np.amax(back[i, :, :])
            if max_f > max_b:
                merged[i, :, :] = front[i, :, :]
            else:
                merged[i, :, :] = back[i, :, :]

    elif method =='std':
        merged = front.copy() * 0
        for i in range(front.shape[0]):
            std_f = np.std(front[i, :, :])
            std_b = np.std(back[i, :, :])
            if std_f > std_b:
                merged[i, :, :] = front[i, :, :]
            else:
                merged[i, :, :] = back[i, :, :]

    elif method == 'wip':
        
        merged = front.copy() * 0

        front_z_avg = np.std(front, axis=(1, 2))
        back_z_avg = np.std(back, axis=(1, 2))
        plt.figure('z', figsize=(10,10))
        plt.plot(front_z_avg / np.amax(front_z_avg), lw=3, label='front')
        plt.plot(back_z_avg / np.amax(back_z_avg), lw=3, label='back')
        plt.legend()
        plt.grid()
        plt.show()
        
        gaus_1 = gaus_2D(merged.shape[1], merged.shape[2], den=(sigma, sigma))
        Gaus_1 = FT2(gaus_1)

        linear = np.linspace(0, 1, front.shape[0])
        counter = np.linspace(1, 0, front.shape[0])
        for i in range(front.shape[0]):
            # deriv_front = np.gradient(front[i, :, :])
            # grad_front = np.abs((deriv_front[0]**2 + deriv_front[1]**2))
            # deriv_back = np.gradient(back[i, :, :])
            # grad_back = np.abs((deriv_back[0]**2 + deriv_back[1]**2))
            Front = FT2(front[i, :, :])
            Back = FT2(back[i, :, :])
            front_weight = np.abs(IFT2(Front - (Front * Gaus_1)))**2
            # front_weight += rescaling(grad_front, front_weight, .1)
            front_weight *= linear[i]
            # front_weight = np.abs(np.real(front_weight))
            back_weigth = np.abs(IFT2(Back - (Back * Gaus_1)))**2
            # back_weigth += rescaling(grad_back, back_weigth, .1)
            back_weigth *= counter[i]
            # back_weigth = np.abs(np.real(back_weigth))
            tot = front_weight + back_weigth
            merged[i, :, :] = np.real((front_weight * front[i, :, :]
                                     + back_weigth * back[i, :, :])\
                                     / tot[:, :])
    # merged = np.abs(np.sqrt(merged * np.conj(merged)))
    return merged

def merge_views_s(
    front,
    back, 
    sigmas,
    method='local_variance',
    ):
    """ Images are merged.
    """
    if method == 'local_variance':
        merged = front.copy() * 0
        for i in range(front.shape[0]):
            gaus_1 = gaus_2D(merged.shape[1], merged.shape[2],
                den=(sigmas[i], sigmas[i]))
            Gaus_1 = FT2(gaus_1)
            Front = FT2(front[i, :, :])
            Back = FT2(back[i, :, :])
            front_weight = np.abs(IFT2(Front - (Front * Gaus_1)))**2
            back_weight = np.abs(IFT2(Back - (Back * Gaus_1)))**2
            tot = front_weight + back_weight
            merged[i, :, :] = np.real((front_weight * front[i, :, :] 
                                     + back_weight * back[i, :, :])\
                                     / tot[:, :])
    return merged

def cookiecutter(image, center, dim):
    """cut a square hole in the image around the center (values inside fo to zero.)"""
    image[center[0]-dim:center[0]+dim, center[1]-dim:center[1]+dim] = 0
    return image

def rotate(points, theta):
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    return (rotation @ points.T).T

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
                mask_vectors[vector_count,0] = ii
                mask_vectors[vector_count,1] = jj
                vector_count += 1
    
    # Return a 2D vector array of the mask in camera space
    return mask_vectors

# def merge_volumes(
#     front,
#     back, 
#     method='local_variance',
#     sigma=10
#     ):
#     """ Images are merged.
#     It assumes dask arrays are passed
#     """
#     if method == 'average':
#         merged = front.copy()
#         # Super simple average: fast but poor quality
#         merged[:, :, :] = ((front[:, :, :] + back[:, :, :] / 2))
#     elif method == 'local_variance':
#         merged = front.copy() * 0
#         # Images are merged using the approximation of local variance
#         # similar to the one defined  in Preibish et al. 2008
#         # 'Mosaicing of Single Plane Illumination Microscopy Images
#         # Using Groupwise Registration and Fast Content-Based Image Fusion'
#         # shorturl.at/nHMY5
#         gaus_1 = gaus_2D(merged.shape[1], merged.shape[2], den=(sigma, sigma))
#         Gaus_1 = FT2(gaus_1)
#         for i in range(front.shape[0]):
#             Front = FT2(front[i, :, :])
#             Back = FT2(back[i, :, :])
#             deriv_front = np.gradient(front[i, :, :])
#             grad_front = np.abs(deriv_front[0]**2 + deriv_front[1]**2)
#             deriv_back = np.gradient(back[i, :, :])
#             grad_back = np.abs(deriv_back[0]**2 + deriv_back[1]**2)
#             front_weight = np.abs(FT2(Front - (Front * Gaus_1)))
#             front_weight += grad_front
#             back_weight = np.abs(FT2(Back - (Back * Gaus_1)))
#             back_weight += grad_back
#             tot = front_weight + back_weight
#             merged[i, :, :] = np.real((front_weight * front[i, :, :] 
#                                      + back_weight * back[i, :, :])\
#                                      / tot[:, :])
#     elif method == 'and_linear':
#         merged = front.copy() * 0
#         # Images are merged using the approximation of local variance
#         # similar to the one defined  in Preibish et al. 2008
#         # 'Mosaicing of Single Plane Illumination Microscopy Images
#         # Using Groupwise Registration and Fast Content-Based Image Fusion'
#         # shorturl.at/nHMY5
#         gaus_1 = gaus_2D(merged.shape[1], merged.shape[2], den=(sigma, sigma))
#         Gaus_1 = FT2(gaus_1)

#         linear = np.linspace(0, 1, front.shape[0])
#         counter = 1 - linear

#         for i in range(front.shape[0]):
#             Front = FT2(front[i, :, :])
#             Back = FT2(back[i, :, :])
#             deriv_front = np.gradient(front[i, :, :])
#             grad_front = np.abs(deriv_front[0]**2 + deriv_front[1]**2)
#             deriv_back = np.gradient(back[i, :, :])
#             grad_back = np.abs(deriv_back[0]**2 + deriv_back[1]**2)
#             front_weight = np.abs(FT2(Front - (Front * Gaus_1)))
#             front_weight += grad_front
#             front_weight *= linear
#             back_weight = np.abs(FT2(Back - (Back * Gaus_1)))
#             back_weight += grad_back
#             back_weight *= counter
#             tot = front_weight + back_weight
#             merged[i, :, :] = np.real((front_weight * front[i, :, :] 
#                                      + back_weight * back[i, :, :])\
#                                      / tot[:, :])
#     elif method == 'original':
#         merged = front.copy() * 0
#         # Images are merged using the approximation of local variance
#         # similar to the one defined  in Preibish et al. 2008
#         # 'Mosaicing of Single Plane Illumination Microscopy Images
#         # Using Groupwise Registration and Fast Content-Based Image Fusion'
#         # shorturl.at/nHMY5
#         gaus_1 = gaus_2D(
#             merged.shape[1],
#             merged.shape[2],
#             den=(20, 20))
#         Gaus_1 = FT2(gaus_1)
#         gaus_2 = gaus_2D(
#             merged.shape[1],
#             merged.shape[2],
#             den=(40, 40))
#         Gaus_2 = FT2(gaus_2)
#         for i in range(front.shape[0]):
#             Front = FT2(front[i, :, :])
#             Back = FT2(back[i, :, :])
#             front_weight = (IFT2(Front - (Front * Gaus_1)))**2
#             front_weight = IFT2((FT2(front_weight) * Gaus_2))
#             front_weight = np.sqrt(front_weight * np.conj(front_weight))

#             # front_weight = np.abs(np.real(front_weight))
#             back_weight = (IFT2(Back - (Back * Gaus_1)))**2
#             back_weight = IFT2((FT2(back_weight) * Gaus_2))
#             back_weight = np.sqrt(back_weight * np.conj(back_weight))

#             # back_weigth = np.abs(np.real(back_weigth))
#             tot = front_weight + back_weight
#             merged[i, :, :] = np.real((front_weight * front[i, :, :] 
#                                      + back_weight * back[i, :, :])\
#                                      / tot[:, :])
#     return merged

def deconvolve_timeseries_RL(stack, 
                  psf,
                  iterations,
                  tell_steps=False):
    """Simple Lucy-Richarson deconvolution.
    N.B. TO DO: add renormalization.
    N.B.2 It returns a real-valued result, not int.
    """
    o = stack.copy().astype(complex)
    # for i in range(stack.shape[0]):
    #     stack[i, :, :, :] += psf

    for k in range(iterations):
        step_0 = stack / ( da.fft.fftn(o, axes=(1, 2, 3)) )

    # for t in range(stack.shape[0]):
        # for k in range(iterations):

    #         step_0 = stack[t, :, :, :] / (IFT3(FT3(o[t, :, :, :]) * psf))
    #         step_1 = IFT3(FT3(step_0) * np.conj(psf))
    #         o[t, :, :, :] *= step_1
    return stack
    # o = np.copy(stack).astype(complex)
    # # Richardson-Lucy in the for loop
    # for k in range (iterations):
    #     for t in range(stack.shape[0]):
    #         step_0 = stack[t, :, :, :] / (IFT3(FT3(o[t, :, :, :]) * psf))
    #         step_1 = IFT3(FT3(step_0[t, :, :, :]) * np.conj(psf))
    #         o[t, :, :, :] *= step_1[t, :, :, :]
    #     if(tell_steps): print(k)
    # return np.real(np.sqrt((o * np.conj(o))))

if __name__ == '__main__':
    #test code here
    pass
