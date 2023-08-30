import numpy as np
import matplotlib.pyplot as plt
from ALP4 import *
# import ALP4
import time
import imageio
import glob
import os
import math_utils as mt
import DMD_utilities as dmu
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error
import re
import pickle
import napari as nap

XP_FOLDER = r'D:\Data\Orb3b_ppk_72F11_ 57C10__Chronos-mVenus_Chronos-mVenus_Chronos-mVenus_rGeco-iRFP'
TARGET_FOLDER = XP_FOLDER + r'\3_488_20230824_143022'
PARAMETERS_FOLDER = r'C:\\Users\\Light Sheet User\\Desktop\\DMD_parameters.pickled'

class dmd():

    def __init__(self, xp_folder=XP_FOLDER, target_folder=TARGET_FOLDER, general_parameters_folder=PARAMETERS_FOLDER):
        """
        - general_parameters are dmd sizes and location of folders for the affine trasform parameters
        """
        self.DMD = ALP4(version='4.3', libDir='C:/Program Files/ALP-4.3/ALP-4.3 API')
        self.DMD.Initialize()

        self.xp_folder = xp_folder # parent folder of current experiment
        self.target_folder = target_folder # the volume showing what to target with the DMD
        try:
            os.mkdir(self.xp_folder + '/excitation_patterns')
        except:
            print('DMD patterns folder already exists.')

        self.parameters = pickle.load(open(general_parameters_folder, 'rb'))
        self.ref_norm, self.norm, self.rotation_matrix, self.center_x, self.center_y, self.ref_center_x, self.ref_center_y = \
        np.load(self.parameters['params_folder'] + '/params_2.npz', allow_pickle=True)['parameters_list']
        
        # Keep a dark image as DEFAULT, so that laser is directed away from sample (for our current config)
        self.dark = np.ones([self.DMD.nSizeY, self.DMD.nSizeX])*(2**8-1) # nSizeY is number of rows
        self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
        self.DMD.SeqControl(2104, 2106) # from the API, these two numbers display a constant image
        self.DMD.SetTiming()
        self.DMD.SeqPut(imgData = self.dark)
        self.DMD.Run(loop = True)

        self.rows = self.DMD.nSizeY # defined just for convenience
        self.cols = self.DMD.nSizeX # defined just for convenience
        # Given DMD positioning, zeros pixels redirect light to the sample, ones (8-bit max) are dark.
        # Can change here, as well as the dark image in case the roles are reversed / the system geomtry changes:
        self.ON = 0
        self.OFF = 2**8 - 1
        # Define a default mask for calibration: few points, in non-symmetric pattern
        # -------------------------------------------------------------------------------------------------------
        # Define some points eyeballing their location. They must be NOT symmetric and at distinguishible heights.
        delta = 3 # points size
        self.calib_points = [[self.rows//2 + 120, self.cols//2 + 120],
                             [self.rows//2, self.cols//2],
                             [self.rows//2 + 180, self.cols//2 - 100],
                             [self.rows//2 - 170, self.cols//2],
                             [self.rows//2 - 0, self.cols//2 - 120]
                            ]
        # this is unnecessary... check and delete
        # ordered points for the calibration routine
        # self.calib_points_ordered = [[self.rows//2 - 170, self.cols//2],
        #                              [self.rows//2, self.cols//2],
        #                              [self.rows//2 - 0, self.cols//2 - 120],
        #                              [self.rows//2 + 180, self.cols//2 - 100],
        #                              [self.rows//2 + 120, self.cols//2 + 120]
        #                              ]
        self.calibration_mask = np.ones((self.rows, self.cols), dtype=np.uint8) * self.OFF
        self.calibration_mask[self.calib_points[0][0] - delta : self.calib_points[0][0] + delta, 
                              self.calib_points[0][1] - delta : self.calib_points[0][1] + delta] = self.ON
        self.calibration_mask[self.calib_points[1][0] - delta : self.calib_points[1][0]  + delta, 
                              self.calib_points[1][1] - delta : self.calib_points[1][1]  + delta] = self.ON
        self.calibration_mask[self.calib_points[2][0] - delta : self.calib_points[2][0]  + delta, 
                              self.calib_points[2][1] - delta : self.calib_points[2][1]  + delta] = self.ON
        self.calibration_mask[self.calib_points[3][0] - delta : self.calib_points[3][0]  + delta, 
                              self.calib_points[3][1] - delta : self.calib_points[3][1]  + delta] = self.ON
        self.calibration_mask[self.calib_points[4][0] - delta : self.calib_points[4][0]  + delta, 
                              self.calib_points[4][1] - delta : self.calib_points[4][1]  + delta] = self.ON
        # -------------------------------------------------------------------------------------------------------
        # here are some quantities used for calibration
        self.calib_points = np.asarray(self.calib_points)
        # self.calib_points_ordered = np.asarray(self.calib_points)
        self.calibration_center_row = self.calib_points[:, 0].mean()
        self.calibration_center_col = self.calib_points[:, 1].mean()
        self.centered_ref = self.calib_points[:, 0] - self.calibration_center_row
        # centered theoretical maxs
        self.centered_ref = np.stack((self.centered_ref, self.calib_points[:,1] - self.calibration_center_col), axis=1)
        # rescaled (normalised) theoretical mask
        self.ref_norm = np.linalg.norm(self.centered_ref)

    def project(self, pattern):
        """Project an image.
        Can accept path to image or directly a numpy array."""
        if type(pattern) == str:
            mask = imageio.imread(pattern)
            mask = np.asarray(mask, dtype=np.uint8)
        else: mask = pattern
        self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
        self.DMD.SeqControl(2104, 2106)
        self.DMD.SetTiming()
        self.DMD.SeqPut(imgData=mask)
        self.DMD.Run(loop=True)

    def idle(self):
        """Project nothing (dark image)"""
        self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
        self.DMD.SeqControl(2104, 2106)
        self.DMD.SetTiming()
        self.DMD.SeqPut(imgData = self.dark)
        self.DMD.Run(loop=True)

    def select_ROIs(self):
        """self.target_folder and self.xp_folder can be edited before this, accessing
        different acquisition without restarting the DMD"""
        target_stack = self.target_folder + '/SPC00_TM00001_ANG000_CM0_CHN00_PH0.stack' # this is a default naming convention
        stack = mt.open_binary_stack(target_stack, size_rows=2304, size_cols=2304, file_type=np.uint16)
        self.target_proj = np.amax(stack, axis=0) # z-proj
        # select shapes
        self.viewer = nap.Viewer()
        self.viewer.add_image(self.target_proj, name='Z-proj image for selecting stimulation areas')
        self.viewer.layers[0].contrast_limits = [np.amin(self.target_proj), np.amax(self.target_proj)]
        nap.run()
        return None
    
    # def import_rois(self):
    #     # make sure user selected two stilation patterns
    #     if ('Shapes' not in self.viewer.layers or 'Shapes [1]' not in self.viewer.layers):
    #         print('Not enought shapes. Program ends.\n')
    #         exit()
    #     odour_1 = np.asarray(self.viewer.layers['Shapes'].data, dtype=np.uint16)
    #     odour_2 = np.asarray(self.viewer.layers['Shapes [1]'].data, dtype=np.uint16)
    #     mask_1 = np.zeros((self.target_proj.shape), dtype=np.uint16)
    #     mask_2 = np.zeros((self.target_proj.shape), dtype=np.uint16)
    #     for shape in odour_1:
    #         mask_1[shape[0][0]:shape[2][0], shape[0][1]:shape[2][1]] = 1
    #     imageio.imwrite(self.folder +'\\' + 'test_mask_1.png', mask_1)
    #     for shape in odour_2:
    #         mask_2[shape[0][0]:shape[2][0], shape[0][1]:shape[2][1]] = 1
    #     imageio.imwrite(self.folder + '\\' + 'test_mask_2.png', mask_2)
    
    def import_N_rois(self, exp=0):
        """Call after select_ROIs.
        i-th excitation pattern is saved as mask_i.png
        in a subfolder called as the exp parameter"""
        self.rois = []
        if len(self.viewer.layers) == 1:
            print('No ROIs drawn. Nothing is done.\n')
            return 0
        else:
            print('Found %i ROI(s)' % (len(self.viewer.layers) - 1))
        save_dir = self.xp_folder + '/excitation_patterns/' + str(exp)
        os.mkdir(save_dir)
        for i in range(1, len(self.viewer.layers)):
            mask = np.zeros((self.target_proj.shape), dtype=np.uint16)
            for shape in self.viewer.layers[i].data:
                mask[int(shape[0][0]):int(shape[2][0]), int(shape[0][1]):int(shape[2][1])] = 1
            self.rois.append(mask)
            imageio.imwrite(save_dir + '\mask_' + str(i) + '.png', mask)


    # def calibration(self, image_path):
    #     #if camera can be accessed, this can be automatised.
    #     # The process should be:
    #     # - load the calibration pattern on DMD
    #     # - take a full chip image
    #     # - run what is already here to identify maxima and affine transform.
    #     top_left = [1000, 500] # because of stray light from DMD. Depends on alignment.
    #     calib = imageio.imread(image_path)[top_left[0]:1900,top_left[1]:1580] # cut around the points.
    #     calib = calib[:, ::-1] # DMD pattern is mirrored by the optical system
    #     tops = [] # maxima (there are 5 hardcoded maxima in self.calib_points)
    #     for i in range(5):
    #         tops.append(np.unravel_index(np.argmax(calib), (calib.shape)))
    #         calib = mt.cookiecutter(calib, tops[i], 10) #10 pixels should beenoguh to get rid of the spot
    #     tops = np.asarray(tops)
    #     for i in range(tops.shape[0]):
    #         tops[i, 0] += top_left[0]
    #         tops[i, 1] += top_left[1]
    #     self.center_row = tops[:, 0].mean()
    #     self.center_col = tops[:, 1].mean()
    #     self.centered = tops[:, 0] - self.center_row
    #     #centered experimental maxs
    #     self.centered = np.stack((self.centered, tops[:, 1] - self.center_col), axis=1)
        
    #     self.norm = np.linalg.norm(self.centered)
    #     self.scaled = self.ref_norm/self.norm * self.centered
    #     self.rotated = mt.rotate(self.scaled, np.pi/4)

    #     # center the mask points

    #     points_found = np.ones((self.rows, self.cols), dtype=np.uint8)
    #     points_found = points_found * 2**8 - 1
    #     for i in range(len(self.rotated)):
    #         points_found[int(self.rotated[i][0]), int(self.rotated[i][1])] = 0

    #     fig = plt.figure('1')
    #     fig.add_subplot(211)
    #     plt.imshow(points_found, cmap='gray')
    #     fig.add_subplot(212)
    #     plt.imshow(self.calibration_mask, cmap='Blues')
    #     plt.show()

    #     return None

    def sequence_of_single_images(self, images: list=[], durations: list=[], repetitions: int=0):
        """
        - images: list of numpy matrices uint8, with 255 or 0 (off/on) pixels
        - durations: list of display time for each image (in seconds)
        - repetitions: how many times to repeat the full list
        """
        assert len(images) == len(durations), \
                            "Images and durations are lists with different lengths"
        for _ in range(repetitions):
            for j in range(len(images)):
                self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
                self.DMD.SeqControl(2104, 2106)
                self.DMD.SetTiming()
                self.DMD.SeqPut(imgData = images[j])
                self.DMD.Run(loop=True)
                time.sleep(durations[j])
        self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
        self.DMD.SeqControl(2104, 2106)
        self.DMD.SetTiming()
        self.DMD.SeqPut(imgData = self.dark)
        self.DMD.Run(loop=True)

    def sequence_with_shift_update(self, images, durations, repetitions,
                                   start, xp_folder, camera, 
                                   cols=1024, rows=512):
        # work in progress,
        # finish in case the labview cannot compensate for drift
        target = np.amax(open_binary_stack(start, cols, rows), axis=0)
        for i in range(repetitions):
            for j in range(len(images)):
                self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
                self.DMD.SeqControl(2104, 2106)
                self.DMD.SetTiming()
                self.DMD.SeqPut(imgData = images[j])
                self.DMD.Run(loop=True)

                if j%2 == 0:
                    list_of_files = filter( os.path.isfile, glob.glob(xp_folder + '*' + camera + '*') )
                    list_of_files = sorted( list_of_files, key=os.path.getmtime)
                    moving = np.amax(open_binary_stack(list_of_files[-2]), axis=0)
                    correlation = mt.spatial_Xcorr_2D(moving, target)
                    row_peak, col_peak = mt.xcorr_peak_2D(correlation)
                    if row_peak > 2 or col_peak > 2:
                        pass

                time.sleep(durations[j])
        self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
        self.DMD.SeqControl(2104, 2106)
        self.DMD.SetTiming()
        self.DMD.SeqPut(imgData = self.dark)
        self.DMD.Run(loop=True)

    def close(self):
        self.idle()
        self.DMD.Halt()
        print('\nDMD halted.')
        self.DMD.FreeSeq()
        self.DMD.Free()
        print('DMD closed.\n')

    def find_affine(self, image, show=False):
        """Calibrate the affine transform to connect DMD and image spaces"""
        calib = imageio.imread(image)
        # remove light from DMD border, which is very bright and would not allow proper maxima identifiation
        calib[:700, :] *= 0
        calib[:, :500] *= 0
        calib[1800:, :] *= 0
        calib[:, 1900:] *= 0
        calib = calib[:, ::-1] # DMD pattern is mirrored by the optical system
        tops = [] # maxima
        for i in range(5):
            tops.append(np.unravel_index(np.argmax(calib), (calib.shape)))
            calib = mt.cookiecutter(calib, tops[i], 10) # 10 pixels should beenoguh to get rid of the spot around each max
        tops = np.asarray(tops)
        # center the point clouds around their center of mass
        camera, camera_center_x, camera_center_y = mt.center(tops) # experimental ponts
        dmd, dmd_center_x, dmd_center_y = mt.center(self.calib_points) # Theoretical points
        # Calculate norm of reference
        ref_norm = np.linalg.norm(dmd)
        norm = np.linalg.norm(camera)
        camera = ref_norm/norm * camera
        # System has a 45deg rotation. We can put this is and then refine the actual angle with a fit
        rotation_matrix = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                    [np.sin(np.pi/4), np.cos(np.pi/4)]])
        rotated_vectors = (rotation_matrix@camera.T).T
        if show:
            plt.figure('Scaled and rotated 45')
            plt.scatter(dmd[:, 0], dmd[:, 1], s=100, label='DMD')
            plt.scatter(rotated_vectors[:, 0], rotated_vectors[:, 1], alpha=.4, s=400, label='Camera')
            plt.legend()
            plt.grid(alpha=.2)
            plt.show()
        rotated_vectors = np.sort(rotated_vectors.astype(np.int16), axis=0)
        dmd = np.sort(dmd, axis=0)
        # The fit to reffine the (small) rotation angle:
        res = opt.minimize(mt.letter_rotation, 0, args=(dmd, rotated_vectors), method="L-BFGS-B")
        theta = res.x[0]
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        rotated_vectors = (rotation_matrix@rotated_vectors.T).T
        if show:
            plt.figure('Small angle rotation')
            plt.scatter(dmd[:, 0], dmd[:, 1], s=100, label='DMD')
            plt.scatter(rotated_vectors[:, 0], rotated_vectors[:, 1], s=400, alpha=.4,  label='Camera')
            plt.show()

        rotation_matrix = np.array([[np.cos(np.pi/4 + theta), -np.sin(np.pi/4 + theta)],
                                    [np.sin(np.pi/4 + theta), np.cos(np.pi/4 + theta)]])
        
        parameters_list = [ref_norm, norm, rotation_matrix, camera_center_x, camera_center_y, dmd_center_x, dmd_center_y]

        parameters_list = np.asarray(parameters_list, dtype=object)
        np.savez(self.parameters['params_folder'] + '/params_2.npz', parameters_list=parameters_list)
        # by default the affine transform parameters
        self.ref_norm, norm, self.rotation_matrix, self.center_x, self.center_y, self.ref_center_x, self.ref_center_y = \
                    np.load(self.parameters['params_folder'] + '/params_2.npz', allow_pickle=True)['parameters_list']
    
    def apply_affine(self, image, output_name):
        """
        Image is already a numpy matrix
        """

        self.ref_norm, self.norm, self.rotation_matrix, self.center_x, self.center_y, self.ref_center_x, self.ref_center_y = \
                    np.load(r'C:\Users\Light Sheet User\Desktop\test_DMD\params_2.npz', allow_pickle=True)['parameters_list']

        image = image[:, ::-1] # mirror
        mask_vectors = mt.find_all_max(image)
        # find the top-left corner of the ROI (usually we crop a lot to increase acquisition speed)
        with (open(self.xp_folder + '/ch0.xml')) as file:
            for line in file:
                dims = re.match('^<info camera_roi', line)
                if dims != None:
                    replaced = line[18:].replace('_',',')
                    roi_values = np.asarray(re.split(',', replaced)[:4]).\
                        astype(np.uint16)
        offsets = np.zeros((1, 2))
        print('offsets', offsets)
        offsets[0, 0] = roi_values[2] #- 1
        offsets[0, 1] = roi_values[0] #- 1
        mask_vectors = np.add(mask_vectors, offsets)

        # Remove distance to center in Camera
        mask_vectors[:,0] = mask_vectors[:,0] - self.center_x[0]
                        # - np.repeat(center_x[0], mask_vectors.shape[0])
        mask_vectors[:,1] = mask_vectors[:,1] - self.center_y[0]
                        # - np.repeat(center_y[0], mask_vectors.shape[0])
        # Scale vectors
        mask_vectors = self.ref_norm/self.norm * mask_vectors
        # Rotate vectors
        rotated_vectors = (self.rotation_matrix@mask_vectors.T).T
        # Move vectors to center of DMD
        rotated_vectors[:,0] = rotated_vectors[:,0] + self.ref_center_x[0]
        # np.repeat(self.ref_center_x[0],
        #                                        mask_vectors.shape[0])
        rotated_vectors[:,1] = rotated_vectors[:,1] + self.ref_center_y[0]
        # np.repeat(self.ref_center_y[0],
        #                                                     mask_vectors.shape[0])

        # Create the DMD mask
        DMD_mask = 255 * np.ones((self.rows, self.cols))

        for i in range(rotated_vectors.shape[0]):
            DMD_mask[int(rotated_vectors[i, 0]), int(rotated_vectors[i, 1])] = 0
        DMD_mask_save = DMD_mask.astype(np.uint8)



        # for ii in range(0, rotated_vectors.shape[0]):
        #     if rotated_vectors[ii, 0] > -1 \
        #         and rotated_vectors[ii, 0] < self.rows-1:
        #         if rotated_vectors[ii, 1] > -1 and rotated_vectors[ii, 1] < self.cols-1:
        #             DMD_mask[int(round(rotated_vectors[ii, 0])),
        #                     int(round(rotated_vectors[ii, 1]))] = 0
        # DMD_mask_save = DMD_mask.astype(np.uint8)



        imageio.imwrite(output_name, DMD_mask_save)
        return DMD_mask

    def camera_to_DMD(self, target, output_name):
        ref_norm,\
        norm,\
        rotation_matrix,\
        center_x,\
        center_y,\
        ref_center_x,\
        ref_center_y = np.load(self.parameters['params_folder'] + '\\params.npz', allow_pickle=True)['parameters_list']

        # Find coordinates of landmarks and store in Nx2 vector
        mask_vectors = mt.find_all_max(target)

        # find the top-left corner of the ROI
        with (open(self.xp_folder + '/ch0.xml')) as file:
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

        rotated_vectors = rotated_vectors[:, ::-1]
        # # Apply flip
        # if not flip_horizontal(rotated_vectors, dmd_size_x):
        #     print('Error: something went wrong when applying the horizontal flip.')

        # Create the DMD mask
        DMD_mask = 255 * np.ones((self.rows, self.cols))

        for ii in range(0, rotated_vectors.shape[0]):
            if rotated_vectors[ii, 0] > -1 \
                and rotated_vectors[ii, 0] < dmd_size_x-1:
                if rotated_vectors[ii, 1] > -1 \
                    and rotated_vectors[ii, 1] < self.cols-1:
                    DMD_mask[int(round(rotated_vectors[ii, 1])),
                            int(round(rotated_vectors[ii, 0]))] = 0
        DMD_mask_save = DMD_mask.astype(np.uint8)
        imageio.imwrite(
            output_name,
            DMD_mask_save
            )
        return DMD_mask

def sequence_of_single_images(images, durations, repetitions):
    dark = np.ones([1200, 1920])*(2**8-1)
    DMD = ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
    DMD.Initialize()
    #laser on if mirrored
    time.sleep(2)
    
    for i in range(repetitions):
        for j in range(len(images)):
            DMD.SeqAlloc(nbImg=1, bitDepth=1)
            DMD.SeqControl(2104, 2106) 
            DMD.SetTiming()
            DMD.SeqPut(imgData = images[j])
            DMD.Run(loop=True)
            time.sleep(durations[j])

    DMD.Halt()
    time.sleep(2)
    #laser off if mirrored
    print('halted')
    DMD.FreeSeq()
    DMD.Free()

    return None

def DMD_protocol(images, durations, fixed_sampling=False):
    sequence = []
    assert len(durations)==len(images), 'images and durations have different lengths. ABORT.'
    DMD = ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
    DMD.Initialize()
    dark = np.ones([DMD.nSizeY, DMD.nSizeX])*(2**8-1)
    total_time = np.sum(durations)
    period = int(total_time * 1000000 + 4000000) #pluys a final dark phase for 4s
    print(period)
    if fixed_sampling==True:
        pass
    
    #multiples of 2?
    for i in range(len(images)):
        for _ in range(0, durations[i], 2):
            sequence.append(images[i].ravel())
    sequence.append(dark.ravel()), sequence.append(dark.ravel())
    images_number = len(sequence)

    imgSeq  = np.concatenate([sequence])
    DMD.SeqAlloc(nbImg = images_number, bitDepth = 1)
    DMD.SeqPut(imgData = imgSeq)
    DMD.SeqControl(2104, 2106)
    DMD.SetTiming(illuminationTime = period)
    # DMD.SetTiming(pictureTime = period)
    DMD.Run(loop=True)
    print('period', period)
    time.sleep(period//1000000 + 1)
    # Run the sequence in an infinite loop
    DMD.Halt()
    print('halted')
    DMD.FreeSeq()
    print('freed')
    DMD.Free()

    return None

# # Binary amplitude image (0 or 1)
# bitDepth = 1
# imgBlack = np.zeros([1200, 1920])
# imgWhite = np.ones([1200, 1920])*(2**8-1)
# dark = np.ones([1200, 1920])*(2**8-1)

# # imgWhite[600:700, 600:700] = 0
# # imgBlack[200:1000, 200:1500] = 1 * (2**8-1)
# roi = np.ones([1200, 1920])*(2**8-1)
# roi[750:800, 1300:1700] = 0
# dark = np.ones([1200, 1920])*(2**8-1)
# folder = r'C:\\Users\\Light Sheet User\\Documents\\SiMView Configuration\\DMD Images'
# images = [
#     imageio.imread(folder+'/one_Piece.png'),
#     dark,
#     imageio.imread(folder+'/Letter_R.png'),
#     imageio.imread(folder+'/000__odour_sum.png'),
#           ]

# DMD = ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
# DMD.Initialize()
# time.sleep(6)
# DMD.Halt()
# print('halted')
# # DMD.FreeSeq()
# DMD.Free()

# # a =  sequence_of_single_images(images, [1, 1, .5, 1], 4)

# bitDepth = 1
# imgBlack = np.zeros([1200, 1920])
# imgWhite = np.ones([1200, 1920])*(2**8-1)
# dark = np.ones([1200, 1920])*(2**8-1)

# imgWhite[600:700, 600:700] = 0
# imgBlack[200:1000, 200:1500] = 1 * (2**8-1)
# roi = np.ones([1200, 1920])*(2**8-1)
# roi[750:800, 1300:1700] = 0

# images =[imgBlack, roi, imgWhite, roi, dark]

# # a =  sequence_of_single_images(images, [2, 2, 2, 2, 5], 3)



# # # Load the Vialux .dll
# # DMD = ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
# # # Initialize the device
# # DMD.Initialize()

# # DMD.setT
# # # Binary amplitude image (0 or 1)
# # bitDepth = 1
# # imgBlack = np.zeros([DMD.nSizeY, DMD.nSizeX])
# # imgWhite = np.ones([DMD.nSizeY, DMD.nSizeX])*(2**8-1)
# # dark = np.ones([DMD.nSizeY, DMD.nSizeX])*(2**8-1)

# # imgWhite[600:700, 600:700] = 0
# # imgBlack[200:1000, 200:1500] = 1 * (2**8-1)
# # roi = np.ones([DMD.nSizeY, DMD.nSizeX])*(2**8-1)
# # roi[750:800, 1300:1700] = 0

# # imgSeq = np.concatenate([imgBlack.ravel(),imgWhite.ravel()])



# # single = dark.ravel()
# # DMD.SeqAlloc(nbImg = 1, bitDepth= bitDepth)
# # DMD.SeqPut(imgData = single)
# # DMD.SeqControl(2104, 2106) 
# # # DMD.SetTiming(illuminationTime = 700000)
# # DMD.Run(loop=True)
# # time.sleep(3)
# # DMD.SeqAlloc(nbImg = 1, bitDepth= bitDepth)
# # single = roi.ravel()
# # DMD.SeqPut(imgData = single)
# # DMD.SeqControl(2104, 2106) 
# # # DMD.SetTiming(illuminationTime = 700000)
# # DMD.Run(loop=True)
# # time.sleep(5)
# # DMD.Halt()
# # print('halted')
# # DMD.FreeSeq()
# # DMD.Free()


# # # DMD = ALP4(version = '4.3', libDir = 'C:/Program Files/ALP-4.3/ALP-4.3 API')
# # # DMD.Initialize()
# # single = imgWhite
# # # imgSeq  = np.concatenate([roi.ravel(), single.ravel()])
# # imgSeq  = np.concatenate([roi.ravel(), single.ravel(), roi.ravel(), roi.ravel(), single.ravel(), roi.ravel(), single.ravel(), roi.ravel(), single.ravel(), roi.ravel(), single.ravel(), roi.ravel()])
# # DMD.SeqAlloc(nbImg = 12, bitDepth = bitDepth)
# # DMD.SeqPut(imgData = imgSeq)
# # DMD.SeqControl(2104, 2106)
# # # DMD.SetTiming(illuminationTime = 500000)
# # DMD.SetTiming(pictureTime = 3000000)
# # DMD.Run(loop=True)
# # time.sleep(60)
# # # Run the sequence in an infinite loop
# # DMD.Halt()
# # print('halted')
# # DMD.FreeSeq()
# # print('freed')
# # DMD.Free()


# # # # Run the sequence in an infinite loop
# # # DMD.Run()
# # # time.sleep(10)
# # # # Stop the sequence display
# # # DMD.Halt()
# # # # Free the sequence from the onboard memory
# # # DMD.FreeSeq()

# # # De-allocate the device
# # # DMD.Free()


# imgfolder = r'C:\Users\Light Sheet User\Documents\SiMView Configuration\DMD Images'
# o1 = imageio.imread(imgfolder + '\\000__odour_1.png') # odour 1
# o2 = imageio.imread(imgfolder + '\\000__odour_2.png')
# sum = imageio.imread(imgfolder + '\\000__odour_sum.png')

# o1_reps = []
# o2_reps = []
# sum_reps = []
# alternate_reps = []

# o1_time = []
# o2_time = []
# sum_time = []
# alternate_time = []

# for i in range(40):
#     o1_reps.append(o1), o1_reps.append(dark)
#     o1_time.append(5), o1_time.append(55)

#     o2_reps.append(o2), o2_reps.append(dark)
#     o2_time.append(5), o2_time.append(55)

#     sum_reps.append(sum), o2_reps.append(dark)
#     sum_time.append(5), o2_time.append(55)

#     alternate_reps.append(o1), alternate_reps.append(dark), alternate_reps.append(o2), alternate_reps.append(dark)
#     alternate_time.append(5), alternate_time.append(55), alternate_time.append(5), alternate_time.append(55)


# fiveHz = []
# fiveHz_time = []

# for i in range(50): # for 10s
#     fiveHz.append(o2), fiveHz.append(dark)
#     fiveHz_time.append(.2), fiveHz_time.append(.2)
# fiveHz.append(dark)
# fiveHz_time.append(40)

# tenHz = []
# tenHz_time = []

# for i in range(50):
#     tenHz.append(o2), tenHz.append(dark)
#     tenHz_time.append(.1), tenHz_time.append(.1)
# tenHz.append(dark)
# tenHz_time.append(40)

# oneHz = []
# oneHz_time = []

# for i in range(10):
#     oneHz.append(o1), oneHz.append(dark)
#     oneHz_time.append(.5), oneHz_time.append(.5)
# oneHz.append(dark)
# oneHz_time.append(40)


# longshort = []
# longshort_time = []

# for i in range(5):
#     longshort.append(o1), longshort.append(dark)
#     longshort_time.append(10), longshort_time.append(30)

# for i in range(5):
#     longshort.append(o1), longshort.append(dark)
#     longshort_time.append(3), longshort_time.append(30)

# longshort.append(dark)
# longshort_time.append(40)

# import csv
# tracefolder = r'D:\Data\freq\traces'
# # opening the CSV file
# values = []
# with open(tracefolder + '\\2_1.csv', mode ='r') as file:
#   csvFile = csv.reader(file)
#   i = 0
#   for lines in csvFile:
#         if i ==0: i+=1
#         else: values.append(float(lines[1]))


# images = [o1, dark]
# imagtimes = [10, 30]