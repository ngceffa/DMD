import numpy as np
import matplotlib.pyplot as plt
from ALP4 import *
import time
import imageio
import os
import math_utils as mt
import scipy.optimize as opt
import re
import pickle
import napari as nap
import laser_control_nidaqmx as daq
import importlib
importlib.reload(daq)

XP_FOLDER = r'D:\Data\Orb3b_ppk_72F11_ 57C10__Chronos-mVenus_Chronos-mVenus_Chronos-mVenus_rGeco-iRFP'
TARGET_FOLDER = XP_FOLDER + r'\3_488_20230824_143022'
GENERAL_PARAMETERS = 'D:\\DMD_transform_parameters\\DMD_parameters.pickled'
PARAMETERS_FILENAME = r'/params.npz' #paramters

class dmd():

    def __init__(self, xp_folder=XP_FOLDER, target_folder=TARGET_FOLDER, general_parameters_folder=GENERAL_PARAMETERS):
        """
        - general_parameters are dmd sizes and location of folders for the affine trasform parameters
        - daq is hardwired, so using only default parameters passing nothing in the constructor:
            not the best way, but it's cleaner to read.
        """
        # In the future, version and directory might change.
        self.DMD = ALP4(version='4.3', libDir='C:/Program Files/ALP-4.3/ALP-4.3 API')
        self.DMD.Initialize()

        self.power = daq.analogOut()

        self.xp_folder = xp_folder # parent folder of current experiment
        self.target_folder = target_folder # the volume showing what to target with the DMD
        try:
            os.mkdir(self.xp_folder + '/excitation_patterns')
        except:
            print('DMD patterns folder already exists.')

        self.parameters = pickle.load(open(general_parameters_folder, 'rb'))
        self.ref_norm, self.norm, self.rotation_matrix, self.center_x, self.center_y, self.ref_center_x, self.ref_center_y = \
        np.load(self.parameters['params_folder'] + '/params.npz', allow_pickle=True)['parameters_list']
        
        # Keep a dark image as DEFAULT, so that laser is directed away from sample (for our current config)
        self.dark = np.ones([self.DMD.nSizeY, self.DMD.nSizeX], dtype=np.uint8)*(2**8-1) # nSizeY is number of rows
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

    def project(self, pattern, value=0.1):
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
        self.power.digital("on")
        self.power.analog(value)
    
    def project_sine_power(self, pattern, frequency, V_max, V_min):
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
        self.power.digital("on")
        self.power.sine(frequency, V_max, V_min)
    
    def project_sawtooth_power(self, pattern, frequency, V_max, V_min):
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
        self.power.digital("on")
        self.power.sawtooth(frequency, V_max, V_min)
    
    def project_square_power(self, pattern, frequency, V_max, V_min):
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
        self.power.digital("on")
        self.power.square(frequency, V_max, V_min)
    
    def project_ramp_plus_delay(self, pattern, frequency, duty, V_max, V_min):
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
        self.power.digital("on")
        self.power.ramp_plus_delay(frequency, duty, V_max, V_min)
    
    def project_triangle_power():
        return None
    
    def project_with_continuous_power_signal(self, pattern, signal="sine", frequency = 1, V_max = 1, V_min = 0, duty=.5):
        """Options are sine, sawtooth, and triangle signal for power variation in time."""

        # Pattern display: 
        if type(pattern) == str:
            mask = imageio.imread(pattern)
            mask = np.asarray(mask, dtype=np.uint8)
        else: mask = pattern
        self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
        self.DMD.SeqControl(2104, 2106)
        self.DMD.SetTiming()
        self.DMD.SeqPut(imgData=mask)
        self.DMD.Run(loop=True)

        # Power control:
        if signal == "sine":
            self.power.sine(frequency, V_max, V_min)
        elif signal == "sawtooth":
            self.power.sawtooth(frequency, V_max, V_min)
        elif signal == "triangle":
            self.power.triangle(frequency, V_max, V_min)
        elif signal == "ramp_plus_delay":
            self.power.ramp_plus_delay(frequency, duty, V_max, V_min)
        else:
            "Unknown signal type: choose from sine, sawtooth, or triangle"

    def idle(self):
        """Project nothing (dark image)"""
        self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
        self.DMD.SeqControl(2104, 2106)
        self.DMD.SetTiming()
        self.DMD.SeqPut(imgData = self.dark)
        self.DMD.Run(loop=True)
        self.power.stop()
    
    def sum_patterns(self, patterns: list=[]):
        """Sum a series of patterns and returns a projectable sum."""
        total = np.ones((patterns[0].shape), dtype=np.uint16)
        for pattern in patterns:
            total *= pattern
            total[total[:, :] > 1] = 255
        total = np.asarray(total, dtype=np.uint8)
        return total

    def select_ROIs(self):
        # perhaps add default arguments for image dimensions, so that any dimension could work
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
    
    def import_N_rois(self, exp: str='xp'):
        """Call after select_ROIs.
        i-th excitation pattern is saved as mask_i.png
        in a subfolder called as the exp parameter"""
        self.rois = []
        if len(self.viewer.layers) == 1:
            print('No ROIs drawn. Nothing is done.\n')
            return 0
        else:
            print('Found %i ROI(s)' % (len(self.viewer.layers) - 1))
        self.save_dir = self.xp_folder + '/excitation_patterns/' + str(exp)
        try:
            os.mkdir(self.save_dir)
        except:
            print('Target ROI(s) folder already exists.')
        for i in range(1, len(self.viewer.layers)):
            mask = np.zeros((self.target_proj.shape), dtype=np.uint16)
            for shape in self.viewer.layers[i].data:
                mask[int(shape[0][0]):int(shape[2][0]), int(shape[0][1]):int(shape[2][1])] = 1 

            self.apply_affine(mask, self.save_dir + '\mask_' + str(i) + '.png' )
            self.rois.append(mask)
            # imageio.imwrite(save_dir + '\mask_' + str(i) + '.png', mask)

    def sequence_of_single_images(self, images: list=[], durations: list=[], repetitions: int=0):
        """,c
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
    
    def power_titration(self, image, duration=5, black_duration=5, steps=10, repetitions: int=1, pow=[0.1, 2]):
        """,c
        - images: list of numpy matrices uint8, with 255 or 0 (off/on) pixels
        - durations: list of display time for each image (in seconds)
        - repetitions: how many times to repeat the full list
        """
        powers = np.linspace(pow[0], pow[1], steps)
        print(powers)
        for _ in range(repetitions):
            for j in range(len(powers)):
                print(j)
                self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
                self.DMD.SeqControl(2104, 2106)
                self.DMD.SetTiming()
                self.DMD.SeqPut(imgData = image)
                self.DMD.Run(loop=True)
                self.power.digital("on")
                self.power.analog(j)
                time.sleep(duration)
                self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
                self.DMD.SeqControl(2104, 2106)
                self.DMD.SetTiming()
                self.DMD.SeqPut(imgData = self.dark)
                self.DMD.Run(loop=True)
                self.power.digital("off")
                time.sleep(black_duration)

    def uploaded_sequence(self, images: list=[], durations: list=[], repetitions: int=0):
        """,c
        - images: list of numpy matrices uint8, with 255 or 0 (off/on) pixels
        - durations: list of display time for each image (in seconds)
        - repetitions: how many times to repeat the full list
        """
        assert len(images) == len(durations), \
                            "Images and durations are lists with different lengths"
        
        self.DMD.SeqAlloc(nbImg=len(images), bitDepth=1)
        self.DMD.SeqControl(2104, 2106)
        self.DMD.SetTiming()
        self.DMD.SeqPut(imgData = images[j])
        self.DMD.Run(loop=True)
        time.sleep(30)
        self.DMD.Halt()


        # for _ in range(repetitions):
        #     for j in range(len(images)):
        #         self.DMD.SeqAlloc(nbImg=1, bitDepth=1)
        #         self.DMD.SeqControl(2104, 2106)
        #         self.DMD.SetTiming()
        #         self.DMD.SeqPut(imgData = images[j])
        #         self.DMD.Run(loop=True)
        #         time.sleep(durations[j])
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
        self.power.stop()
        self.power.close()
        print("DND kaser off.\n")

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
            calib = mt.cookiecutter(calib, tops[i], 30) # 30 pixels should beenoguh to get rid of the spot around each max
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
        print(rotation_matrix)
        
        parameters_list = [ref_norm, norm, rotation_matrix, camera_center_x, camera_center_y, dmd_center_x, dmd_center_y]

        parameters_list = np.asarray(parameters_list, dtype=object)
        np.savez(self.parameters['params_folder'] + PARAMETERS_FILENAME, parameters_list=parameters_list)
        # by default the affine transform parameters
        self.ref_norm, norm, self.rotation_matrix, self.center_x, self.center_y, self.ref_center_x, self.ref_center_y = \
                    np.load(self.parameters['params_folder'] + PARAMETERS_FILENAME, allow_pickle=True)['parameters_list']


    def find_affine_2(self, image, show=False):
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
            calib = mt.cookiecutter(calib, tops[i], 30) # 30 pixels should beenoguh to get rid of the spot around each max
        # tops = np.asarray(tops)
        # center the point clouds around their center of mass
        camera, camera_center_x, camera_center_y = mt.center(tops) # experimental ponts
        dmd, dmd_center_x, dmd_center_y = mt.center(self.calib_points) # Theoretical points




        tops = np.asarray(tops[0] - 1152, tops[1] - 1152)
        theory = []
        for point in self.calib_points:
            theory.append(point[0] - self.rows/2, point[1] - self.cols/2)
        theory = np.asarray(theory)
        # System has a 45deg rotation. We can put this is and then refine the actual angle with a fit
        rotation_matrix = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                    [np.sin(np.pi/4), np.cos(np.pi/4)]])
        
        rotated_vectors = (rotation_matrix@theory)

        # tbc
        




        







        # # Calculate norm of reference
        # ref_norm = np.linalg.norm(dmd)
        # norm = np.linalg.norm(camera)
        # camera = ref_norm/norm * camera
        # # System has a 45deg rotation. We can put this is and then refine the actual angle with a fit
        # rotation_matrix = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
        #                             [np.sin(np.pi/4), np.cos(np.pi/4)]])
        # rotated_vectors = (rotation_matrix@camera.T).T
        # if show:
        #     plt.figure('Scaled and rotated 45')
        #     plt.scatter(dmd[:, 0], dmd[:, 1], s=100, label='DMD')
        #     plt.scatter(rotated_vectors[:, 0], rotated_vectors[:, 1], alpha=.4, s=400, label='Camera')
        #     plt.legend()
        #     plt.grid(alpha=.2)
        #     plt.show()
        # rotated_vectors = np.sort(rotated_vectors.astype(np.int16), axis=0)
        # dmd = np.sort(dmd, axis=0)
        # # The fit to reffine the (small) rotation angle:
        # res = opt.minimize(mt.letter_rotation, 0, args=(dmd, rotated_vectors), method="L-BFGS-B")
        # theta = res.x[0]
        # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
        #                             [np.sin(theta), np.cos(theta)]])
        # rotated_vectors = (rotation_matrix@rotated_vectors.T).T
        # if show:
        #     plt.figure('Small angle rotation')
        #     plt.scatter(dmd[:, 0], dmd[:, 1], s=100, label='DMD')
        #     plt.scatter(rotated_vectors[:, 0], rotated_vectors[:, 1], s=400, alpha=.4,  label='Camera')
        #     plt.show()

        # rotation_matrix = np.array([[np.cos(np.pi/4 + theta), -np.sin(np.pi/4 + theta)],
        #                             [np.sin(np.pi/4 + theta), np.cos(np.pi/4 + theta)]])
        # print(rotation_matrix)
        
        # parameters_list = [ref_norm, norm, rotation_matrix, camera_center_x, camera_center_y, dmd_center_x, dmd_center_y]

        # parameters_list = np.asarray(parameters_list, dtype=object)
        # np.savez(self.parameters['params_folder'] + PARAMETERS_FILENAME, parameters_list=parameters_list)
        # # by default the affine transform parameters
        # self.ref_norm, norm, self.rotation_matrix, self.center_x, self.center_y, self.ref_center_x, self.ref_center_y = \
        #             np.load(self.parameters['params_folder'] + PARAMETERS_FILENAME, allow_pickle=True)['parameters_list']
    
    def apply_affine(self, image, output_name):
        """
        Image is already a numpy matrix
        """
        image = image[:, ::-1] # mirror
        mask_vectors = mt.find_all_max(image)
        # find the top-left corner of the ROI (usually we crop a lot to increase acquisition speed)
        with (open(self.target_folder + '/ch0.xml')) as file:
            for line in file:
                dims = re.match('^<info camera_roi', line)
                if dims != None:
                    replaced = line[18:].replace('_',',')
                    roi_values = np.asarray(re.split(',', replaced)[:4]).\
                        astype(np.uint16)
        offsets = np.zeros((1, 2))
        offsets[0, 0] = roi_values[2] #- 1
        offsets[0, 1] = roi_values[0] #- 1
        mask_vectors = np.add(mask_vectors, offsets)

        # Remove distance to center in Camera
        mask_vectors[:,0] = mask_vectors[:,0] - self.center_x[0]
        mask_vectors[:,1] = mask_vectors[:,1] - self.center_y[0]
        # Scale vectors
        mask_vectors = self.ref_norm/self.norm * mask_vectors
        # Rotate vectors
        rotated_vectors = (self.rotation_matrix@mask_vectors.T).T
        # Move vectors to center of DMD
        rotated_vectors[:,0] = rotated_vectors[:,0] + self.ref_center_x[0]
        rotated_vectors[:,1] = rotated_vectors[:,1] + self.ref_center_y[0]

        # Create the DMD mask
        DMD_mask = 255 * np.ones((self.rows, self.cols))

        for i in range(rotated_vectors.shape[0]):
            DMD_mask[int(rotated_vectors[i, 0]), int(rotated_vectors[i, 1])] = 0
        DMD_mask_save = DMD_mask.astype(np.uint8)

        imageio.imwrite(output_name, DMD_mask_save)
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
