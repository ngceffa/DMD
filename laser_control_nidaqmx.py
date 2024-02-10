import numpy as np
# These are the libraries needed to work with NI DAQ module.
import nidaqmx
import nidaqmx.stream_writers
from scipy.signal import sawtooth, square
from random import shuffle

SAVE_PATH = r'D:\Data'
# ------------------------------------------------------------------------------

# The analogOut class is designed to work with the USB-6.. DAQ card from Ni to
# control the DMD 488 excitation laser.
# There are only 2 analog-output channels that we can use.
# One is a fake-digital: 5V = laser active; 0V inactive;.
# The second controls the power in the active state.
# 1% from the old LabView code is 100mV from the DAQ.

# ------------------------------------------------------------------------------
class analogOut(object):
    
    def __init__(self, device_name='Dev1', channel_name='ao0', digital_channel_name='ao1'):
    
        """ Constructor:
                - device_name = obvious;
                - channel_name = obvious;   
                - digital_channel = NI USB-6211 doe snot have a digital out, so
                    we can use the other analog, 0V for OFF, 5V for ON ("trigger")
        """ 
        self.device_name = device_name
        self.channel_name = channel_name
        self.digital_channel_name = digital_channel_name

        self.power = nidaqmx.Task("power")
        self.trigger = nidaqmx.Task("trigger")

        self.power.ao_channels.add_ao_voltage_chan(device_name + '/' + channel_name)
        self.trigger.ao_channels.add_ao_voltage_chan(device_name + '/' + digital_channel_name)
# ------------------------------------------------------------------------------
    def digital(self, value):
        if value=='off':
            self.trigger.write(0)
            print('dmdpowerOFF')
        elif value=='on':
            self.trigger.write(5)
            print('dmdpowerON')
        return None
    
    def analog(self, value):
        """ For the DMD power control, 1V equals 10% of the LabView code scale.
        """
        assert(value<=5.1 or value >0.), \
                                    '\n max output 10V: you put   %f' %value
        self.power.write(value)

 # ------------------------------------------------------------------------------
    def square(self, frequency = 1, V_max = 1, V_min = 0, num_samples = 10**5):
                    
        """ Generate square wave:
                - frequency = of the signal, [Hz];
                - V_max = obvious;
                - num_samples = sampling of signals;
        """
        assert(V_max <= 5.), '\n max output 5V: you put   %f' %V_max
                    
        self.t = np.linspace(0, 1, int(num_samples/frequency))

        self.signal = (square(2 * np.pi * self.t) + 1) * (V_max - V_min) / 2. + V_min

        self.power.timing.cfg_samp_clk_timing(num_samples,\
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                    samps_per_chan= num_samples)
        self.power.write(self.signal)
        self.power.control(nidaqmx.constants.TaskMode.TASK_COMMIT)
        self.digital("on")
        self.power.start()

    def sine(self, frequency = 1, V_max = 1, V_min = 0, num_samples = 10**5):
                    
        """ Generate sin wave:
                - frequency = of the signal, [Hz];
                - V_max = obvious;
                - V_min = obvious;
                - num_samples = sampling of signals;
        """
        assert(V_max <= 5.1), '\n max output 10V: you put   %f' %V_max
        assert(V_min >= 0.), '\n max output 10V: you put %f' %V_min
                    
        self.t = np.linspace(0, 1, int(num_samples/frequency))
        
        self.signal = (np.sin(np.pi*self.t/.5)+1) * (V_max-V_min)/2. + V_min
        
        self.power.timing.cfg_samp_clk_timing(num_samples,\
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                    samps_per_chan= num_samples)
        self.power.write(self.signal)
        self.power.control(nidaqmx.constants.TaskMode.TASK_COMMIT)
        self.digital("on")
        self.power.start()

    def sawtooth(self, frequency = 1, V_max = 1, V_min=0, num_samples = 10**5):
                    
        """ Generate sin wave:
                - frequency = of the signal, [Hz];
                - V_max = obvious;
                - num_samples = sampling of signals;
        """
        assert(V_max <= 5.), '\n max output 5V: you put   %f' %V_max
                    
        self.t = np.linspace(0, 1, int(num_samples/frequency))

        self.signal = (sawtooth(2 * np.pi * self.t) + 1) * (V_max - V_min) / 2. + V_min

        self.power.timing.cfg_samp_clk_timing(num_samples,\
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                    samps_per_chan= num_samples)
        self.power.write(self.signal)
        self.power.control(nidaqmx.constants.TaskMode.TASK_COMMIT)
        self.digital("on")
        self.power.start()
    
    def triangle(self, frequency = 1, V_max = 1, V_min=0, num_samples = 10**5):
                    
        """ Generate sin wave:
                - frequency = of the signal, [Hz];
                - V_max = obvious;
                - num_samples = sampling of signals;
        """
        assert(V_max <= 5.), '\n max output 5V: you put   %f' %V_max
                    
        self.t = np.linspace(0, 1, int(num_samples/frequency))

        self.signal = (sawtooth(2 * np.pi * self.t, 0.5) + 1) * (V_max - V_min) / 2. + V_min
        
        self.power.timing.cfg_samp_clk_timing(num_samples,\
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                    samps_per_chan= num_samples)
        self.power.write(self.signal)
        self.power.control(nidaqmx.constants.TaskMode.TASK_COMMIT)
        self.digital("on")
        self.power.start()
        
    def ramp_plus_delay(self, frequency = 1, duty=0.5, V_max = 1, V_min = 0, num_samples = 10**5):
                    
        """ Generate sin wave:
                - frequency = of the signal, [Hz];
                - V_max = obvious;
                - V_min = obvious;
                - num_samples = sampling of signals;
        """
        assert(V_max <= 10.), '\n max output 10V: you put   %f' %V_max
        assert(V_min >= 0.), '\n max output 10V: you put %f' %V_min
                    
        self.t = np.linspace(0, 1, int(num_samples/frequency))
        end = int(len(self.t) * duty)
        print(end)
        t_saw = np.linspace(0, 1, end)
        self.signal = np.zeros((len(self.t)))
        self.signal[:end] = (sawtooth(2 * np.pi * t_saw) + 1) * (V_max - V_min) / 2. + V_min
        
        self.power.timing.cfg_samp_clk_timing(num_samples,\
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                    samps_per_chan= num_samples)
        self.power.write(self.signal)
        self.power.control(nidaqmx.constants.TaskMode.TASK_COMMIT)
        self.digital("on")
        self.power.start()

# ------------------------------------------------------------------------------
        
    def randomised_building_blocks(self, V_max = 1, V_min = 0, num_samples = 10**5):
        # 5s per phase
        # 1- raising phase
        # 2- constant phase (MAX)
        # 3- decreasing phase
        # V_min is included but always supposed to be 0 for the protocol
        # having 4 blocks:
        # A- square (immediate max)
        # B- max after 4s
        # C-  max after 3s
        # D- max after 2s

        samples_per_sec = 10000
        block_duration = 20 # sec

        time = np.linspace(0, 1, samples_per_sec * block_duration) # 10k samples/sec (20s tota)
                                            # more than enough to get smooth signal compared to LS sampling
        signal_A = np.ones((len(time))) * V_min
        signal_B = np.ones((len(time))) * V_min
        signal_C = np.ones((len(time))) * V_min
        signal_D = np.ones((len(time))) * V_min

        signal_A[:samples_per_sec * 15] = V_max

        signal_B[:samples_per_sec * 4] = np.linspace(V_min, V_max, samples_per_sec * 4)
        signal_B[samples_per_sec * 4 : samples_per_sec * 11] = V_max
        signal_B[samples_per_sec * 11 : samples_per_sec * 15] = np.linspace(V_max, V_min, samples_per_sec * 4)

        signal_C[:samples_per_sec * 3] = np.linspace(V_min, V_max, samples_per_sec * 3)
        signal_C[samples_per_sec * 3 : samples_per_sec * 12] = V_max
        signal_C[samples_per_sec * 12 : samples_per_sec * 15] = np.linspace(V_max, V_min, samples_per_sec * 3)

        signal_D[:samples_per_sec * 2] = np.linspace(V_min, V_max, samples_per_sec * 2)
        signal_D[samples_per_sec * 2 : samples_per_sec * 13] = V_max
        signal_D[samples_per_sec * 13 : samples_per_sec * 15] = np.linspace(V_max, V_min, samples_per_sec * 2)

        full_list = [1, 2, 3, 4, 
                     1, 2, 3, 4, 
                     1, 2, 3, 4, 
                     1, 2, 3, 4] # four repetitions ofeach stimulus
        shuffle(full_list)
        np.savetxt(SAVE_PATH + "/protocol.txt", full_list)

        total_signal = []
        for i in full_list:
            if i == 1:
                total_signal.append(signal_A)
            elif i == 2:
                total_signal.append(signal_B)
            elif i == 3:
                total_signal.append(signal_C)
            elif i == 4:
                total_signal.append(signal_D)
        total_signal = np.asarray(total_signal).flatten()
        
        self.power.timing.cfg_samp_clk_timing(samples_per_sec,\
                    sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan= len(total_signal))
        self.power.write(total_signal)
        self.power.control(nidaqmx.constants.TaskMode.TASK_COMMIT)
        self.digital("on")
        self.power.start()

# ------------------------------------------------------------------------------
    def stop(self):
        """ It stops the movement.
        """
        self.trigger.stop()
        
        self.trigger.close()
        self.trigger = nidaqmx.Task(self.digital_channel_name)
        self.trigger.ao_channels.add_ao_voltage_chan(\
                            self.device_name+'/'+self.digital_channel_name)
        self.trigger.write(0)
        self.power.stop()
        
        self.power.close()
        self.power = nidaqmx.Task(self.channel_name)
        self.power.ao_channels.add_ao_voltage_chan(\
                            self.device_name+'/'+self.channel_name)
        self.power.write(0)
# ------------------------------------------------------------------------------
    def close(self):
        """ It closes the task, i.e. the galvo object stops working at all.
        """
        self.stop()
        self.power.close()
        self.trigger.close()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------





















