import numpy as np
# These are the libraries needed to work with NI DAQ module.
import nidaqmx
import nidaqmx.stream_writers
from scipy.signal import sawtooth, square
from random import shuffle

# ------------------------------------------------------------------------------

# The analogOut class is designed to work with the USB-6... DAQ card from Ni to
# control the DMD 488 excitation laser.
# There are only 2 analog-output channels that we can use.
# One is employed as fake-digital: 5V = laser active; 0V inactive;.
# The second controls the power in the active state.
# 1% from the old LabView code is 0.1V from the DAQ.

# ------------------------------------------------------------------------------
class analogOut(object):
    
    def __init__(self, device_name='Dev1', channels=['ao0', 'ao1'], save_path=r'D:\Data' ):
    
        """ Constructor:
                - device_name = obvious;
                - channels: ao0 is actual analog, ao1 is used as digital trigger
                - digital_channel = NI USB-6211 doe snot have a digital out, so
                    we can use the other analog, 0V for OFF, 5V for ON ("trigger")
        """

        self.save_path = save_path
        self.device_name = device_name
        self.channels = channels
        self.task = nidaqmx.Task()
        for i in (self.channels):
            # add the active channels to the task
            self.task.ao_channels.add_ao_voltage_chan(device_name + '/'+ i)
# ------------------------------------------------------------------------------
    def constant(self, value=0):
        assert(value <= 3.), '\n max output 3V: you put   %f' %value
        if value > 0:
            self.task.write([value, 5])
            print('DMD ON,  input voltage %f \n' %value)
        else:
            self.task.write([0, 0])
            print('DMD OFF,  input voltage %f \n' %value)

 # ------------------------------------------------------------------------------
    def square(self, frequency = 1, V_max = 1, V_min = 0, num_samples = 10**5):
                    
        """ Generate square wave:
                - frequency = of the signal, [Hz];
                - V_max - V_min = output extrema;
                - num_samples = sampling of signals;
        """
        assert(V_max <= 3.), '\n max output 10V: you put   %f' %V_max
        assert(V_min >= 0.), '\n max output 10V: you put %f' %V_min

        t = np.linspace(0, 1, int(num_samples / frequency))
        one_second = int(len(t) * frequency)
        signal = (square(2 * np.pi * t) + 1) * (V_max - V_min) / 2. + V_min
        trigger = np.zeros((len(t)))
        trigger[signal > V_min] = 5.
        output_matrix = np.zeros((2, len(t)))
        output_matrix[0, :] = signal
        output_matrix[1, :] = trigger
        # CONTINUOUS means that will go on forever.
        # "num samples *2" because every channel transmits "samples", and we have 2 channels
        self.task.timing.cfg_samp_clk_timing(rate=one_second,\
                                             sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                             samps_per_chan=num_samples)
        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.task.out_stream, auto_start=True)
        writer.write_many_sample(output_matrix)

    def square_finite(self, frequency = 1, V_max = 1, V_min = 0, num_samples = 10**5, reps=2):
                    
        """ Generate square wave:
                - frequency = of the signal, [Hz];
                - V_max - V_min = output extrema;
                - num_samples = sampling of signals;
                - reps = repetitions = for how many seconds to stream the signal;
                - (!) call the stop method after this, to reset the Task, making it available
                for another stream.
        """
        assert(V_max <= 5.), '\n max output 5V: you put   %f' %V_max 
        assert(V_min >= 0.), '\n max output 10V: you put %f' %V_min

        t = np.linspace(0, 1, int(num_samples / frequency))
        one_second = int(len(t) * frequency)
        signal = (square(2 * np.pi * t) + 1) * (V_max - V_min) / 2. + V_min
        trigger = np.zeros((len(t)))
        trigger[signal > V_min] = 5.
        output_matrix = np.zeros((2, len(t)))
        output_matrix[0, :] = signal
        output_matrix[1, :] = trigger
        # CONTINUOUS means that will go on forever.
        # "num samples *2" because every channel transmits "samples", and we have 2 channels
        # self.task.timing.cfg_samp_clk_timing(rate=one_second,\
        #                                      sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
        #                                      samps_per_chan=num_samples * 2)
        self.task.timing.cfg_samp_clk_timing(rate=one_second,\
                    sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan=int((num_samples * reps)-1))

        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.task.out_stream, auto_start=True)
        writer.write_many_sample(output_matrix)
 
    def ramp_plus_delay(self, frequency = 1, duty=0.5, V_max = 1, V_min = 0, num_samples = 10**5):
                    
        """ Generate linear ramp, with an extra delay:
                - frequency = of the signal, [Hz];
                - V_max = obvious;
                - V_min = obvious;
                - num_samples = sampling of signals;
        """
        assert(V_max <= 3.), '\n max output 10V: you put   %f' %V_max
        assert(V_min >= 0.), '\n max output 10V: you put %f' %V_min
                    
        t = np.linspace(0, 1, int(num_samples/frequency))
        one_second = int(len(t) * frequency)
        end = int(len(t) * duty)
        t_saw = np.linspace(0, 1, end)
        signal = np.zeros((len(t)))
        signal[:end] = (sawtooth(2 * np.pi * t_saw) + 1) * (V_max - V_min) / 2. + V_min
        trigger = np.zeros((len(t)))
        trigger[signal > V_min] = 5.
        output_matrix = np.zeros((2, len(t)))
        output_matrix[0, :] = signal
        output_matrix[1, :] = trigger

        self.task.timing.cfg_samp_clk_timing(rate=one_second,\
                                             sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                             samps_per_chan=num_samples)
        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.task.out_stream, auto_start=True)
        writer.write_many_sample(output_matrix)

# # ------------------------------------------------------------------------------
        
    def randomised_building_blocks(self, V_max = 1, V_min = 0, num_samples = 10**5, protocol_name='1'):
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
        # always 50s delay between phases -> 1 minute / building block

        samples_per_sec = 10000
        block_duration = 60 # sec

        time = np.linspace(0, 1, samples_per_sec * block_duration) # 10k samples/sec (60s tota)
                                            # more than enough to get smooth signal compared to LS sampling
        signal_A = np.ones((len(time))) * V_min
        signal_B = np.ones((len(time))) * V_min
        signal_C = np.ones((len(time))) * V_min
        signal_D = np.ones((len(time))) * V_min
        trigger_A = np.zeros((len(time)))
        trigger_B = np.zeros((len(time)))
        trigger_C = np.zeros((len(time)))
        trigger_D = np.zeros((len(time)))

        signal_A[:samples_per_sec * 20] = V_max # on for 10 secs
        trigger_A[signal_A > 0] = 5.

        signal_B[:samples_per_sec * 8] = np.linspace(V_min, V_max, samples_per_sec * 8)
        signal_B[samples_per_sec * 8 : samples_per_sec * 12] = V_max
        signal_B[samples_per_sec * 12 : samples_per_sec * 20] = np.linspace(V_max, V_min, samples_per_sec * 8)
        trigger_B[signal_B > 0] = 5.

        signal_C[:samples_per_sec * 4] = np.linspace(V_min, V_max, samples_per_sec * 4)
        signal_C[samples_per_sec * 4 : samples_per_sec * 16] = V_max
        signal_C[samples_per_sec * 16: samples_per_sec * 20] = np.linspace(V_max, V_min, samples_per_sec * 4)
        trigger_C[signal_C > 0] = 5.

        signal_D[:samples_per_sec * 2] = np.linspace(V_min, V_max, samples_per_sec * 2)
        signal_D[samples_per_sec * 2 : samples_per_sec * 18] = V_max
        signal_D[samples_per_sec * 18 : samples_per_sec * 20] = np.linspace(V_max, V_min, samples_per_sec * 2)
        trigger_D[signal_D > 0] = 5.

        full_list = [1, 2, 3, 4, 
                     1, 2, 3, 4, 
                     1, 2, 3, 4, 
                     1, 2, 3, 4] # four repetitions of each stimulus
        shuffle(full_list)

        part = [1, 2, 3, 4]
        total_shuf = []
        for i in range(4):
            shuffle(part)
            total_shuf += part
            part=[1, 2, 3, 4]
        print(total_shuf)
        np.savetxt(self.save_path + "/" + protocol_name + ".txt", total_shuf)

        total_signal = []
        total_trigger = []
        for i in total_shuf:
            if i == 1:
                total_signal.append(signal_A)
                total_trigger.append(trigger_A)
            elif i == 2:
                total_signal.append(signal_B)
                total_trigger.append(trigger_B)
            elif i == 3:
                total_signal.append(signal_C)
                total_trigger.append(trigger_C)
            elif i == 4:
                total_signal.append(signal_D)
                total_trigger.append(trigger_D)
        total_signal = np.asarray(total_signal).flatten()
        total_trigger = np.asarray(total_trigger).flatten()

        output_matrix = np.zeros((2, len(total_signal)))
        output_matrix[0, :] = total_signal
        output_matrix[1, :] = total_trigger
        # CONTINUOUS means that will go on forever.
        self.task.timing.cfg_samp_clk_timing(rate=samples_per_sec,\
                                             sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                             samps_per_chan=len(total_signal)-1)
        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                        self.task.out_stream, auto_start=True)
        writer.write_many_sample(output_matrix)

# ------------------------------------------------------------------------------
    def stop(self):
        """ Stop the task.
            Close it (forget eerything).
            Then open again.
        """
        self.task.stop()
        self.task.close()
        self.task = nidaqmx.Task()
        for i in range (len(self.channels)):
            # add the active channels to the task
            self.task.ao_channels.add_ao_voltage_chan(\
                                self.device_name + '/' + self.channels[i])
        # default everything to 0
        self.task.write([0, 0])

    def close(self):
        """ Completely free the task.
            Close everything.
        """
        self.task.stop()
        self.task.close()

# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------





















