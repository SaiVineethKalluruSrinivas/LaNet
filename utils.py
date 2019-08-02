import numpy as np
from scipy.fftpack import fft


class Batching:
    def __init__(self, length, stride, signal):
        self.length = length
        self.stride = stride
        self.signal = signal
        self.curr_index = 0
    def getNextBatch(self):
        if (self.curr_index >= len(self.signal)):
            return
        upper_lim = min(self.curr_index + self.length, len(self.signal))
        to_ret =  self.signal[self.curr_index:upper_lim, 0]
        self.curr_index += self.stride
        done = False
        if (self.length > len(to_ret)):
            ##padding drives with less than "d"
            to_ret = np.pad(to_ret, (0,self.length - len(to_ret)), 'constant', constant_values=(0))
            done = True
        return to_ret.reshape(self.length, 1), done
        
class FFT:
    def __init__(self, sampling_rate=50, bucket_size=500):
        self.sampling_rate = sampling_rate
        self.bucket_size = bucket_size
        
    def returnFFT(self, signal):
        N = self.bucket_size
        # Nyquist Sampling Criteria
        T = 1/self.sampling_rate 
        x = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

        # FFT algorithm
        yr = fft(signal) # "raw" FFT with both + and - frequencies
        y = 2/N * np.abs(yr[0:np.int(N/2)]) # positive freqs only
        return y
        

def process_drive(drive, sub_drive_stride, sub_segment_stride, input_dim, sampling_rate, range_len):
    ## If the drive is padded with a large number at the end, remove the padding. 
    indices = np.where(drive > 100)
    if (len(indices) == 0 or len(indices[0]) == 0):
        pass
    elif (indices[0][0] >= range_len):
        drive = drive[:indices[0][0]]
    else:
        return None, None, None 
    batching_obj = Batching(length=range_len, stride=sub_drive_stride,signal = drive.reshape(-1,1))
    #Drive level batching to produce sub drives
    this_batch, is_done = batching_obj.getNextBatch()
    num_batches = 0
    this_drive_batches = []
    while(this_batch is not None and this_batch.shape[0] > 0 and not is_done):
        num_batches += 1
        this_drive_batches.append(this_batch[:,0])
        this_batch, is_done = batching_obj.getNextBatch()
    #Sub Drive level batching to produce sub-segments
    total_drives_sub_batches = []
    for i in range(num_batches):
        segment = this_drive_batches[i]
        batching_obj = Batching(length=input_dim, stride=sub_segment_stride,signal = segment.reshape(-1,1))
        this_batch, is_done = batching_obj.getNextBatch()
        num_sub_batches = 0
        this_drive_sub_batches = []
        while(this_batch is not None and this_batch.shape[0] > 0 and not is_done):
            num_sub_batches += 1
            this_drive_sub_batches.append(this_batch[:, 0])
            this_batch, is_done = batching_obj.getNextBatch()
        total_drives_sub_batches.append(this_drive_sub_batches)
    return total_drives_sub_batches, num_batches, num_sub_batches

