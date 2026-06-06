# DEEPCRAFT Studio 5.12.5418.0
# Copyright © 2026 Imagimob AB, All Rights Reserved.
# 
# Layer                 Shape           Type       Function
# GCC_PHAT_4MIC        [3, N]          float      compute
# 

#pragma IMAGINET_INCLUDES_BEGIN
import numpy as np
import enum
#pragma IMAGINET_INCLUDES_END

#pragma IMAGINET_FRAGMENT_BEGIN "gcc_phat_4mic_py"

def gcc_phat(sig4ch, output, num_samples):
    """
    Computes GCC-PHAT for 4 channels using Mic 0 as the reference channel.
    Input shape: (4, num_samples)
    Output shape: (3, num_samples) -> Pairs: (0-1), (0-2), (0-3)
    """
    # 1. Compute FFT across the time axis (axis=-1)
    X = np.fft.fft(sig4ch, axis=-1)
    
    # Isolate reference channel (Mic 0) and its complex conjugate
    X_ref_conj = np.conj(X[0, :])
    
    # Initialize an array to hold the cross-power spectrums
    cc_channels = np.zeros((3, num_samples), dtype=np.complex128)
    
    for i in range(1, 4):
        # 2. Compute cross-power spectrum
        G_xy = X[i, :] * X_ref_conj
        
        # 3. Apply the Phase Transform (PHAT) weighting: divide by magnitude
        G_xy_phat = G_xy / (np.abs(G_xy) + 1e-9)
        
        cc_channels[i-1, :] = G_xy_phat

    # 4. Convert back to time domain via Inverse FFT
    cc_time = np.real(np.fft.ifft(cc_channels, axis=-1))
    
    # 5. Shift the zero-frequency component to the center of the spectrum
    cc_shifted = np.fft.fftshift(cc_time, axes=-1)
    
    np.copyto(output, cc_shifted)

class ReturnStatus(enum.Enum): 
    RET_SUCCESS = 0
    RET_NODATA = -1
    RET_NOMEM = -2

class Model:
    def __init__(self, window_size=1024):
        self.window_size = window_size
        self.data_in_count = 1
        self.data_in_shape = (4, self.window_size)
        self.data_out_count = 1
        self.data_out_shape = (3, self.window_size)
        self.api = 'function'

    def compute(self, data_in : np.array, data_out : np.array):
        sig_input = data_in.reshape((4, self.window_size))
        out_buffer = data_out.reshape((3, self.window_size))
        gcc_phat(sig_input, out_buffer, self.window_size)
        return ReturnStatus.RET_SUCCESS.value

# Global mapping entry point explicitly invoked by the imunit file execution loop
def compute_gcc_phat_py(input, window_size, output):
    model_instance = Model(window_size=window_size)
    model_instance.compute(input, output)

#pragma IMAGINET_FRAGMENT_END