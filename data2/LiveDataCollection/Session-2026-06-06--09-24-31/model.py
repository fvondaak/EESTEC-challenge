# DEEPCRAFT Studio 5.12.5418.0
# Copyright © 2026 Imagimob AB, All Rights Reserved.
# 
# Layer                         Shape           Type       Function
# GCC_PHAT_4MIC                [3, N]          float      compute
# 

import numpy as np
import enum

def gcc_phat(sig4ch, output, num_samples):
    """
    Computes GCC-PHAT for 4 channels using Mic 0 as the reference channel.
    Input shape: (4, num_samples)
    Output shape: (3, num_samples) -> Pairs: (0-1), (0-2), (0-3)
    """
    # 1. Compute FFT across the time axis (axis=-1)
    # Using standard fft since DeepCraft inputs are real audio signals
    X = np.fft.fft(sig4ch, axis=-1)
    
    # Isolate reference channel (Mic 0) and its complex conjugate
    X_ref_conj = np.conj(X[0, :])
    
    # Initialize an array to hold the cross-power spectrums
    # We have 3 pairs remaining (0-1, 0-2, 0-3)
    cc_channels = np.zeros((3, num_samples), dtype=np.complex128)
    
    for i in range(1, 4):
        # 2. Compute cross-power spectrum
        G_xy = X[i, :] * X_ref_conj
        
        # 3. Apply the Phase Transform (PHAT) weighting: divide by magnitude
        # Add a tiny epsilon to avoid division by zero
        G_xy_phat = G_xy / (np.abs(G_xy) + 1e-9)
        
        cc_channels[i-1, :] = G_xy_phat

    # 4. Convert back to time domain via Inverse FFT
    # Take the real part as cross-correlation is real-valued
    cc_time = np.real(np.fft.ifft(cc_channels, axis=-1))
    
    # 5. Shift the zero-frequency component to the center of the spectrum
    # This aligns 0 delay in the center of the array window
    cc_shifted = np.fft.fftshift(cc_time, axes=-1)
    
    np.copyto(output, cc_shifted)

class ReturnStatus(enum.Enum): 
    RET_SUCCESS = 0
    RET_NODATA = -1
    RET_NOMEM = -2

class Model:
    def __init__(self, window_size=1024):
        """
        window_size: The frame length of the audio slices processing through the block.
        Common sizes are 512, 1024, or 2048 samples.
        """
        self.window_size = window_size
        
        # Expecting a flattened or structured input tensor of 4 channels * window_size
        self.data_in_count = 1
        self.data_in_shape = (4, self.window_size)
        
        # Outputting 3 cross-correlation curves (Mic 0-1, Mic 0-2, Mic 0-3)
        self.data_out_count = 1
        self.data_out_shape = (3, self.window_size)
        self.api = 'function'

    def compute(self, data_in : np.array, data_out : np.array):
        """
        Compute a forward pass for GCC-PHAT processing.
        
        Parameters:
         data_in(float[4, window_size]): 4 synchronized audio channels.
         data_out(float[3, window_size]): 3 GCC-PHAT cross-correlation curves.
        """
        # Ensure input data matches the expected 2D grid shape (4 channels, N samples)
        sig_input = data_in.reshape((4, self.window_size))
        
        # Reshape output container to match the output pair shapes
        out_buffer = data_out.reshape((3, self.window_size))
        
        # Execute GCC-PHAT calculation
        gcc_phat(sig_input, out_buffer, self.window_size)
        
        return ReturnStatus.RET_SUCCESS.value