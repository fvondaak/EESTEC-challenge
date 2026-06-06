# DEEPCRAFT Studio 5.12.5418.0
# Copyright © 2026 Imagimob AB, All Rights Reserved.
# 
# Layer           Shape           Type       Function
# GCC_PHAT_4MIC   [6, 100]        float      compute
# 

#pragma IMAGINET_INCLUDES_BEGIN
import numpy as np
import enum
#pragma IMAGINET_INCLUDES_END

#pragma IMAGINET_FRAGMENT_BEGIN "gcc_phat_4mic_py"

def gcc_phat_dcase(sig4ch, output, num_samples, sampling_rate=16000):
    """
    Computes GCC-PHAT for 4 channels across ALL 6 unique microphone pairs
    exactly as specified in the DCASE paper:
    1. Standard generalized cross-correlation with PHAse Transform (fixing Eq. 1 errors)
    2. Windowing to a strict ±50ms window
    3. Resampling/Evaluating down to exactly 100 discrete lag steps
    4. Global min-max normalization across the entire 600-dimensional frame (Eq. 2)
    
    Input shape: (4, num_samples)
    Output shape: (6, 100)
    """
    # 6 unique pairs from a 4-channel microphone array
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    # Calculate index boundaries for ±50ms
    max_lag_samples = int(0.050 * sampling_rate) # e.g., 800 samples at 16kHz
    
    # 1. Compute FFT across the time axis
    X = np.fft.fft(sig4ch, axis=-1)
    
    cc_time_all = np.zeros((6, num_samples), dtype=np.float64)
    
    for idx, (i, j) in enumerate(pairs):
        # 2. Correct Cross-Power Spectrum using complex conjugate: X_i * conj(X_j)
        G_xy = X[i, :] * np.conj(X[j, :])
        
        # 3. Apply PHAT normalization
        G_xy_phat = G_xy / (np.abs(G_xy) + 1e-9)
        
        # 4. Convert to time domain via Inverse FFT
        cc_time_all[idx, :] = np.real(np.fft.ifft(G_xy_phat, axis=-1))

    # 5. Shift zero-lag to the center
    cc_shifted = np.fft.fftshift(cc_time_all, axes=-1)
    center_idx = num_samples // 2
    
    # Crop the raw time-domain window corresponding to [-50ms, +50ms]
    start_lag = center_idx - max_lag_samples
    end_lag = center_idx + max_lag_samples + 1
    cc_cropped = cc_shifted[:, start_lag:end_lag]
    
    # 6. Resample the lag axis down to exactly 100 uniform steps as per the paper
    num_target_lags = 100
    cc_resampled = np.zeros((6, num_target_lags), dtype=np.float64)
    
    raw_indices = np.linspace(0, cc_cropped.shape[1] - 1, cc_cropped.shape[1])
    target_indices = np.linspace(0, cc_cropped.shape[1] - 1, num_target_lags)
    
    for idx in range(6):
        cc_resampled[idx, :] = np.interp(target_indices, raw_indices, cc_cropped[idx, :])
    
    # 7. Apply Equation (2): Min-max normalization globally across the entire vector phi(n)
    val_min = np.min(cc_resampled) 
    val_max = np.max(cc_resampled)
    
    cc_norm = (cc_resampled - val_min) / (val_max - val_min + 1e-9)
    
    np.copyto(output, cc_norm)

class ReturnStatus(enum.Enum): 
    RET_SUCCESS = 0
    RET_NODATA = -1
    RET_NOMEM = -2

class Model:
    def __init__(self, window_size=1024, sampling_rate=16000):
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        
        # Change output shape to a flat 600-dimension feature vector
        self.num_lags = 100
        self.data_in_shape = (4, self.window_size)
        self.data_out_shape = (600,)  # Flattened target shape
        self.api = 'function'

    def compute(self, data_in : np.array, data_out : np.array):
        sig_input = data_in.reshape((4, self.window_size))
        
        # Allocating intermediate buffer for the 2D matrix calculation
        matrix_buffer = np.zeros((6, self.num_lags), dtype=np.float64)
        gcc_phat_dcase(sig_input, matrix_buffer, self.window_size, self.sampling_rate)
        
        # Flatten the (6, 100) matrix into a 1D array of 600 elements
        flat_features = matrix_buffer.flatten()
        
        # Copy directly to the framework's output buffer
        np.copyto(data_out.reshape(600), flat_features)
        print(f"output is {self.data_out.shape()}")
        return ReturnStatus.RET_SUCCESS.value

# Global mapping entry point explicitly invoked by the imunit file execution loop
def compute_gcc_phat_py(input, window_size, output):
    model_instance = Model(window_size=window_size, sampling_rate=16000)
    model_instance.compute(input, output)

#pragma IMAGINET_FRAGMENT_END
