# Phase Map 4MIC Zhang preprocessor

DeepCraft/Imaginet custom preprocessor for the Zhang et al. Interspeech 2019 baseline-style DOA feature.

Pipeline:

```text
DataTrack [4] @ 16000 Hz
→ Sliding Window (data points): Window Shape [256,4], Stride 512
→ IMUnits / Phase Map 4MIC Zhang: output [4,129,1]
→ Conv2D model with input [4,129,1]
```

Parameters:

- Window Length: 256
- DFT Length: 256
- Channel Major: 0 for DeepCraft WAV/datatrack [time,mic]
- Apply Hann: 1

This unit computes the wrapped phase angle of the one-sided real FFT for each of four microphones.
