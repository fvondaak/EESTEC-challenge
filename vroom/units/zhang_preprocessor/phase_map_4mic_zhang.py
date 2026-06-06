# the pragma directives tells imaginet to look for includes here
# you are not allowed to put comments in the include section
#pragma IMAGINET_INCLUDES_BEGIN
import numpy as np
#pragma IMAGINET_INCLUDES_END

#pragma IMAGINET_FRAGMENT_BEGIN "phase_map_4mic_zhang_py"
def phase_map_4mic_zhang_py(input, window_length, n_fft, channel_major, apply_hann, output):
    """Compute 4-mic STFT phase map for one framed 4-channel window.

    Expected input layout:
      channel_major = 0: flattened [time, mic], shape [window_length, 4]
      channel_major = 1: flattened [mic, time], shape [4, window_length]

    Output shape: [4, n_fft//2 + 1, 1], values wrapped to (-pi, pi].
    """
    wl = int(window_length)
    nf = int(n_fft)
    cm = int(channel_major)
    hann = int(apply_hann)

    flat = np.asarray(input, dtype=np.float32).reshape(-1)

    if cm == 0:
        x = flat[:wl * 4].reshape((wl, 4)).T  # [4, wl]
    else:
        x = flat[:wl * 4].reshape((4, wl))    # [4, wl]

    if hann != 0:
        win = np.hanning(wl).astype(np.float32)
        x = x * win[None, :]

    if nf > wl:
        pad = np.zeros((4, nf - wl), dtype=np.float32)
        x_fft_in = np.concatenate([x, pad], axis=1)
    else:
        x_fft_in = x[:, :nf]

    spec = np.fft.rfft(x_fft_in, n=nf, axis=1)
    phase = np.angle(spec).astype(np.float32)  # [4, nf//2+1]

    np.copyto(output.reshape(-1), phase.reshape(-1))
#pragma IMAGINET_FRAGMENT_END
