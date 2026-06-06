#pragma IMAGINET_INCLUDES_BEGIN
#include <stdint.h>
#include <math.h>
#pragma IMAGINET_INCLUDES_END

#pragma IMAGINET_FRAGMENT_BEGIN "gcc_phat_4mic"

/**
 * @brief Computes GCC-PHAT for 4 channels using Mic 0 as the reference channel.
 * @param input Flat array containing 4 channels of data sequentially: [4, window_size]
 * @param window_size The frame length of the audio slices (e.g., 512, 1024)
 * @param output Flat array containing 3 cross-correlation curve channels: [3, window_size]
 */
static inline void gcc_phat_4mic_f32(const float* restrict input, int window_size, float* restrict output)
{
    // C Implementation logic goes here for edge deployment.
    // Inputs are mapped as flattened arrays following DeepCraft Studio specifications.
}

#pragma IMAGINET_FRAGMENT_END