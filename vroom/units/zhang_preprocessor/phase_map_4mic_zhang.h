#pragma IMAGINET_INCLUDES_BEGIN
#include <stdint.h>
#include <math.h>
#pragma IMAGINET_INCLUDES_END

#pragma IMAGINET_FRAGMENT_BEGIN "phase_map_4mic_zhang"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float phase_map_4mic_read_sample_f32(const float* input, int t, int mic, int window_length, int channel_major)
{
    if (channel_major == 0)
    {
        return input[t * 4 + mic];      /* flattened [time, mic] */
    }
    else
    {
        return input[mic * window_length + t];  /* flattened [mic, time] */
    }
}

static inline void phase_map_4mic_zhangf32(const float* restrict input,
                                           int window_length,
                                           int n_fft,
                                           int channel_major,
                                           int apply_hann,
                                           float* restrict output)
{
    const int num_mics = 4;
    const int k_bins = n_fft / 2 + 1;

    for (int mic = 0; mic < num_mics; ++mic)
    {
        for (int k = 0; k < k_bins; ++k)
        {
            float re = 0.0f;
            float im = 0.0f;

            for (int n = 0; n < window_length; ++n)
            {
                float x = phase_map_4mic_read_sample_f32(input, n, mic, window_length, channel_major);

                if (apply_hann != 0)
                {
                    float w = 0.5f - 0.5f * cosf((2.0f * (float)M_PI * (float)n) / (float)(window_length - 1));
                    x *= w;
                }

                float angle = -2.0f * (float)M_PI * (float)k * (float)n / (float)n_fft;
                re += x * cosf(angle);
                im += x * sinf(angle);
            }

            output[(mic * k_bins + k)] = atan2f(im, re);
        }
    }
}

#pragma IMAGINET_FRAGMENT_END
