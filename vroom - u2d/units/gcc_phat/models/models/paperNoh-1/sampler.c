/*
* DEEPCRAFT Studio 5.12.5418.0+7793ebcc9f383586f202c2d2f6eafbd7ebe6519d
* Copyright © 2023- Imagimob AB, All Rights Reserved.
* 
* Generated at 06/06/2026 09:57:19 UTC. Any changes will be lost.
* 
* Model ID  e5d60389-1f0c-4822-b5f2-cff4075cb88e
* 
* Layer                          Shape           Type       Function
* GCC-PHAT 4-Channel             [3,1,1,2]       float      compute
*    window_size = 512
* 
* Exported functions:
* 
* void IMAI_compute(const float *restrict data_in, float *restrict data_out)
*    Description: Compute a forward pass
*    Parameter data_in is Input of size float[2].
*    Parameter data_out is Output of size float[3,1,1,2].
* 
* void IMAI_init(void)
*    Description: Initializes buffers to initial state. This function also works as a reset function.
* 
* 
* Disclaimer:
*   The generated code relies on the optimizations done by the C compiler.
*   For example many for-loops of length 1 must be removed by the optimizer.
*   This can only be done if the functions are inlined and simplified.
*   Check disassembly if unsure.
*   tl;dr Compile using gcc with -O3 or -Ofast
*/

#include <math.h>
#include <stdint.h>

#include "sampler.h"

// Working memory

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

#define __RETURN_ERROR(_exp) do { int __ret = (_exp); if(__ret < 0) return __ret; } while(0)
#define __RETURN_ERROR_BREAK_EMPTY(_exp) {  int __ret = (_exp); if(__ret == -1) break; if(__ret < 0) return __ret;  } 

void IMAI_compute(const float *restrict data_in, float *restrict data_out) {    
    gcc_phat_4mic_f32(data_in, 512, data_out);
}

void IMAI_init(void) {    
}

