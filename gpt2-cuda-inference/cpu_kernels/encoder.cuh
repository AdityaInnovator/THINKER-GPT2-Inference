//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#ifndef ENCODER_CUH
#define ENCODER_CUH

#include <assert.h>
#include <math.h>
#include <float.h>

void encoder_forward_cpu(float* out, const int* inp, const float* wte, const float* wpe,
                         int B, int T, int C) {
    for (int b = 0; b < B; ++b) {   // for each batch
        for (int token_pos = 0; token_pos < T; ++token_pos) {   // for each token position
            for (int c = 0; c < C; ++c) {   // compute for each component of the vector (token)
                int token = inp[b * T + token_pos]; // can be moved above this outside the c loop
                out[b * T * C + token_pos * C + c] = wte[token * C + c] + wpe[token_pos * C + c];
            }
        }
    }
}

#endif // ENCODER_CUH