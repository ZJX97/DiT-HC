#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <array>
#include <cstdio>
#include <kblas.h>

namespace kpops
{
    
    void linear_forward( const float* X, const float* W, const float* b, float* Y,  int batch_size, int in_features, int out_features); 
    
    void linear_backward( const float* dY, const float* X, const float* W, float* dX, float* dW, float* db, int batch_size, int in_features, int out_features);


}  // namespace kpops::cpu
