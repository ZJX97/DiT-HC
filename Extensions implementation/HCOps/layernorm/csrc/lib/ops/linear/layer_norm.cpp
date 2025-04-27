#include "./ops.hpp"
#include <arm_sve.h>
namespace kpops{

// LayerNorm forward pass implementation
void layer_norm_forward(
    float* output,           // Output: normalized tensor [batch_size, hidden_size]
    float* mean,             // Output: mean values [batch_size]
    float* invstd,           // Output: inverse std values [batch_size]
    const float* input,      // Input: tensor to normalize [batch_size, hidden_size]
    const float* weight,     // Input: gamma parameter [hidden_size]
    const float* bias,       // Input: beta parameter [hidden_size]
    int batch_size,          // Input: batch size
    int hidden_size,         // Input: hidden dimension size
    float eps                // Input: epsilon for numerical stability
) {
    // For each sample in the batch
    for (int i = 0; i < batch_size; i++) {
        // Calculate mean
        mean[i] = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            mean[i] += input[i * hidden_size + j];
        }
        mean[i] /= hidden_size;
        
        // Calculate variance
        float var = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            float diff = input[i * hidden_size + j] - mean[i];
            var += diff * diff;
        }
        var /= hidden_size;
        
        // Calculate inverse standard deviation
        invstd[i] = 1.0f / sqrtf(var + eps);
        
        // Normalize, scale and shift
        for (int j = 0; j < hidden_size; j++) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * invstd[i];
            output[i * hidden_size + j] = normalized * weight[j] + bias[j];
        }
    }
}
	
// LayerNorm backward pass implementation
void layer_norm_backward(
    float* d_input,          // Output: gradient w.r.t. input [batch_size, hidden_size]
    float* d_weight,         // Output: gradient w.r.t. weight [hidden_size]
    float* d_bias,           // Output: gradient w.r.t. bias [hidden_size]
    const float* d_output,   // Input: gradient from upstream [batch_size, hidden_size]
    const float* input,      // Input: original input [batch_size, hidden_size]
    const float* weight,     // Input: gamma parameter [hidden_size]
    const float* mean,       // Input: saved mean from forward pass [batch_size]
    const float* invstd,     // Input: saved inv_std from forward pass [batch_size]
    int batch_size,          // Input: batch size
    int hidden_size          // Input: hidden dimension size
) {
    // Initialize gradients for weight and bias
    for (int j = 0; j < hidden_size; j++) {
        d_weight[j] = 0.0f;
        d_bias[j] = 0.0f;
    }
    
    // For each sample in the batch
    for (int i = 0; i < batch_size; i++) {
        // Calculate gradients for weight and bias
        for (int j = 0; j < hidden_size; j++) {
            float normalized = (input[i * hidden_size + j] - mean[i]) * invstd[i];
            d_weight[j] += d_output[i * hidden_size + j] * normalized;
            d_bias[j] += d_output[i * hidden_size + j];
        }
        
        // Calculate gradient w.r.t. normalized inputs
        std::vector<float> d_normalized(hidden_size, 0.0f);
        for (int j = 0; j < hidden_size; j++) {
            d_normalized[j] = d_output[i * hidden_size + j] * weight[j];
        }
        
        // Calculate sum of gradient w.r.t. normalized inputs
        float sum_d_normalized = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum_d_normalized += d_normalized[j];
        }
        
        // Calculate sum of input-centered * gradient w.r.t. normalized inputs
        float sum_centered_times_d_normalized = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            float centered = (input[i * hidden_size + j] - mean[i]);
            sum_centered_times_d_normalized += centered * d_normalized[j];
        }
        
        // Calculate gradients for input
        for (int j = 0; j < hidden_size; j++) {
            float centered = (input[i * hidden_size + j] - mean[i]);
            d_input[i * hidden_size + j] = invstd[i] * (
                d_normalized[j] - 
                sum_d_normalized / hidden_size - 
                centered * sum_centered_times_d_normalized * invstd[i] * invstd[i] / hidden_size
            );
        }
    }

}
} //namespace kpops
