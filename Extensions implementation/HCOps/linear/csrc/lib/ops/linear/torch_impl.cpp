#include "../ops.hpp"
#include "../torch_impl.hpp"
#include <c10/util/ArrayRef.h>
#include <iostream>

namespace kpops::torch_impl
{
void kp_linear_forward(const torch::Tensor& X, const torch::Tensor& W, const torch::Tensor& b, torch::Tensor& Y){
    TORCH_CHECK(X.scalar_type()== torch::kFloat32, "Customed Tensor Type X must be float32");
    //std::cout << "========Custom Linear Tensor Size Check========X:" << X.sizes() << "     W:" << W.sizes() << std::endl;
    

    TORCH_CHECK(W.dim()==2, "Dim Confirm from CustomLinear: W must be 2D tensor");
    TORCH_CHECK(b.dim()==1, "Dim Confirm from CustomLinear: bias must be 1D tensor");
    TORCH_CHECK(X.size(-1)==W.size(1),"X and W shape confirm from CustomLinnear: last dim of input must match weight.size(1)");

    //auto input_shape = input.sizes().vec();
    //auto input_flatten = input.reshape({-1, input.size(-1)});
    //int batch_size = input_flatten.size(0);
    //int in_features = input_flatten.size(1);
    //int out_features = W.size(0);

    int batch_size = X.size(0);
    int in_features = X.size(1);
    int out_features = W.size(0);
    
    //auto X_contig = input_flatten.contiguous();
    auto X_contig = X.contiguous();
    auto W_contig = W.contiguous();
    auto b_contig = b.contiguous();
    auto Y_contig = Y.contiguous();

    const float* X_ptr = X_contig.data_ptr<float>();
    const float* W_ptr = W_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* Y_ptr = Y_contig.data_ptr<float>();
    
     
    linear_forward(X_ptr, W_ptr, b_ptr, Y_ptr, batch_size, in_features, out_features);

    if(!Y.is_contiguous()){
        Y.copy_(Y_contig);
    }
}
void kp_linear_backward(const torch::Tensor& dY, const torch::Tensor& X, const torch::Tensor& W, torch::Tensor& dX, torch::Tensor& dW, torch::Tensor& db){
    int batch_size=X.size(0);
    int in_features=X.size(1);
    int out_features= W.size(0);

    auto dY_contig = dY.contiguous();
    auto  X_contig =  X.contiguous();
    auto  W_contig =  W.contiguous();
    auto dX_contig = dX.contiguous();
    auto dW_contig = dW.contiguous();
    auto db_contig = db.contiguous();

    const float* dY_ptr = dY_contig.data_ptr<float>();
    const float*  X_ptr =  X_contig.data_ptr<float>();
    const float*  W_ptr =  W_contig.data_ptr<float>();
    float* dX_ptr = dX_contig.data_ptr<float>();
    float* dW_ptr = dW_contig.data_ptr<float>();
    float* db_ptr = db_contig.data_ptr<float>();

    linear_backward(dY_ptr, X_ptr, W_ptr, dX_ptr, dW_ptr, db_ptr, batch_size, in_features, out_features);

    if(!dX.is_contiguous()){
        dX.copy_(dX_contig);
    }
    if(!dW.is_contiguous()){
        dW.copy_(dX_contig);
    }
    if(!db.is_contiguous()){
        db.copy_(dX_contig);
    }

}





}  // namespace kpops::toruh_impl

