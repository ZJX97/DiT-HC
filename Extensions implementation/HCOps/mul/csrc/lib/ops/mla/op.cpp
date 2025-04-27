#include "../ops.hpp"
#include <arm_sve.h>
#include <ATen/Parallel.h>

namespace kpops {
    void mlaKernel(const torch::Tensor& x, const torch::Tensor& gate, 
        const torch::Tensor& y, torch::Tensor& result) {
            int64_t xstrideN = x.stride(0);
            assert(x.stride(1) == 1 && "kpops.mla:: x stride(1) should be 1");
            int64_t xstrideD = x.stride(2);

            int64_t gatestrideN = gate.stride(0);

            int64_t ystrideN = y.stride(0);
            int64_t ystrideT = y.stride(1);
            int64_t ystrideD = y.stride(2);

            int64_t N = x.size(0);
            int64_t T = x.size(1);
            int64_t D = x.size(2);

            float* x_ptr = x.data_ptr<float>();
            float* g_ptr = gate.data_ptr<float>();
            float* y_ptr = y.data_ptr<float>();
            float* r_ptr = result.data_ptr<float>();

            at::parallel_for(0, N * D, 1, [&](std::size_t begin, std::size_t end){
                float* buff = (float *)malloc(16 * sizeof(float));

                for (std::size_t i = begin; i < end; ++i) {
                    std::size_t n = i / D;
                    std::size_t d = i % D;

                    float gsca = g_ptr[n * gatestrideN + d];
                    float* local_x = x_ptr + n * xstrideN + d * xstrideD;
                    float* local_r = r_ptr + n * xstrideN + d * xstrideD;

                    for (int64_t j = 0;j < T; j += 16) {
                        SO pred = svwhilelt_b32(j, T);
                        int64_t base = n * ystrideN + d * ystrideD + j * ystrideT;
                        for (int64_t t = 0;t < 16 && t + j < T; ++t) {
                            buff[t] = y_ptr[base];
                            base += ystrideT;
                        }

                        SF svy = svld1_f32(pred, buff);
                        SF svx = svld1_f32(pred, local_x + j);
                        SF svm = svmul_n_f32_m(pred, svy, gsca);

                        SF res = svadd_f32_m(pred, svm, svx);
                        svst1_f32(pred, local_r + j, res);
                    }
                }
                free(buff);
            });
    }
    void mlaBackwardKernel(const torch::Tensor& gate, const torch::Tensor& y,
        const torch::Tensor& grad_output, torch::Tensor& grad_gate, torch::Tensor& grad_y) {
            int64_t gstrideN = grad_output.stride(0);
            int64_t gstrideT = grad_output.stride(1);

            int64_t ystrideN = y.stride(0);
            int64_t ystrideT = y.stride(1);

            int64_t gatestrideN = gate.stride(0);
            int64_t ggstrideN = grad_gate.stride(0);
            int64_t gystrideN = grad_y.stride(0);
            int64_t gystrideT = grad_y.stride(1);

            assert(y.stride(2) == 1 && "mla backward: y.stride(2) should be 1");

            int64_t N = y.size(0);
            int64_t T = y.size(1);
            int64_t D = y.size(2);
            float* gate_ptr = gate.data_ptr<float>();
            float* y_ptr = y.data_ptr<float>();
            float* g_ptr = grad_output.data_ptr<float>();
            float* gg_ptr = grad_gate.data_ptr<float>();
            float* gy_ptr = grad_y.data_ptr<float>();
            at::parallel_for(0, N * T, 1, [&](std::size_t begin, std::size_t end){
                for (std::size_t i = begin;i < end; ++i) {
                    std::size_t n = i / T;
                    std::size_t t = i % T;

                    float* local_gate = gate_ptr + n * gatestrideN;
                    float* local_gg = gg_ptr + n * ggstrideN;
                    float* local_y = y_ptr + n * ystrideN + t * ystrideT;
                    float* local_g = g_ptr + n * gstrideN + t * gstrideT;
                    float* local_gy = gy_ptr + n * gystrideN + t * gystrideT;

                    for (int64_t j = 0;j < D; j += 16) {
                        SO pred = svwhilelt_b32(j, D);
                        SF svg = svld1_f32(pred, local_g + j);
                        SF svy = svld1_f32(pred, local_y + j);
                        SF svgate = svld1_f32(pred, local_gate + j);
                        SF svgg = svld1_f32(pred, local_gg + j);
                        SF svgyg = svmla_f32_m(pred, svgg, svg, svy);
                        svst1_f32(pred, local_gg + j, svgyg);
                        SF svggg = svmul_f32_m(pred, svg, svgate);
                        svst1_f32(pred, local_gy + j, svggg);
                    }
                }
            });

        }
}