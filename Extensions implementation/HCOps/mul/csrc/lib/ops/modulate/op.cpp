#include "../ops.hpp"
#include <arm_sve.h>
#include <ATen/Parallel.h>

namespace kpops {
    void modulateKernel(const torch::Tensor& x, 
        const torch::Tensor& sh, const torch::Tensor& sc, torch::Tensor& result) {
            int64_t xstrideN = x.stride(0);
            int64_t xstrideT = x.stride(1);
            int64_t sstrideN = sh.stride(0);
            int64_t N = x.size(0);
            int64_t T = x.size(1);
            int64_t D = x.size(2);
            float* x_ptr = x.data_ptr<float>();
            float* sh_ptr = sh.data_ptr<float>();
            float* sc_ptr = sc.data_ptr<float>();
            float* r_ptr = result.data_ptr<float>();
            at::parallel_for(0, N * T, 1, [&](std::size_t begin, std::size_t end){
                for (std::size_t i = begin;i < end; ++i) {
                    std::size_t n = i / T;
                    std::size_t t = i % T;
                    float* local_a = x_ptr + n * xstrideN + t * xstrideT;
                    float* local_r = r_ptr + i * D;
                    float* local_sh = sh_ptr + n * sstrideN;
                    float* local_sc = sc_ptr + n * sstrideN;
                    for (int64_t j = 0;j < D; j += 16) {
                        SO pred = svwhilelt_b32(j,D);
                        SF svx = svld1_f32(pred, local_a + j);
                        SF svsh = svld1_f32(pred, local_sh + j);
                        SF svsc = svld1_f32(pred, local_sc + j);
                        SF svsc1 = svadd_n_f32_m(pred, svsc, 1);
                        SF res = svmla_f32_m(pred, svsh, svsc1, svx);
                        svst1_f32(pred, local_r + j, res);
                    }
                }
            });
        }
    void modulateBackwardKernel(const torch::Tensor& x, const torch::Tensor& sc,torch::Tensor& grad_x,
        torch::Tensor& grad_sh, torch::Tensor& grad_sc,const torch::Tensor& grad_output) {
            int64_t xstrideN = x.stride(0);
            int64_t xstrideT = x.stride(1);
            int64_t sstrideN = sc.stride(0);
            int64_t gstrideN = grad_output.stride(0);
            int64_t gstrideT = grad_output.stride(1);
            int64_t gshstrideN = grad_sh.stride(0);
            int64_t gscstrideN = grad_sc.stride(0);
            int64_t gxstrideN = grad_x.stride(0);
            int64_t gxstrideT = grad_x.stride(1);

            int64_t N = x.size(0);
            int64_t T = x.size(1);
            int64_t D = x.size(2);
            float* x_ptr = x.data_ptr<float>();
            float* sc_ptr = sc.data_ptr<float>();

            float* g_ptr = grad_output.data_ptr<float>();
            float* gx_ptr = grad_x.data_ptr<float>();
            float* gsh_ptr = grad_sh.data_ptr<float>();
            float* gsc_ptr = grad_sc.data_ptr<float>();


            at::parallel_for(0, N * T, 1,[&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin;i < end; ++i) {
                    std::size_t n = i / T;
                    std::size_t t = i % T;

                    float* local_a = x_ptr + n * xstrideN + t * xstrideT;
                    float* local_sc = sc_ptr + n * sstrideN;
                    float* local_g = g_ptr + n * gstrideN + t * gstrideT;
                    float* local_gx = gx_ptr + n * gxstrideN + t * gxstrideT;
                    float* local_gsh = gsh_ptr + n * gshstrideN;
                    float* local_gsc = gsc_ptr + n * gscstrideN;

                    for (int64_t j = 0;j < D; j += 16) {
                        SO pred = svwhilelt_b32(j, D);
                        SF svgsh = svld1_f32(pred, local_gsh + j);
                        SF svg = svld1_f32(pred, local_g + j);
                        SF gsh = svadd_f32_m(pred, svg, svgsh);
                        svst1_f32(pred, local_gsh + j, gsh);

                        SF svx = svld1_f32(pred, local_a + j);
                        SF svgsc = svld1_f32(pred, local_gsc + j);
                        SF svgx = svmla_f32_m(pred,svgsc, svx, svg);
                        svst1_f32(pred, local_gsc + j, svgx);

                        SF svsc = svld1_f32(pred, local_sc + j);
                        SF svsc1 = svadd_n_f32_m(pred, svsc, 1);
                        SF gsc = svmul_f32_m(pred, svsc1, svg);
                        svst1_f32(pred, local_gx + j, gsc);
                    }
                }
            });

    }




}