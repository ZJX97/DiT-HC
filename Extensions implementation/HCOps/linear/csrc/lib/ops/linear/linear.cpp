#include "../ops.hpp"
//#include <arm_sve.h>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <cstdlib>
#define COL
const int THREADNUM_L1 = std::getenv("THREADNUM_L1") ? std::atoi(std::getenv("THREADNUM_L1"))  : 4 ;
const int THREADNUM_L2 = std::getenv("THREADNUM_L2") ? std::atoi(std::getenv("THREADNUM_L2"))  : 36;



namespace kpops{
    void linear_forward( const float* X, const float* W, const float* b, float* Y,  int batch_size, int in_features, int out_features){
        if(batch_size % THREADNUM_L1 != 0){//TODO:remove this func in the final version, check size outside.
            std::cout <<"N cannot be divided equally, check the bs" <<std::endl; 
        }
        int N_per_thread = (int)(batch_size/THREADNUM_L1); 
        
        //Clac Y^T = W*X^T
#ifdef COL
#pragma omp parallel num_threads(THREADNUM_L1)
    {
        int l1_rank = omp_get_thread_num();
        float* B_ptr = (float*)X + l1_rank * in_features * N_per_thread;
        float* C_ptr = Y + l1_rank * out_features * N_per_thread;
        BlasSetNumThreadsLocal(THREADNUM_L2);
        cblas_sgemm(
                CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                out_features, //M  out_features
                //batch_size, //N  bs
                N_per_thread, //N  bs
                in_features, //K in_features
                1.0f,
                W,
                in_features,
                //X,
                B_ptr,
                in_features,
                1.0f,
                //Y,
                C_ptr,
                out_features
                );
        BlasSetNumThreadsLocal(1);
    }
    for(int i = 0; i <batch_size; i++){
        for(int j = 0; j < out_features; j++){
            Y[i*out_features + j] += b[j];
        }
    }
        
#else
        std::cout <<"Row Major" << std::endl;
       cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            batch_size,
            out_features,
            in_features,
            1.0f,
            X,
            in_features,
            W,
            in_features,
            0.0f,
            Y,
            out_features
        );
        for (int i = 0; i < batch_size; i++){
            for (int j = 0; j < out_features; j++){
                Y[i*out_features + j] += b[j];
            }
        }
#endif
    }
    
    void linear_backward( const float* dY, const float* X, const float* W, float* dX, float* dW, float* db, int batch_size, int in_features, int out_features){
        //dX^T = W^T * dY^T;//dY^T has already save in a trans way. 
        int N_per_thread_1 =  (int)(batch_size/THREADNUM_L1); 
        int N_per_thread_2 =  (int)(out_features/THREADNUM_L1); 
#ifdef COL
#pragma omp parallel num_threads(THREADNUM_L1)
    {
        int l1_rank = omp_get_thread_num();
        //std::cout << "current rankid :" <<l1_rank<<std::endl;
        float* B_ptr_1 = (float*)dY + l1_rank * out_features * N_per_thread_1; 
        //float* B_ptr_2 = (float*)dY + l1_rank * out_features * N_per_thread_2; 
        float* B_ptr_2 = (float*)dY + l1_rank  * N_per_thread_2; 
        float* C_ptr_1 = dX + l1_rank * in_features * N_per_thread_1; 
        float* C_ptr_2 = dW + l1_rank * in_features * N_per_thread_2; 
        BlasSetNumThreadsLocal(THREADNUM_L2);
        cblas_sgemm(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            in_features,
            //batch_size,
            N_per_thread_1,
            out_features,
            1.0f,
            W,
            in_features,
            //dY,
            B_ptr_1,
            out_features,
            1.0f,
            //dX,
            C_ptr_1,
            in_features
        );
        //dW^T = X^T * dY
        cblas_sgemm(
            CblasColMajor,
            CblasNoTrans,
            CblasTrans,
            in_features,
            //out_features,
            N_per_thread_2,
            batch_size,
            1.0f,
            X,
            in_features,
            //dY,
            B_ptr_2,
            out_features,
            1.0f,
            //dW,
            C_ptr_2,
            in_features
        );
        BlasSetNumThreadsLocal(1);

    }
        //db = sum(dY, axis=0)
        memset(db, 0, sizeof(float)*out_features);
        for(int i = 0; i <batch_size; i++){
            for (int j = 0; j < out_features; j++){
                db[j] += dY[i* out_features + j];
            }
        }




//        cblas_sgemm(
//            CblasColMajor,
//            CblasNoTrans,
//            CblasNoTrans,
//            in_features,
//            batch_size,
//            out_features,
//            1.0f,
//            W,
//            in_features,
//            dY,
//            out_features,
//            0.0f,
//            dX,
//            in_features
//        );
//        //dW^T = X^T * dY
//        cblas_sgemm(
//            CblasColMajor,
//            CblasNoTrans,
//            CblasTrans,
//            in_features,
//            out_features,
//            batch_size,
//            1.0f,
//            X,
//            in_features,
//            dY,
//            out_features,
//            0.0f,
//            dW,
//            in_features
//        );
//
//        //db = sum(dY, axis=0)
//        memset(db, 0, sizeof(float)*out_features);
//        for(int i = 0; i <batch_size; i++){
//            for (int j = 0; j < out_features; j++){
//                db[j] += dY[i* out_features + j];
//            }
//        }
#else
        std::cout << "Row Major backward" << std::endl;
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            batch_size,
            in_features,
            out_features,
            1.0f,
            dY,
            out_features,
            W,
            in_features,
            0.0f,
            dX,
            in_features
        );
        cblas_sgemm(
            CblasRowMajor,
            CblasTrans,
            CblasNoTrans,
            out_features,
            in_features,
            batch_size,
            1.0f,
            dY,
            out_features,
            X,
            in_features,
            0.0f,
            dW,
            in_features
        );

        memset(db, 0, sizeof(float)*out_features);
        for (int i = 0; i <batch_size; i++){
            for (int j=0; j <out_features; j ++){
                db[j] += dY[i*out_features +j];
            }
        }
#endif
    }

} //namespace kpops
