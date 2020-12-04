/****************************************
*\file utils.h
*\brief 一些工具函数
*\author runrunrun1994
*\date 2020-11-27 
*****************************************/

#ifndef  _UTILS_H_
#define  _UTILS_H_
#include <cuda_runtime.h>

#define CHECK(call) do {             \
    const cudaError_t error = call;  \
    if (error != cudaSuccess) {      \
        printf("\e[0;31m ERROR: %s:%d\033[0m", __FILE__, __LINE__);         \
        printf(" code: %d, reason:%s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    }            \
} while(0)

#endif
