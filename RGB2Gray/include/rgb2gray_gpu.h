#ifndef _RGB2GRAY_GPU_H_
#define _RGB2GRAY_GPU_H_
#include <cuda_runtime.h>

///\brief GPU RGB转Gray
///\param input  输入图像RGB三通道
///\param output 输出图像灰度图
///\param imgH   高
///\param imgW   宽
__global__ void rgb2gray_gpu(uchar3* input, unsigned char* output, const int imgH, const int imgW);

#endif