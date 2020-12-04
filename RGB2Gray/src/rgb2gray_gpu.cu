#include "rgb2gray_gpu.h"

__global__ void rgb2gray_gpu(uchar3* input, unsigned char* output, const int imgH, const int imgW){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    //gray = 0.299*R + 0.578*G + 0.114 * B
    if ((imgH > x) && (imgW > y)){
        uchar3 bgr = input[y * imgW + x];
        output[y * imgW + x] = 0.114f * bgr.x + 0.578f * bgr.y + 0.299f * bgr.z;
    }
}