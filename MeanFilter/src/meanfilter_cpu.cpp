#include "meanfilter_cpu.h"
#include <iostream>

void meanfilter_cpu(cv::Mat& src, cv::Mat& dst, const int radius){
    if (src.empty()){
        std::cout << "The RGB image is invalid!\n";

        return ;
    }

    const int W = src.cols;
    const int H = src.rows;
    int sumR, sumB, sumG;
    const int M = (radius*2 + 1)*(radius*2 + 1);
    const int offset = src.step - W * 3;  //图片可能对齐了

    uchar* pSrc = src.data;
    for (int row = 0; row < H; ++row){
        uchar* pDst = dst.data + row * dst.step;

        for (int col = 0; col < W; ++col){
            sumR = 0;
            sumG = 0;
            sumB = 0;

            for (int i = -radius; i <= radius; ++i){
                for (int j = -radius; j <= radius; ++j){
                    int ny = CLIP3(row + i, 0, H-1);
                    int nx = CLIP3(col + j, 0, W-1);
                    int pos = nx * 3 + ny * src.step;
                    sumB += pSrc[pos + 0];
                    sumG += pSrc[pos + 1];
                    sumR += pSrc[pos + 2];
                }
            }

            pDst[0] = sumB / M;
            pDst[1] = sumG / M;
            pDst[2] = sumR / M;
            pDst += 3;
        }
        pDst += offset;
    }

}