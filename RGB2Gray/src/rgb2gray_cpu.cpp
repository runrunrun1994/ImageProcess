#include "rgb2gray_cpu.h"
#include <iostream>

void rgb2gray_cpu(cv::Mat& src, cv::Mat& gray)
{
    //gray = 0.299*R + 0.578*G + 0.114 * B
    if (src.empty())
    {
        std::cout << "The RGB image is invalid!\n";
        return;        
    }

    int w = src.cols;
    int h = src.rows;

    for (int row = 0; row < h; row++)
    {
        uchar* pSrc = src.data + row * src.step;
        uchar* pGray = gray.ptr<uchar>(row);

        for (int col = 0; col < w; col++)
        {
            pGray[col] = (uchar)(0.114f * pSrc[0] + 0.578 * pSrc[1] + 0.299 * pSrc[2]);
            pSrc += 3;
        }
    }
}