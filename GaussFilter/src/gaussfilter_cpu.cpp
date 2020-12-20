#include "gaussfilter_cpu.h"
#include <cmath>
#include <iostream>

void gaussmask_cpu(const int r, double sigma, double* gaussMask)
{
    if (r <= 0)
        return ;
    if (gaussMask == nullptr)
        return ;

    double sum = 0.0;
    const double PI = 3.1415926;
    const int stride = 2 * r + 1;
    for (int i = -r, h = 0; i <=r; ++i, ++h)
    {
        for (int j = -r, w = 0; j <=r; ++j, ++w)
        {
            gaussMask[h * stride + w] = exp((-(i*i + j*j))/(2.0*sigma*sigma));
            sum += gaussMask[h * stride + w];
        }
    }

    for (int i = 0; i < stride*stride; ++i)
    {
         gaussMask[i] /= sum;
         //std::cout << gaussMask[i] << std::endl;
    }
}

void gaussfilter_cpu(cv::Mat& src, cv::Mat& dst, const int radius, double sigma)
{
    if (src.empty() || radius <= 0)
        return ;
    
    const int H = src.rows;
    const int W = src.cols;
    const int offset = src.step - 3 * W;
    const int stride = 2*radius + 1;

    double* pSigMask = new double[stride*stride];
    gaussmask_cpu(radius, sigma, pSigMask);

    uchar* pSrc = src.data;
    double sumB = 0.0;
    double sumG = 0.0;
    double sumR = 0.0;
    int nX = 0;
    int nY = 0;
    int pos = 0;

    for (int row = 0; row < H; ++row)
    {
        uchar* pDst = dst.data + row * dst.step;
        for (int col = 0; col < W; ++col)
        {
            sumB = 0.0;
            sumG = 0.0;
            sumR = 0.0;
            for (int i = -radius, krow=0; i <= radius; ++i, ++krow)
            {
                for (int j = -radius, kcol=0; j <= radius; ++j, ++kcol)
                {
                    nX = col + j;
                    nY = row + i;
                    if (nX < 0 || nX > W)
                        nX = col;
                    if (nY < 0 || nY > H)
                        nY = row;
                    pos = nY * src.step + nX * 3;

                    sumB += pSrc[pos] * pSigMask[krow * stride + kcol];
                    sumG += pSrc[pos+1] * pSigMask[krow * stride + kcol];
                    sumR += pSrc[pos+2] * pSigMask[krow * stride + kcol];
                    //std::cout << pSigMask[krow * stride + kcol] << " ";
                }
            }
        
            pDst[0] = (int)(sumB);
            pDst[1] = (int)(sumG);
            pDst[2] = (int)(sumR);
            pDst += 3;
        }
    }
}