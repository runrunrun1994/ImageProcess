/***********************************************
*\file gaussfilter_cpu.h
*\brief cpu版本的高斯滤波实现,只支持RGB图
*\author runrunrun1994
*\date 2020-12-20 
************************************************/

#ifndef __GAUSSFILTER_CPU_H__
#define __GAUSSFILTER_CPU_H__

#include <opencv2/opencv.hpp>

#define MIN(a, b) ((a) < (b)?(a):(b))
#define MAX(a, b) ((a) > (b)?(a):(b))
#define CLIP3(a, b, c) MIN(MAX(a, b), c)

/// \brief 生成高斯核
/// \param r 高斯核的半径
/// \param sigma 标准差
/// \param gaussMask 用来保存高斯核的结果
void gaussmask_cpu(const int r, double sigma, double* gaussMask);

/// \brief 高斯滤波
/// \param src RGB原始图像
/// \param dst 经过高斯滤波之后的图像
/// \param radius 高斯核半径
/// \param sigma 标准差
void gaussfilter_cpu(cv::Mat& src, cv::Mat& dst, const int radius, double sigma);
#endif