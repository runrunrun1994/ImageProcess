/***********************************************
*\file meanfilter_cpu.h
*\brief cpu版本的均值滤波实现
*\author runrunrun1994
*\date 2020-12-8 
************************************************/

#ifndef __MEAN_FILTER_H__
#define __MEAN_FILTER_H__
#include <opencv2/opencv.hpp>

#define MIN(a, b) ((a) < (b)?(a):(b))
#define MAX(a, b) ((a) > (b)?(a):(b))
#define CLIP3(a, b, c) MIN(MAX(a, b), c)

///< \brief 均值滤波简单实现
///< \param src 输入原始图像
///< \param dst 结果图
///< \param radius 均值滤波核大小
///< \param stride 步长
void meanfilter_cpu(cv::Mat& src, cv::Mat& dst,const int radius);

#endif