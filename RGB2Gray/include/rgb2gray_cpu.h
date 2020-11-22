/****************************************
*\file rgb2gray_cpu.h
*\brief CPU上RGB彩色图转灰度图
*\author runrunrun1994
*\date 2020-11-21 
****************************************/

#ifndef _RGB_GRAY_CPU_H_
#define _RGB_GRAY_CPU_H_

#include <opencv2/opencv.hpp>

/// \brief 图像灰度化
/// \param src 彩色图
/// \param gray 灰度结果图
void rgb2gray_cpu(cv::Mat& src, cv::Mat& gray);

#endif