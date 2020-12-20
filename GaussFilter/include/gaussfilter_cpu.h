/***********************************************
*\file gaussfilter_cpu.h
*\brief cpu版本的高斯滤波实现,只支持RGB图
*\author runrunrun1994
*\date 2020-12-20 
************************************************/

#ifndef __GAUSSFILTER_CPU_H__
#define __GAUSSFILTER_CPU_H__

#include <opencv2/opencv.hpp>

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

/// \brief 生成整数形式的高斯核权重
/// \param r 高斯核半径
/// \param gaussMask 用来保存高斯核权重的数组
/// \param sum 高斯核权重的和
void gaussmaskV2_cpu(const int r, double sigma, int* gaussMask, int& sum);

/// \brief 使用整数形式计算，最后进行归一化
/// \param src 原始图像
/// \param dst 进过高斯滤波的结果图
/// \param radius 高斯核半径
/// \param sigma 标准差
void gaussfilterV2_cpu(cv::Mat& src, cv::Mat& dst, const int radius, double sigma);

/// \brief 生成整数形式的高斯核权重,1D
/// \param r 高斯核半径
/// \param gaussMask 用来保存高斯核权重的数组
/// \param sum 高斯核权重的和
void gaussmaskV3_cpu(const int r, double sigma, int* gaussMask, int& sum);

/// \brief 使用整数形式计算，最后进行归一化,采用行列分拆
/// \param src 原始图像
/// \param dst 进过高斯滤波的结果图
/// \param radius 高斯核半径
/// \param sigma 标准差
void gaussfilterV3_cpu(cv::Mat& src, cv::Mat& dst, const int radius, double sigma);
#endif