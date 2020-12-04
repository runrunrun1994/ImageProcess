#include "utils.h"
#include "rgb2gray_gpu.h"
#include "rgb2gray_cpu.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    if (argc < 1)
    {
        std::cout << "Too fewer parameter！\n";
        exit(-1);
    }

    cv::Mat image = cv::imread(argv[1]);

    if (image.empty())
    {
        std::cout << "The image is invalid!" << std::endl;
        exit(-1);
    }

    cv::Mat gray = cv::Mat(image.rows, image.cols, CV_8UC1);
    rgb2gray_cpu(image, gray);

    cv::imwrite("./gray_cpu.jpg", gray);
    //cv::imshow("src", image);
    //cv::imshow("gray_cpu", gray);

    //GPU 
    int imgH = image.rows;
    int imgW = image.cols;
    cv::Mat grayGpu = cv::Mat(image.rows, image.cols, CV_8UC1);
    uchar3* devInput;
    unsigned char* devOutput;

    CHECK(cudaMalloc((void**)&devInput, imgH*imgW*sizeof(uchar3)));
    CHECK(cudaMalloc((void**)&devOutput, imgH*imgW*sizeof(unsigned char)));
    CHECK(cudaMemcpy(devInput, image.data, imgH*imgW*sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 blockPerGrid((imgW + threadsPerBlock.x -1) / threadsPerBlock.x , (imgH + threadsPerBlock.y -1) / threadsPerBlock.y);

    rgb2gray_gpu <<<blockPerGrid, threadsPerBlock>>> (devInput, devOutput, imgH, imgW);

    CHECK(cudaMemcpy(grayGpu.data, devOutput, imgW * imgH * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cv::imwrite("./gray_gpu.jpg", grayGpu);

    cv::waitKey(0);

    return 0;
}