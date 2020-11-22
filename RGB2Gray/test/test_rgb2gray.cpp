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

    cv::imwrite("./gray.jpg", gray);
    cv::imshow("src", image);
    cv::imshow("gray", gray);

    cv::waitKey(0);

    return 0;
}