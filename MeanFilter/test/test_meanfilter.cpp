#include "meanfilter_cpu.h"
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

    cv::Mat dst = cv::Mat(image.rows, image.cols, image.type());
    meanfilter_cpu(image, dst, 2);

    cv::imwrite("./meanfilter_cpu.jpg", dst);
    cv::waitKey(0);

    return 0;
}