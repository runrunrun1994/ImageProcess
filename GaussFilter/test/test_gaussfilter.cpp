#include "gaussfilter_cpu.h"
#include <opencv2/opencv.hpp>
#include <sys/time.h>

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
    struct timeval tp;
    struct timeval tp1;
    int start;
    int end;

    gettimeofday(&tp, NULL);
    start = tp.tv_sec*1000 + tp.tv_usec/1000;

    gaussfilter_cpu(image, dst, 4, 1.2);

    gettimeofday(&tp1, NULL);
    end = tp1.tv_sec * 1000 + tp1.tv_usec/1000;

    std::cout <<"gaussfilter: " << (end -start) << std::endl;

    gettimeofday(&tp, NULL);
    start = tp.tv_sec*1000 + tp.tv_usec/1000;

    gaussfilterV2_cpu(image, dst, 4, 1.2);

    gettimeofday(&tp1, NULL);
    end = tp1.tv_sec * 1000 + tp1.tv_usec/1000;

    std::cout <<"gaussfilterV2: " << (end -start) << std::endl;

    gettimeofday(&tp, NULL);
    start = tp.tv_sec*1000 + tp.tv_usec/1000;
    cv::GaussianBlur(image, dst, cv::Size(9, 9), 1.2, 1.2);

    gettimeofday(&tp1, NULL);
    end = tp1.tv_sec * 1000 + tp1.tv_usec/1000;

    std::cout <<"opencv: " << (end -start) << std::endl;

    gettimeofday(&tp, NULL);
    start = tp.tv_sec*1000 + tp.tv_usec/1000;
    gaussfilterV3_cpu(image, dst, 4, 1.2);

    gettimeofday(&tp1, NULL);
    end = tp1.tv_sec * 1000 + tp1.tv_usec/1000;

    std::cout <<"gaussfilterV3: " << (end -start) << std::endl;

    cv::imwrite("./gaussfilterV3_cpu.jpg", dst);
    cv::waitKey(0);

    return 0;
}