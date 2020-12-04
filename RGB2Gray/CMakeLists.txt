project(RGB2GRAY)

cmake_minimum_required(VERSION 3.5)
set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++11")

find_package(CUDA REQUIRED)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../bin)

set(OPENCV_LIBS "")
if(OpenCV_FOUND)
    message(STATUS "OpenCV3 Found (${OpenCV_CONFIG_PATH})")
    list(APPEND OPENCV_LIBS ${OpenCV_LIBS})
endif()
if(NOT OpenCV_FOUND)
    # OpenCV 2
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
    message(STATUS "OpenCV2 Found (${OpenCV_CONFIG_PATH})")
    list(APPEND OPENCV_LIBS ${OpenCV_LIBS})
endif()
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV libs: (${OPENCV_LIBS})")
else()
    message(WARNING "Not find OpenCV. project maybe invalid ")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_LIST)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/test TEST_LIST)

cuda_add_executable(rgb2gray ${TEST_LIST} ${SRC_LIST} ${GPU_SRC_LIST})
target_link_libraries(rgb2gray ${OPENCV_LIBS})