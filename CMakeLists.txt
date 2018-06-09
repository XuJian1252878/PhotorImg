# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.9) # 指定运行此配置文件所需的 CMake 的最低版本；
# 项目信息
project(ImageRegistration) # project：参数值是 ImageRegistration，该命令表示项目的名称是 ImageRegistration 。

# opencv的配置
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

# 指定生成目标
add_executable(ImageRegistration main.cpp StarImageRegistBuilder.h StarImage.h StarImagePart.h StarImage.cpp StarImagePart.cpp StarImageRegistBuilder.cpp Util.h Util.cpp) #  将名为 main.cpp 的源文件编译成一个名称为 ImageRegistration 的可执行文件。
# 让 ImageRegistration 可以执行的文件去连接OpenCV的动态库
target_link_libraries(ImageRegistration ${OpenCV_LIBS})