cmake_minimum_required(VERSION 3.6)
project(Sample_Atlas)

# Compile options
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
SET(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
add_compile_options(-std=c++11)
add_compile_options(-DIS_RC)
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
include_directories(
        ./inc/
        $ENV{DDK_PATH}/include/inc/
        $ENV{DDK_PATH}/include/third_party/protobuf/include
        $ENV{DDK_PATH}/include/third_party/cereal/include
        $ENV{DDK_PATH}/include/libc_sec/include
        $ENV{DDK_PATH}/include/third_party/opencv/include
)
link_directories(
        ./lib/
        $ENV{NPU_HOST_LIB} $ENV{HOME}/ascend_ddk/host/lib
        /home/HwHiAiUser/HIAI_PROJECTS/ascend_lib
        $ENV{HOME}/ascend_ddk/device/lib)
add_executable(main src/main.cpp inc/Main.h)
#target_link_libraries(main AtlasApi
#        matrixdaemon hiai_common hiai_common media_mini opencv_world)

target_link_libraries(main matrixdaemon AtlasApi pthread )

