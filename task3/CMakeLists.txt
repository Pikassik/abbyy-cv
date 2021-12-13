cmake_minimum_required(VERSION 3.21)
project(task3)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
#SET(CMAKE_CXX_FLAGS  "-O2")

add_executable(task3 main.cpp)
target_link_libraries(task3 ${OpenCV_LIBS})