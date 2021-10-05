cmake_minimum_required(VERSION 3.21)
project(task1)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

add_executable(task1 main.cpp)
target_link_libraries(task1 ${OpenCV_LIBS})