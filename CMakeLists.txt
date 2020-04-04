cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD 11)

project(opticalflow)

find_package( OpenCV REQUIRED )
set(SOURCE_FILES residual.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

#add_executable(opticalflow main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)

