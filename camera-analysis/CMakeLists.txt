cmake_minimum_required(VERSION 3.5)

project(camera-analysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fstack-protector-strong ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=array-bounds -Werror=sizeof-pointer-div -Werror=sign-compare ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=format -Werror=missing-field-initializers -Werror=return-type")

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(RunningStats REQUIRED)
include_directories(${RUNNINGSTATS_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS system filesystem iostreams)

add_executable(camera-analysis main.cpp)

target_link_libraries(camera-analysis
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${RUNNINGSTATS_LIBRARIES}
    ceres
    raw
)

install(TARGETS camera-analysis
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
