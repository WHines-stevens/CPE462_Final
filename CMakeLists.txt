cmake_minimum_required(VERSION 3.10)
project(ImageRestoration)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your source files
add_executable(ImageRestoration
    main.cpp
    utils.cpp
    filters.cpp
)

# Link OpenCV libraries
target_link_libraries(ImageRestoration ${OpenCV_LIBS})
