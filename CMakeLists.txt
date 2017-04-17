cmake_minimum_required(VERSION 3.1)
project (OPENCL_IN_ACTION)

set(OpenCL_LIBRARY_PATH /opt/amdgpu-pro/lib/x86_64-linux-gnu/)
link_directories(${OpenCL_LIBRARY_PATH})
set(OpenCL_LIBRARY OpenCL)

add_executable(matvec matvec.c)
target_include_directories(matvec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(matvec ${OpenCL_LIBRARY})

configure_file(${CMAKE_SOURCE_DIR}/kernels/matvec.cl ${CMAKE_CURRENT_BINARY_DIR}/kernels/matvec.cl COPYONLY)
