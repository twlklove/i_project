
project(HELLO)
set (src_list main.cpp)
message(STATUS "this is binary dir" ${PROJECT_BINARY_DIR})
message(STATUS "this is source dir" ${PROJECT_SOURCE_DIR})
add_executable(run ${src_list})
