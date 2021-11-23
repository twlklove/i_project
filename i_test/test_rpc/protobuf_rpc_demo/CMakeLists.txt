PROJECT(protobuf-rpc-demp)
cmake_minimum_required(VERSION 3.2)
add_compile_options(-std=c++11)

include_directories(
    ./src
    /root/.local/include
    /root/local/include)
link_directories(
    /root/.local/lib
    ./build)

set(LIBMYRPC_SRC ./src/rpc_meta.pb.cc)
add_library(myrpc ${LIBMYRPC_SRC})
add_executable(server ./src/echo.pb.cc ./src/test_myrpc.cpp)
target_link_libraries(server libprotobuf.a libboost_system.a myrpc pthread)
add_executable(client ./src/echo.pb.cc ./src/test_myrpc_client.cpp)
target_link_libraries(client libprotobuf.a libboost_system.a myrpc pthread)
