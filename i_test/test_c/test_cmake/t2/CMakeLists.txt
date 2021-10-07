
cmake_minimum_required(VERSION 3.7)

project(HELLO)
add_subdirectory(src)

## need : cmake -DCMAKE_INTALL_PREFIX=/tmp/usr/ ..
## default value of CMAKE_INSTALL_PREFIX is : /usr/local
#INSTALL(FILES COPYRIGHT README 
#	DESTINATION share/doc/cmake/t2)
#INSTALL(PROGRAMS runhello.sh 
#	DESTINATION bin)
#INSTALL(DIRECTORY doc/ 
#	DESTINATION share/doc/cmake/t2)
#set (src_list main.cpp)
#message(STATUS "this is binary dir" ${PROJECT_BINARY_DIR})
#message(STATUS "this is source dir" ${PROJECT_SOURCE_DIR})
#add_executable(run ${src_list})
