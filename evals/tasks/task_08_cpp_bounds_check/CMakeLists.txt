cmake_minimum_required(VERSION 3.20)
project(task_08_buffer_view CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(buffer_view src/buffer_view.cpp)
target_include_directories(buffer_view PUBLIC include)

enable_testing()
add_executable(test_buffer_view tests/test_buffer_view.cpp)
target_link_libraries(test_buffer_view PRIVATE buffer_view)
add_test(NAME test_buffer_view COMMAND test_buffer_view)
