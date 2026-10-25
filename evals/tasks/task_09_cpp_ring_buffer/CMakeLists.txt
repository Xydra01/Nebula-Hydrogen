cmake_minimum_required(VERSION 3.20)
project(task_09_atomic_ring CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()
add_executable(test_atomic_ring tests/test_atomic_ring.cpp)
target_include_directories(test_atomic_ring PRIVATE include)
add_test(NAME test_atomic_ring COMMAND test_atomic_ring)
