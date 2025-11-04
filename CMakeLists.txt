cmake_minimum_required(VERSION 3.22.1)
project(MPI C)

set(CMAKE_C_STANDARD 11)

add_executable(MPI main.c
        tasks/pi.c
        tasks/matrix_multiplication.c
        tasks/cannon.c
)

target_include_directories(MPI PRIVATE ${CMAKE_SOURCE_DIR}/tasks)