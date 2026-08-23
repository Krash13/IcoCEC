cmake_minimum_required(VERSION 3.16)
project(ico_cec2017 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_executable(ico_cec2017
    meta_optimization.cpp
    cec17_test_func.c
)

target_link_libraries(ico_cec2017 PRIVATE m)

# input_data/ должна находиться рядом с исполняемым файлом во время запуска.
add_custom_command(TARGET ico_cec2017 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/input_data
        $<TARGET_FILE_DIR:ico_cec2017>/input_data
)
