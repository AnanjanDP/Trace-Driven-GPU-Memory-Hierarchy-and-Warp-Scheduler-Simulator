cmake_minimum_required(VERSION 3.16)
project(gpu_memory_simulator VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler Warnings
if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Core simulator library
add_library(sim_core STATIC
    src/config.cpp
    src/cache.cpp
    src/coalescer.cpp
    src/trace_reader.cpp
    src/scheduler.cpp
    src/dram_model.cpp
    src/memory_hierarchy.cpp
    src/simulator.cpp
)

target_include_directories(sim_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Simulator CLI executable
add_executable(gpu_simulator
    src/main.cpp
)

target_link_libraries(gpu_simulator PRIVATE
    sim_core
)

# Unit & Integration Tests
enable_testing()

add_executable(gpu_simulator_tests
    tests/test_main.cpp
    tests/test_config.cpp
    tests/test_cache.cpp
    tests/test_coalescer.cpp
    tests/test_scheduler.cpp
    tests/test_simulator.cpp
    tests/test_validation.cpp
)

target_include_directories(gpu_simulator_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)

target_link_libraries(gpu_simulator_tests PRIVATE
    sim_core
)

add_test(NAME AllUnitTests COMMAND gpu_simulator_tests)
set_tests_properties(AllUnitTests PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")


