cmake_minimum_required (VERSION 3.12)
project (MACHINE_LEARNING LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include_directories(${CMAKE_SOURCE_DIR}/Headers)
# Add source to this project's executable.
add_executable (NeuralNetwork "NeuralNetwork.cpp" "Headers/SelfAttention.cpp" "Headers/TextProcessing.cpp")

