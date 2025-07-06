cmake_minimum_required (VERSION 3.12)
project (MACHINE_LEARNING LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Add source to this project's executable.
add_executable (NeuralNetwork "NeuralNetwork.cpp" "NeuralNetwork.h" "Headers/SelfAttention.h" "Headers/TextProcessing.h" "Headers/SelfAttention.cpp" "Headers/TextProcessing.cpp"  "Headers/LSTM.h" "Headers/LSTM.cpp")


