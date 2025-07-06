cmake_minimum_required (VERSION 3.12)
project (MACHINE LEARNING)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Add source to this project's executable.
add_executable (NeuralNetwork "NeuralNetwork.cpp" "NeuralNetwork.h" "Headers/SelfAttention.h" "Headers/TextProcessing.h" "Headers/SelfAttention.cpp" "Headers/TextProcessing.cpp"  "Headers/LSTM.h" "Headers/LSTM.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET NeuralNetwork PROPERTY CXX_STANDARD 20)
endif()

