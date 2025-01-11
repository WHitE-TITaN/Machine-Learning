#ifndef SelfAttention_H
#define SelfAttention_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "TextProcessing.h"
#include <unordered_map>

// making context awair / multiplying the matrix
std::vector<std::vector<double>> selfAttention(std::vector<double> vector);



//softmax normilization of matrix or self awair.
std::vector<std::vector<double>>Softmax(std::vector<std::vector<double>> MainVector, int size);

//std::vector<double> averagePooling(const std::vector<std::vector<double>>& vectors);




#endif