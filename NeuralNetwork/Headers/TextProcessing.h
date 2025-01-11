#ifndef TextProcessing_H
#define TextProcessing_H

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include "SelfAttention.h"


constexpr double COSINE_THRESHOLD = 0.7; // Minimum similarity threshold
constexpr double REPETITION_PENALTY = 0.1; // Penalty factor for repeated words

class userInput {
	public:
		std::vector<std::string> tokenize(std::string& UInput);
		std::vector<std::vector<double>> convertTokensToVectors(const std::vector<std::string>& tokens);
		int size = 0;

		//the loaded vector -> 
		std::unordered_map<std::string, std::vector<double>> glove_vectors;

		//loading the pretrainned data to the memory for the execution
		void loadGloveVectors(const std::string& gloveFile);
		std::vector<double> averagePooling(const std::vector<std::vector<double>>& vectors);
		std::vector<double> RandomVectorGenerator(int size);
		void addEOS(std::unordered_map<std::string, std::vector<double>>& gloveVectors, int vectorSize);

	private:
		std::default_random_engine generator;
		std::unordered_map<std::string, std::vector<double>> token_vectors;
		int Size();
	
};


//mapping the output to actual value
//the calculator
double cosineSimilarity(const std::vector<double>& vec1, const std::vector<double>& vec2,
	const std::unordered_map<std::string, int>& wordUsage,
	const std::string& candidateWord,
	double temperature = 1.0);

std::vector<std::string> mapOutputToWords(
	const std::unordered_map<std::string, std::vector<double>>& gloveVectors,
	const std::vector<double>& outputVector,
	int topN,
	const std::unordered_map<std::string, int>& wordUsage = {}, // Optional
	double temperature = 1.0);


std::string generateSentence(
	const std::unordered_map<std::string, std::vector<double>>& gloveVectors,
	const std::vector<double>& initialVector,
	int maxLength = 20,  // Max words in the sentence
	int topN = 3 ,       // Consider top N words for diversity
	double temperature = 1.0
);


std::string assemblePhrase(const std::vector<std::string>& words);

#endif
