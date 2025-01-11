#include "SelfAttention.h"


// multiplying all the array of the word to every other word
// ex -> matrix [11,12,21,22] 
//will be m(11) is multiplication of first(1st) word independent vector to itself
// m(12) will be the multiplication of 1st word independent vector to second(2nd) independent vector array -> and so on



//exited with code - 1073740791 (0xc0000409). stack over flow memory access out of bound.
std::vector<std::vector<double>> selfAttention(std::vector<double> vector) {
	std::vector<std::vector<double>> returnVector;
	for (size_t i = 0; i < vector.size(); i++) {
		std::vector<double> Processing(vector.size(), 0.0);
		for (size_t j = 0; j < vector.size(); j++) {
			Processing[j] = vector[i] * vector[j];
		}
		returnVector.push_back(Processing);
	}
	return returnVector;
}





// for normilization of the matrix
std::vector<std::vector<double>> Softmax(std::vector<std::vector<double>> MainVector, int size){
	double** Temp = new double* [size];
	for (int i = 0; i < size; i++) {
		Temp[i] = new double[size];
		for (int j = 0; j < size; j++) {
			Temp[i][j] = 0;
		}
	}

	std::vector<std::vector<double>> NormalizedVector;
	
	double ExpSum=0;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (Temp[j][i] == 0) {
				Temp[i][j] = std::exp(MainVector[i][j]);
			}
			else {
				Temp[i][j] = Temp[j][i];
			}
			ExpSum += Temp[i][j];
		}
	}

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			MainVector[i][j] = Temp[i][j]/ExpSum;
		}
	}

	return MainVector;
}





