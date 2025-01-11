#include "TextProcessing.h"
#include <sstream>



//loading the files take a signrfrcrnt amount of space
void userInput::loadGloveVectors(const std::string& gloveFile) {
	std::cout << "\n \n \t LOading Flies wait Might take a siginificant amount of time & space\n\n\t\t";
	std::ifstream file(gloveFile);
	if (!file.is_open()) {
		std::cerr << "\t !Could not open GloVe file!" << std::endl;
		return;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string word;
		iss >> word;  // First token is the word

		std::vector<double> vector;
		double value;
		while (iss >> value) {
			vector.push_back(value);  // Remaining tokens are the vector values
		}

		glove_vectors[word] = vector;
	}

	std::cout << "Loaded GloVe vectors: " << glove_vectors.size() << std::endl;
	file.close();
}





// converting the sentence to tockens 
std::vector<std::string> userInput::tokenize(std::string& UInput) {
	std::vector<std::string> tockens;
	std::string tocken; 					
	std::stringstream word(UInput); // converting the string to stream
	while (word >> tocken){			// dividing words with incoming stream
		tockens.push_back(tocken);	// pushing the word to vector
		size++;
	}
	tockens.push_back("<EOS>");
	std::cout << "\n\n\tsize array ->> " << size;
	return tockens;
}


int userInput::Size() {
	return size;
}



// generating random vector for token
std::vector<double> userInput::RandomVectorGenerator(int size) {
	std::vector<double> GenetreatedVector(size);  // vector array size of total no. of elements in text


	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	
	for (int i = 0; i < size; i++) {
		GenetreatedVector[i] = distribution(generator);
	}
	
	return GenetreatedVector;
}

void userInput::addEOS(std::unordered_map<std::string, std::vector<double>>& gloveVectors, int vectorSize) {
	userInput EOSobj;
	// Add <EOS> with a random or zero vector
	std::vector<double> eosVector = EOSobj.RandomVectorGenerator(vectorSize); // or std::vector<double>(vectorSize, 0.0);
	gloveVectors["<EOS>"] = eosVector;
}


//std::vector<std::vector<double>> userInput::convertTokensToVectors(const std::vector<std::string>& tokens) {
	//std::vector<std::vector<double>> vectors;

	//for (const auto& token : tokens) {
		//if (token_vectors.find(token) == token_vectors.end()) {
			// Generate a random vector for a new token
			//token_vectors[token] = RandomVectorGenerator();
		//}
	//	vectors.push_back(token_vectors[token]);
	//}

	//return vectors;
//}




/*std::vector<std::vector<double>> userInput::convertTokensToVectors(const std::vector<std::string>& tokens) {
	std::vector<std::vector<double>> IndependentVectors;
	for (auto token : tokens) {
		auto NewGeneratedVector = RandomVectorGenerator();
		IndependentVectors.push_back(NewGeneratedVector);
	}
	return IndependentVectors;
}*/



std::vector<std::vector<double>> userInput::convertTokensToVectors(const std::vector<std::string>& tokens) {
	std::vector<std::vector<double>> IndependentVectors;

	std::cout << "Sample GloVe vectors loaded:\n";
	for (const auto& pair : glove_vectors) {
		std::cout <<"  Vector size: " << pair.second.size() << "\n";
		break; // Only print the first word to avoid flooding the console
	}

	for (const auto& token : tokens) {
		if (glove_vectors.find(token) != glove_vectors.end()) {
			// Use the pre-trained GloVe vector
			IndependentVectors.push_back(glove_vectors[token]);
		}
		else {
			// Fall back to generating a random vector if word not found in GloVe
			auto NewGeneratedVector = RandomVectorGenerator(50);
			IndependentVectors.push_back(NewGeneratedVector);
		}
	}
	return IndependentVectors;
}



std::vector<double> userInput::averagePooling(const std::vector<std::vector<double>>& vectors) {
	std::vector<double> avg_vector(vectors[0].size(), 0.0);
	for (const auto& vec : vectors) {
		for (size_t i = 0; i < vec.size(); i++) {
			avg_vector[i] += vec[i];
		}
	}
	for (auto& val : avg_vector) {
		val /= vectors.size();
	}
	return avg_vector;
}



//word embadding-> output
double cosineSimilarity(const std::vector<double>& vec1, const std::vector<double>& vec2,
	const std::unordered_map<std::string, int>& wordUsage,
	const std::string& candidateWord,
	double temperature) {
	// Step 1: Compute dot product and norms
	double dotProduct = 0.0, normA = 0.0, normB = 0.0;
	for (size_t i = 0; i < vec1.size(); ++i) {
		dotProduct += vec1[i] * vec2[i];
		normA += vec1[i] * vec1[i];
		normB += vec2[i] * vec2[i];
	}
	if (normA == 0 || normB == 0) return 0; // Avoid division by zero

	// Step 2: Compute raw cosine similarity
	double similarity = dotProduct / (sqrt(normA) * sqrt(normB));

	// Step 3: Apply threshold filtering
	if (similarity < COSINE_THRESHOLD) return 0; // Filter out low scores

	// Step 4: Apply repetition penalty (if the word has been used before)
	if (wordUsage.count(candidateWord)) {
		int usageCount = wordUsage.at(candidateWord);
		similarity -= usageCount * REPETITION_PENALTY; // Penalize based on usage
		if (similarity < 0) similarity = 0; // Ensure non-negative similarity
	}

	// Step 5: Apply temperature scaling (optional, for randomness)
	if (temperature > 0) {
		similarity = pow(similarity, 1.0 / temperature);
	}

	return similarity;
}





std::vector<std::string> mapOutputToWords(
	const std::unordered_map<std::string, std::vector<double>>& gloveVectors,
	const std::vector<double>& outputVector,
	int topN,
	const std::unordered_map<std::string, int>& wordUsage, // Optional
	double temperature) {

	std::vector<std::pair<std::string, double>> similarityScores;

	// Debug: Limit vocabulary size
	int debugVocabularyLimit = 5000;
	int counter = 0;

	for (const auto& [word, vector] : gloveVectors) {
		if (counter++ > debugVocabularyLimit) break; // Stop after 5000 words

		// Use the updated cosineSimilarity function
		double similarity = cosineSimilarity(outputVector, vector, wordUsage, word, temperature);

		if (similarity > 0) { // Skip words that are filtered by the threshold
			similarityScores.push_back({ word, similarity });
		}
	}

	// Sort words by similarity in descending order
	std::sort(similarityScores.begin(), similarityScores.end(), [](const auto& a, const auto& b) {
		return a.second > b.second;
		});

	// Collect the top N words
	std::vector<std::string> topWords;
	for (int i = 0; i < std::min(topN, (int)similarityScores.size()); ++i) {
		topWords.push_back(similarityScores[i].first);
	}

	return topWords;
}



std::string generateSentence(
	const std::unordered_map<std::string, std::vector<double>>& gloveVectors,
	const std::vector<double>& initialVector,
	int maxLength, 
	int topN,      
	double temperature
) {
	std::vector<std::string> generatedWords;
	std::vector<double> currentVector = initialVector;
	std::unordered_map<std::string, int> wordUsage; // Track word frequencies
	std::string eos = "<EOS>"; // Define the EOS token

	// Random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	for (int i = 0; i < maxLength; ++i) {
		// Map output vector to top-N words with updated logic
		std::vector<std::string> topWords = mapOutputToWords(
			gloveVectors,
			currentVector,
			topN,
			wordUsage,
			temperature
		);

		if (topWords.empty()) {
			std::cerr << "No suitable words found at step " << i << ".\n";
			break;
		}

		// Randomly pick one from the top-N words
		std::uniform_int_distribution<> dis(0, topWords.size() - 1);
		std::string selectedWord = topWords[dis(gen)];

		// Stop if EOS is generated
		if (selectedWord == eos) {
			break;
		}

		// Append the word to the output sentence
		generatedWords.push_back(selectedWord);

		// Update the word usage map
		wordUsage[selectedWord]++;

		// Update the vector using the selected word
		std::vector<double> nextVector = gloveVectors.at(selectedWord);
		std::transform(
			currentVector.begin(), currentVector.end(),
			nextVector.begin(),
			currentVector.begin(),
			[](double a, double b) { return 0.7 * a + 0.3 * b; }
		);

		// Debug: Print progress
		std::cout << "Step " << i << ": Selected Word = " << selectedWord << "\n";
		std::cout << "Generated Words So Far: ";
		for (const auto& word : generatedWords) {
			std::cout << word << " ";
		}
		std::cout << "\n";
	}

	// Assemble the final sentence
	return assemblePhrase(generatedWords);
}





std::string assemblePhrase(const std::vector<std::string>& words) {
	std::string phrase;
	for (const auto& word : words) {
		phrase += word + " ";
	}
	return phrase;
}