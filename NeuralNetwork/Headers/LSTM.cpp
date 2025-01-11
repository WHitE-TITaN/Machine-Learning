#include "LSTM.h"

double longTermMemory =0 /* cell state */,
shortTermMemory=0, /* Hidden state */
learningRate = 0.99; /* learning rate */



//local file variable;
double LTMactivation;
double LSTMUpdateActivation;
double LSTMPotentailActivation;
double outputActivation;

//declaring layers to store the wights and biases 
layer longTermRemember;


layer percentageLTMPotential, potentialLTM, percentageSTMPotental;

double sigmoid_Activation(double entity_Vector) {
	//will map any x axis coordinate to its corosponding y axis coordiante
	//betwqeen 0-1;
	return 1 / (1 + exp(-entity_Vector));

}

double tanh_Activation(double entity_Vector) {
	//will map any x axis coordinate to its corosponding y axis coordiante
	//betwqeen 1 & -1;
	return (exp(entity_Vector) - exp(-entity_Vector)) / (exp(entity_Vector) + exp(-entity_Vector));
}



void forgetGate(double currentInputVector) {
	LTMactivation = (shortTermMemory * longTermRemember.shortTermWeight)
		+ (currentInputVector * longTermRemember.inputWeight) + longTermRemember.bias; //adding up the equations

	longTermMemory = longTermMemory * sigmoid_Activation(LTMactivation); // adding to long term memory
}


void inputGate(double currentInputVector) {
	//to calculate potential to add up to long term memory
	LSTMUpdateActivation = (shortTermMemory * potentialLTM.shortTermWeight)
		+ (currentInputVector * potentialLTM.inputWeight) + potentialLTM.bias; // adding up the values

	double potentialActivation = tanh_Activation(LSTMUpdateActivation);

	//to calculate the ammount of potential to be added
	LSTMPotentailActivation = (shortTermMemory * percentageLTMPotential.shortTermWeight)
		+ (currentInputVector * percentageLTMPotential.inputWeight) + percentageLTMPotential.bias;
	
	double percentagePotential = sigmoid_Activation(LSTMPotentailActivation);

	longTermMemory = longTermMemory + (percentagePotential * potentialActivation);
}


void outputGate(double currentInputVector) {
	//calculating the potential short term memory 
	double potentialSTM = tanh_Activation(longTermMemory);

	double outputActivation = (shortTermMemory * percentageSTMPotental.shortTermWeight)
		+ (currentInputVector * percentageSTMPotental.inputWeight) + percentageSTMPotental.bias; // adding up the values
	//calculate actual percentage to account for
	double percentagePotential = sigmoid_Activation(outputActivation);

	shortTermMemory = percentagePotential * potentialSTM; // update short term memory


}


//main flow or driver
std::vector<double> lstmMainFlow(std::vector<double> wordEmbedding) {
	//if the input is empty for backpropagation
	if (wordEmbedding.empty()) {
		std::cerr << "Error: Input vector is empty.\n";
		return std::vector<double>();
	}

	std::vector<double> producedOutPut;
	for (auto x : wordEmbedding) {
		forgetGate(x);
		inputGate(x);
		outputGate(x);
		producedOutPut.push_back(shortTermMemory);
	}
	return producedOutPut;
}


//backpropogation vectors 
void backPropForgetGate(double dLoss, double currentVector) {
	double dLoss_ForgetGate = dLoss * longTermMemory;//calculating grdient desent 
	double dloss_ForgetGateDerivative = sigmoid_Activation(LTMactivation) * (1 - sigmoid_Activation(LTMactivation));// deravitive of sigmoid function;

	longTermRemember.shortTermWeight -= learningRate * dLoss_ForgetGate * dloss_ForgetGateDerivative * shortTermMemory;
	longTermRemember.bias -= learningRate * dLoss_ForgetGate * dloss_ForgetGateDerivative;
	longTermRemember.inputWeight -= learningRate * dLoss_ForgetGate * dloss_ForgetGateDerivative * currentVector;
}

void backPropInputGate(double dloss, double currentVector) {
	double dLoss_InputGate = dloss * (1 - longTermMemory);//gradient desent of the input gate
	double dLoss_InputGateDerivative = sigmoid_Activation(LSTMUpdateActivation) * (1 - sigmoid_Activation(LSTMUpdateActivation));//calculating deravative

	//updating
	potentialLTM.bias -= learningRate * dLoss_InputGate * dLoss_InputGateDerivative;
	potentialLTM.inputWeight -= learningRate * dLoss_InputGate * dLoss_InputGateDerivative * currentVector;
	potentialLTM.shortTermWeight -= learningRate * dLoss_InputGate * dLoss_InputGateDerivative * shortTermMemory;

	double dLoss_dPercentagePotential = dloss * longTermMemory; // How loss changes wrt percentage potential
	double dPercentagePotential_dSigmoidInput = sigmoid_Activation(LSTMPotentailActivation) * (1 - sigmoid_Activation(LSTMPotentailActivation));

	//updating
	percentageLTMPotential.bias -= learningRate * dLoss_dPercentagePotential * dPercentagePotential_dSigmoidInput;
	percentageLTMPotential.inputWeight -= learningRate * dLoss_dPercentagePotential * dPercentagePotential_dSigmoidInput * currentVector;
	percentageLTMPotential.shortTermWeight -= learningRate * dLoss_dPercentagePotential * dPercentagePotential_dSigmoidInput * shortTermMemory;

}

void backPropOutputGate(double dLoss, double currentVector) {
	double dLoss_OutputGate = dLoss * tanh_Activation(longTermMemory);//gradient decente
	double dLoss_OutputGateDeravative = sigmoid_Activation(outputActivation) * (1 - sigmoid_Activation(outputActivation));//deravative change..... 

	//updating...
	percentageSTMPotental.bias -= learningRate * dLoss_OutputGate * dLoss_OutputGateDeravative;
	percentageSTMPotental.inputWeight -= learningRate * dLoss_OutputGate * dLoss_OutputGateDeravative * currentVector;
	percentageSTMPotental.shortTermWeight -= learningRate * dLoss_OutputGate * dLoss_OutputGateDeravative * shortTermMemory;
}

void lstmBackprop(std::vector<double> predictedOutput, std::vector<double> actualOutput, std::vector<double> wordEmbedding) {
	//checking the size
	if (predictedOutput.size() != actualOutput.size()) {
		std::cerr << "Error: Size mismatch between predicted and actual outputs.\n";
		return;
	}
	// Loop through each timestep in reverse for backpropagation through time
	for (int t = wordEmbedding.size() - 1; t >= 0; --t) {
		// Calculate the loss derivative wrt STM (short-term memory)
		double dLoss_dSTM = predictedOutput[t] - actualOutput[t];  // Gradient of loss with respect to the predicted output

		// Pass in the gradient and the current input vector for backpropagation
		backPropOutputGate(dLoss_dSTM, wordEmbedding[t]);
		backPropInputGate(dLoss_dSTM, wordEmbedding[t]);
		backPropForgetGate(dLoss_dSTM, wordEmbedding[t]);
	}
}



