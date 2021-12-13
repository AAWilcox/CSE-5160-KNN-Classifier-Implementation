/*****************************************************************************
* Name:    Alyssa Wilcox
* Id:      006861225
* Date:    November 23, 2020
* Purpose: A KNN classifier that reads test instance data and training instance
		   data from external files. The classifier then classifies the test
		   instances based on the training instances.
*******************************************************************************/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

/*****************************************************************************
* Purpose:       Read from the text data file. Save each test instance in
				 a vector, save each test instance vector into a test
				 set vector.
* Parameter(s):  infile: input stream object.
* Return:        Returns a 2D vector, the test set vector. This vector contains
				 each test instance vector.
*******************************************************************************/
vector<vector<double> > getTestingSet(ifstream& infile) {
	//Create seed data variables
	double A, P, C, L, W, AC, LG, y;

	//Vector to hold all training instances
	vector<vector<double>> test_set;

	//Read through the file
	while (!infile.eof()) {
		//Read the data values
		infile >> A >> P >> C >> L >> W >> AC >> LG >> y;
		//Create a training instance vector
		vector<double> test_inst = { A, P, C, L, W, AC, LG, y };
		//Push training instance to training set
		test_set.push_back(test_inst);
	}
	return test_set;
}

/*****************************************************************************
* Purpose:       Read from the text data file. Save each training instance in
				 a vector, save each training instance vector into a training
				 set vector.
* Parameter(s):  infile: input stream object.
* Return:        Returns a 2D vector, the training set vector. This vector contains
				 each training instance vector.
*******************************************************************************/
vector<vector<double> > getTrainingSet(ifstream& infile) {
	//Create seed data variables
	double A, P, C, L, W, AC, LG, y;

	//Vector to hold all training instances
	vector<vector<double>> training_set;

	//Read through the file
	while (!infile.eof()) {
		//Read the data values;
		infile >> A >> P >> C >> L >> W >> AC >> LG >> y;
		//Create a training instance vector
		vector<double> training_inst = { A, P, C, L, W, AC, LG, y };
		//Push training instance to training set
		training_set.push_back(training_inst);
	}
	return training_set;
}

/*****************************************************************************
* Purpose:       Feature scaling through min-max normalization - standardizes
				 the range of feature values
* Parameter(s):  training_set: The 2D training set array. Passed by reference.
				 toClassify: Input vector we are trying to classify. Passed
				 by reference.
* Return:        Returns nothing.
*******************************************************************************/
void minMaxNormalization(vector<vector<double> >& training_set, vector<double>& toClassify) {

	//Find min and max values:

	//To hold the min and max values for each feature
	vector<double> max_values;
	vector<double> min_values;
	//Feature number
	int n = 0;
	//Traverse the training set feature by feature
	while (n < 7) {
		//Set initial max value to zero
		double max = 0;
		//Set initial min value to the first training inst's
		//n feature value
		double min = training_set[0][n];
		//Traverse the n feature of each training instance
		//m training instances
		for (int m = 0; m < training_set.size(); m++) {
			if (max < training_set[m][n])
				max = training_set[m][n];
			if (min > training_set[m][n])
				min = training_set[m][n];
		}
		//Check the n feature of the test instance
		if (max < toClassify[n])
			max = toClassify[n];
		if (min > toClassify[n])
			min = toClassify[n];
		//Add the found max and min values for a feature to vector
		max_values.push_back(max);
		min_values.push_back(min);
		//After going through one feature, go through next feature
		n++;
	}

	//Min-Max Normalization:

	//Feature number
	n = 0;
	//Traverse the training set feature by feature
	while (n < 7) {
		//Traverse the n feature of each training instance
		//m training instances
		for (int m = 0; m < training_set.size(); m++) {
			training_set[m][n] = ((training_set[m][n] - min_values[n]) / (max_values[n] - min_values[n]));
		}
		//min-max normalize the input data
		toClassify[n] = ((toClassify[n] - min_values[n]) / (max_values[n] - min_values[n]));
		//After going through one feature, go through next feature
		n++;
	}
}

/*****************************************************************************
* Purpose:       Finds Euclidean distances between input vector and each
				 training instance.
* Parameter(s):  training_set: The 2D training set array. Passed by reference.
				 toClassify: Input vector we are trying to classify. Passed
				 by reference.
* Return:        Returns nothing.
*******************************************************************************/
void distances(vector<vector<double> >& training_set,const vector<double>& toClassify) {
	//Traverse through all m training instances
	for (int m = 0; m < training_set.size(); m++) {
		double sum = 0;
		//Traverse through n-1 features per instance
		for (int n = 0; n < training_set[m].size() - 1; n++) {
			//Keep track of the sum
			sum = sum + pow(toClassify[n] - training_set[m][n], 2);
		}
		//Take the square root of the sum, push it onto the back of the
		//training instance
		training_set[m].push_back(sqrt(sum));
	}
}

/*****************************************************************************
* Purpose:       Acts as a driver function to sort our 2D training_set vector
				 on the basis of a particular column. The column used here is
				 the 8th column, the column holding the Euclidean distances.
* Parameter(s):  v1: First comparison vector, passed by reference.
				 v2: Second comparions vector, passed by reference.
* Return:        Returns bool value, depending on if the 8th column of v1
				 is less than the 8th column of v2.
*******************************************************************************/
bool sortcol(const vector<double>& v1, const vector<double>& v2) {
	return v1[8] < v2[8];
}

/*****************************************************************************
* Purpose:       Finds the output y of the k nearest neighbors in the
				 training_set vector.
* Parameter(s):  training_set: The 2D training set array. Passed by reference.
				 k: The number of k nearest neighbors.
* Return:        Returns output, a vector containing the output y of the k
				 nearest neighbors.
*******************************************************************************/
vector<int> KNNoutput(const vector<vector<double> >& training_set, const int k) {
	//Vector to hold KNN outputs
	vector<int> output;
	//Get output y of each nearest neighbor
	for (int m = 0; m < k; m++) {
		output.push_back(training_set[m][7]);
	}
	return output;
}

/*****************************************************************************
* Purpose:       Classifies the input based on the output y of the k nearest
				 neighbors.
* Parameter(s):  KNNoutput: Contains the output y of the k nearest neighbors.
* Return:        Returns an integer, which is the classification of the input.
				 Return 0 if there is no majority.
				 Return 1 if the majority of the knn nearest neighbors are
				 classified as 1.
				 Return 2 if the majority of the knn nearest neighbors are
				 classified as 2.
				 Return 3 if the majority of the knn nearest neighbors are
				 classified as 3.
*******************************************************************************/
int classify(const vector<int>& KNNoutput) {
	//Output types
	int one = 0;
	int two = 0;
	int three = 0;
	//Count how many of each output types we have
	for (int i = 0; i < KNNoutput.size(); i++) {
		if (KNNoutput[i] == 1)
			one++;
		else if (KNNoutput[i] == 2)
			two++;
		else if (KNNoutput[i] == 3)
			three++;
	}
	//Find which output classification is the majority
	if ((one > two) && (one > three)) {
		return 1;
	}
	else if ((two > one) && (two > three)) {
		return 2;
	}
	else if ((three > one) && (three > two)) {
		return 3;
	}
	//There was no majority
	else {
		return 0;
	}
}

/*****************************************************************************
* Purpose:       Validates the user input, ensuring it is of type int.
* Parameter(s):  data_value: The user input data value. Passed by reference.
* Return:        Returns data_value, which is a valid int user input.
*******************************************************************************/
int inputValidation(int& data_value) {
	//Initialize valid as false
	bool valid = false;
	//Check if input is valid
	while (valid == false) {
		//Input is not valid
		if (cin.fail()) {
			cout << "Please enter valid input ";
			//Reset failbit
			cin.clear();
			//Ignore previous input
			cin.ignore();
			cin >> data_value;
		}
		//Input is valid
		else valid = true;
	}
	return data_value;
}

/*****************************************************************************
* Purpose:       Main entry point.
* Parameter(s):  None.
* Return:        Returns 0 upon program completion.
*******************************************************************************/
int main() {
	//Introduction
	cout << "KNN Classifier" << endl;
	cout << "This program classifies three varieties of wheat seeds: "
		<< "Kama(1), Rosa(2), and Canadian(3)" << endl << endl;

	//k nearest neighbors
	int k;

	//Get user input
	cout << "How many nearest neighbors? ";
	cin >> k;
	inputValidation(k);
	cout << endl;
	
	//Create input file names
	string infn_test = "seeds_training_error.txt";
	string infn_training = "seeds_training_set.txt";

	//Input file stream to open a file
	ifstream infile_test;
	ifstream infile_training;

	//Open the files
	infile_test.open(infn_test);
	infile_training.open(infn_training);

	//If the files fail to open
	if (infile_test.fail()) {
		cout << "Error opening " << infn_test << endl;
		return 0;
	}
	else if (infile_training.fail()) {
		cout << "Error opening " << infn_training << endl;
		return 0;
	}

	//Get test instances
	vector<vector<double>> test_set = getTestingSet(infile_test);

	//Training set
	vector<vector<double>> training_set;

	//Used later for console display
	int instance = 1;

	//Classify each test instance
	for (int m = 0; m < test_set.size(); m++) {

		//Get training set
		training_set = getTrainingSet(infile_training);

		//min-max normalize training set and test instance
		minMaxNormalization(training_set, test_set[m]);

		//Find euclidean distances between training instances and test instance
		distances(training_set, test_set[m]);

		//Sort the training set based on distances
		sort(training_set.begin(), training_set.end(), sortcol);

		//Get output of nearest neighbors
		vector<int> KNNoutputY = KNNoutput(training_set, k);

		//Make the prediction
		int prediction = classify(KNNoutputY);
		
		cout << "Instance " << instance << ": " << prediction << endl;
		instance++;

		//Clear the vectors
		training_set.clear();
		KNNoutputY.clear();	

		//Reset ifstream object
		infile_training.clear();
		infile_training.seekg(0);
	}

	//Close the files
	infile_test.close();
	infile_training.close();

	return 0;
}
