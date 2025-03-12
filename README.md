# deep-learning-challenge
Module 21 Challenge

# Deep Learning Challenge: Charity Funding Predictor

## Overview of the Analysis

### Purpose:
The goal of this analysis is to develop a binary classifier that predicts whether organizations funded by the nonprofit foundation Alphabet Soup will use the funding successfully. By leveraging historical data from over 34,000 organizations, these models aims to supports Alphabet Soup in identifying applicants with the highest likelihood of success, thereby improving funding allocation decisions.

### Background:
Alphabet Soup has provided a dataset with detailed metadata—including application type, affiliated sector, government classification, and more—for organizations that have received funding over the years. The analysis involves:

* Preprocessing the data to handle rare categories and encode categorical variables.
* Building and optimizing a neural network model using TensorFlow/Keras.
* Evaluating the model’s performance and exploring further optimization avenues.

## Data Preprocessing

### Target Variable:
IS_SUCCESSFUL: This binary column indicates if an organization used the funding effectively (1 for success, 0 for non-success).

### Feature Variables:
After preprocessing, the features include:

* NAME: The organization’s name, which is grouped by replacing infrequent names with 5 or less with “Other.”
* APPLICATION_TYPE: Grouped with fewer than 1000 or 500 occurrences are replaced with “Other.”
* CLASSIFICATION: Similarly, rare classifications 1000 or 100 occurrences are grouped into “Other.”
* Additional features generated from the remaining columns binary encoding (e.g., AFFILIATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT).

### Variables Removed:
* EIN: Dropped from the dataset because it is a unique identifier and does not contribute predictive value.

### Processing Steps:
### * Binning Rare Values:
        * For NAME, replace those that appear 5 or fewer times with “Other.”
        * For APPLICATION_TYPE and CLASSIFICATION, replace values with counts less than 1000 or 500 with “Other.”

### Encoding:
* Convert categorical variables to numeric using pd.get_dummies().
* Convert all columns to a consistent numeric types int or float to ensure compatibility with TensorFlow.

## Compiling, Training, and Evaluating the Model

### Best Optimized Model Architecture: 

The optimized neural network is structured as follows:

### * Input Layer:
        * Determined by the number of features in the preprocessed dataset.

### * Hidden Layers:
        * First Hidden Layer: 80 neurons with the ReLU activation function.
            Rationale: ReLU activation helps mitigate vanishing gradients and efficiently captures non-linear relationships.
        * Second Hidden Layer: 30 neurons with the Sigmoid activation function.
            Rationale: ReLU is common, using a sigmoid in the second layer can sometimes help in bounding the activations.

### Output Layer:
* 1 neuron with Sigmoid activation, which is standard for binary classification tasks.

### Compilation and Training:

### * Compilation:
        * Loss Function: Binary crossentropy.
        * Optimizer: Adam.
        * Metrics: Accuracy.

### * Training:
The model is trained for 100 epochs on the scaled training data.

### * Evaluation:
        * The model is evaluated on the test set, achieving an accuracy above the target 75%.

### Optimization Efforts:
* Retained the “NAME” variable by grouping infrequent names rather than dropping it entirely.
* Adjusted the grouping thresholds for both APPLICATION_TYPE and CLASSIFICATION to reduce noise in the categorical variables.
* Experimented with different activation functions across layers and settled with a combination of ReLU in the first hidden layer and Sigmoid in the second.

## Summary and Recommendation
### Summary:
The deep learning model effectively predicts the success of funded organizations with an accuracy above the target threshold exceeding 75%. In the optimized run, performance improved to around 79%. The strategy of grouping rare categories and selecting the network’s architecture of 80 neurons in the first layer, 30 in the second, proved effective in enhancing predictive performance.

### Recommendation for an Alternative Approach:
While the deep learning model performs well, an alternative solution such as a Random Forest Classifier could also be considered for this classification problem.

### * Advantages of Random Forest:
* Works well with overfitting due to the combination of many models.
Capable of processing diverse data types while evaluating metrics to assess the significance of individual features. 
* Often simpler to tune and interpret compared to deep neural networks.

### * Rationale:
A Random Forest model might achieve a similar accuracy around 78–80% while offering easier interpretability and potentially lower computational requirements for training and deployment.

