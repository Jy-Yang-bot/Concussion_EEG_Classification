# Pediatric Concussion Classification of ML using EEG biomarker
Decision tree classification based on EEG signals in healthy children and pediatric concussion patients.

### Project description
The project to use machine learning methods to determine if functional connectivity and graph theory features could accurately classify children with concussion from healthy controls.
We selected the decision tree algorithm (criterion: entropy) using theta frequency EEG features as the final model of classification.
The steps of the ML classification are listed below:

## Step 0: Define helper functions
This step will define some helper functions for the following steps. We will define functions for:
- helper_general_info.py: print the representation and counts for different injury status in the concussion dataset.

- helper_group_classify.py: pre-process the dataset and split into X (features), y(injury status), and id_groups(patient id). This group classify helper function includes: 1) full dataset splitting; 2) averaged subset based on different regions of interest (ROI) (frontal, temporal, parietal, occipital); 3) multiple subsets based on frequency band (delta, theta, alpha, and beta) and functional connectivity method (wPLI vs. AEC); 4) test set manipulations for permutation and bootstrapping tests.

- helper_classification_script.py: perform the classification training and testing. This function will return the accuracies of classification, f1 scores, confusion matrix, result of predicted injury status, and the actual injury status of the patients. 

- helper_pickle_loading.py: this function will load the pickle file based on their names. It will extract the full accuracy data in the pickle files and calculate the mean accuracy of each hyperparameter (C).


## Step 1: Feature selection
This step will create different subsets of features to avoid model overfitting. We included five catagories of feature reduction:
- step2_all_feature.py: all EEG features
- step2_pca_80%var.py: PCA selected features with 80% variance
- step2_mean_roi.py: averaged ROI features as described above
- step2_freq_band.py: separated frequency band features (delta, theta, alpha, and beta)
- step2_connectivity.py: connectivity types (wPLI vs. AEC).

#### Four algorithms will be included: logistic regression, support vector machine (SVM), decision tree, and linear discriminant analysis (LDA).

#### Two ways of model training will be included: non-voted and voted. 
#### Way 1: Non-Voted
- There won't be any modification of classification process for non-voted training. 
#### Way 2: Voted
- For the voted training, we will calculate the majority of injury status prediction for each participant. The prediction majority will be counted as the final voted decision for the injury status of the specific patient, and this will be used to calculate the voted accuracy.

After the classification using each method of feature selection, the results of classification will be saved for further visualization.


## Step 2: Model training and visualization
This step will perform classification on each method described in step 1 and store the classification results. All methods and algorithms will be trained for both non-voted and voted approach.
Based on the non-voted and voted accuracies, we will decide the optimal model with the highest accuracy. In this project, the final model will be the decision tree algorithm (criterion: entropy) using theta frequency EEG features.


## Step 3: Decision tree classification

