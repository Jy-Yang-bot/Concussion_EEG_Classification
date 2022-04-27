# Goal: input the dataset, replacing each NaN cell with the average of the column
# Edit: JY Yang, Jan. 28, Feb. 11, Mar. 11
# Updated Fed. 3, change input to df initial dataset
# Adapted from: Liz code pre_process_group_classify, Step0_DefineModelsandTests

# Import the pandas and other modules
import pandas as pd
from math import floor
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import permutation_test_score
from sklearn.utils import resample

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import joblib

import pickle

import multiprocessing as mp
import os
import sys

# import helper function modules
from helper_classification_script import *


# define a function that split the dataset into X, y, and p_id
'''
pre_process_group_classify(df): (dataframe)

input: csv concussion dataset
output: return X: EEG features
               y: group variable (0 = healthy, 1 = concussion)
               id_group: id number of each subject
'''
def pre_process_group_classify(df):
    # the dataset features used for the training includes all EEG features, while the group represents prediction of module
    # X = EEG features

    # copy all features into X set
    X = df.iloc[: , 4:]
    
    # split the dataset with id numbers of participants
    # id = participants representing id, each participant has multiple windows of recordings
    p_id = df.iloc[: , :1]
    
    # visualize if the p_id can be converted into a single list of patient id corresponding to index of the EEG features dataframe
    id_groups = np.array(p_id.stack().tolist())
    
    # Get the group variable as model prediction output from the dataset
    # y = group (if concussion or not)
    y = df.iloc[: , 1:2]

    return X, y, id_groups


# define a function to search for if specific key word exist in the feature
# intend to group features based on areas in specific frequency bands and connectivity types
'''
search(searchFor, feature): (list, str)

input: list of key words stating for interest areas
        string of each feature names (we have 192 features)
output: return boolean value --> True / False
'''
def search(searchFor, feature):
    # iterate through the splited feature name --> into key words
    # start a count
    count = 0
    ele = feature.split('_')
    for keys in ele:
        # if we have a keyword in list, add count
        if keys in searchFor:
            count += 1
        
    # if all feature key words are iterated, we have more than 3 in the given reference list --> interest area --> return True
    # if we have less than 3, thus not in interest area --> return False
    if count >= 3:
        return True
    else:
        return False


# define a function to split feature names by areas of interest
''' 
mean_area_feature(X): (dataframe)

input: X
        192 EEG features as dataframe
output: list
        return a list enclosing the feature names by interest areas in sublist
'''
def area_split(X):
    # define the 3 lists by groups of key words
    areas = [['fp1', 'fp2', 'fz', 'f3', 'f4', 'f7', 'f8'], ['cz', 'c3', 'c4'], 
             ['t3', 't4', 't5', 't6'], ['pz', 'p3', 'p4'], ['o1', 'o2']]
    freq_bands = [['delta'], ['theta'], ['alpha'], ['beta']]
    connect = [['aec'], ['wpli']]
    column_names = list(X)
    
    # define an empty list for dividing the different areas of interest
    fea_by_area = []
    # iterate through each type keywords
    for types in connect:
        for freq in freq_bands:
            for element in areas:
                # create and empty the list in each iteration of each keyword
                single_fea = []
                # define the key by the combination of kinds --> specific interest area
                keys = types + freq + element
                # iterate through each features
                for feature in column_names:
                    # check if they are in the current interest area
                    if search(keys, feature):
                        # yes --> append into the temporary list
                        single_fea.append(feature)
                    else:
                        # no --> exit the current interest area
                        # append the previous area temporary list into overall area list
                        fea_by_area.append(single_fea)
                        # zeroing the temporary list
                        single_fea = []

    # remove any empty list in the overall area list
    fea_by_area = [x for x in fea_by_area if x != []]
    return fea_by_area



# define a function to calculate the averaged feature values based on specific interest areas
# each average is calculated from multiple electrodes with data in specific frequency bands and connectivity in 1 area
''' 
mean_area_feature(X): (dataframe)

input: X
        192 EEG features as dataframe
output: X_mean
        return averaged EEG features by interest areas as a dataframe
'''
'''
Not sure if to include these: not in specific area
pathlen, geff, etc.

'''
def mean_area_feature(X):
    # create a list of new column names of the returned X_mean dataframe
    new_names = [
                 'f_delta_aec', 'c_delta_aec', 't_delta_aec', 'p_delta_aec', 'o_delta_aec',
                 'f_theta_aec', 'c_theta_aec', 't_theta_aec', 'p_theta_aec', 'o_theta_aec',
                 'f_alpha_aec', 'c_alpha_aec', 't_alpha_aec', 'p_alpha_aec', 'o_alpha_aec',
                 'f_beta_aec', 'c_beta_aec', 't_beta_aec', 'p_beta_aec', 'o_beta_aec',
                 
                 'f_delta_wpli', 'c_delta_wpli', 't_delta_wpli', 'p_delta_wpli', 'o_delta_wpli',
                 'f_theta_wpli', 'c_theta_wpli', 't_theta_wpli', 'p_theta_wpli', 'o_theta_wpli',
                 'f_alpha_wpli', 'c_alpha_wpli', 't_alpha_wpli', 'p_alpha_wpli', 'o_alpha_wpli',
                 'f_beta_wpli', 'c_beta_wpli', 't_beta_wpli', 'p_beta_wpli', 'o_beta_wpli'
                ]
    
    singles = ['pathlen_delta_aec', 'geff_delta_aec', 'cluster_delta_aec', 'bsw_delta_aec',
                'mod_delta_aec', 'pathlen_theta_aec', 'geff_theta_aec', 'cluster_theta_aec',
                'bsw_theta_aec', 'mod_theta_aec', 'pathlen_alpha_aec', 'geff_alpha_aec',
                'cluster_alpha_aec', 'bsw_alpha_aec', 'mod_alpha_aec', 'pathlen_beta_aec',
                'geff_beta_aec', 'cluster_beta_aec', 'bsw_beta_aec', 'mod_beta_aec',
              
                'pathlen_delta_wpli', 'geff_delta_wpli', 'cluster_delta_wpli', 'bsw_delta_wpli',
                'mod_delta_wpli', 'pathlen_theta_wpli', 'geff_theta_wpli', 'cluster_theta_wpli',
                'bsw_theta_wpli', 'mod_theta_wpli', 'pathlen_alpha_wpli', 'geff_alpha_wpli', 
                'cluster_alpha_wpli', 'bsw_alpha_wpli', 'mod_alpha_wpli', 'pathlen_beta_wpli',
                'geff_beta_wpli', 'cluster_beta_wpli', 'bsw_beta_wpli', 'mod_beta_wpli'
              ]
    # split the original features by the interest areas
    fea_by_area = area_split(X)
    
    # create a new list to fill the data
    mean_X = pd.DataFrame()
    
    # give the corresponding mean values into the new dataframe
    for index in range(len(fea_by_area)):
        # add the corresponding feature names into the new list
        mean_X[new_names[index]] = X[fea_by_area[index]].mean(axis=1)
        
    for ele in singles:
        mean_X[ele] = X[ele]
        
    
    return mean_X



# define a function that split the dataset into X (only specific frequency band, y, and p_id
'''
pre_process_group_classify_freq(df, band_name): (dataframe, str)

input: csv concussion dataset + desired name of the frequency band to choose from 
output: return X: EEG features in delta only frequency bands
               y: group variable (0 = healthy, 1 = concussion)
               id_group: id number of each subject
'''
def pre_process_group_classify_freq(df, band_name):
    # select for all columns containing the string of 'delta'
    # this only icludes the features of delta frequency bands --> thus considered as the 
    X = df.loc[:,[band_name in i for i in df.columns]]
    
    # split the dataset with id numbers of participants
    # id = participants representing id, each participant has multiple windows of recordings
    p_id = df.iloc[: , :1]
    
    # visualize if the p_id can be converted into a single list of patient id corresponding to index of the EEG features dataframe
    id_groups = np.array(p_id.stack().tolist())
    
    # Get the group variable as model prediction output from the dataset
    # y = group (if concussion or not)
    y = df.iloc[: , 1:2]

    return X, y, id_groups


# define a function that split the dataset into X (only specific connectivity, y, and p_id
'''
pre_process_group_classify_connect(df, connect_name): (dataframe, str)

input: csv concussion dataset + desired name of the frequency band to choose from 
output: return X: EEG features in delta only connectivity
               y: group variable (0 = healthy, 1 = concussion)
               id_group: id number of each subject
'''
def pre_process_group_classify_connect(df, connect_name):
    # select for all columns containing the string of 'delta'
    # this only icludes the features of delta frequency bands --> thus considered as the 
    X = df.loc[:,[connect_name in i for i in df.columns]]
    
    # split the dataset with id numbers of participants
    # id = participants representing id, each participant has multiple windows of recordings
    p_id = df.iloc[: , :1]
    
    # visualize if the p_id can be converted into a single list of patient id corresponding to index of the EEG features dataframe
    id_groups = np.array(p_id.stack().tolist())
    
    # Get the group variable as model prediction output from the dataset
    # y = group (if concussion or not)
    y = df.iloc[: , 1:2]

    return X, y, id_groups



# define a function that evaluate whether the accuracy is better than the chance
'''
Support function to determine if model accuracy is significantly better than chance
Args:
    X (numpy matrix): this is the feature matrix with row being a data point, EEG features in delta only connectivity
    y (numpy vector): this is the label vector with row belonging to a data point
    p_id (numpy vector): this is the p_id vector (which is a the participant id)
    clf (sklearn classifier): this is model being tested (see gridsearch function for specific models)
    num_permutation (int): the number of time to permute y
    random_state (int): this is used for reproducible output
Returns:
    f1s (list): the f1 at for each leave one out participant

Permutes targets (Y) to generate ‘randomized data’ and compute the empirical p-value against the null hypothesis that
    features and targets are independent.

The p-value represents the fraction of randomized data sets where the estimator
    performed as well or better than in the original data. A small p-value suggests that there is a real dependency
    between features and targets which has been used by the estimator to give good predictions. A large p-value may be
    due to lack of real dependency between features and targets or the estimator was not able to use the dependency to
    give good predictions.
'''
def permutation_test(X, y, id_groups, clf, num_permutation=10000):
    #use LOSO cross validation and split data into test and training sets
    logo = LeaveOneGroupOut()
    train_test_splits = logo.split(X, y, id_groups)

    #calculate p-value using sklearn permutation test function
    with joblib.parallel_backend('loky'): #run procedure in parallel
        (accuracies, permutation_scores, p_value) = permutation_test_score(clf, X, y, groups=id_groups,
                                                                           cv=train_test_splits,
                                                                           n_permutations=num_permutation,
                                                                           verbose=num_permutation, n_jobs=-1)
        #clf is model specified in gridsearch, n_jobs=1 is default

    #output accuracies, permutation test scores, and the p-value
    return accuracies, permutation_scores, p_value 





# define a function that classify the bootstrapping samples
'''
Helper function for the bootstrapping test to select for specific samples in analysis
Inputs:
    X (numpy matrix): this is the feature matrix with row being a data point, EEG features in delta only connectivity
    y (numpy vector): this is the label vector with row belonging to a data point
    p_id (numpy vector): this is the p_id vector (which is the participant id)
    clf (sklearn classifier): this is model being tested (see gridsearch function for specific models)
    sample_id: this is the variable that is used to select for data based on given id
'''
def bootstrap_classify(X, y, id_groups, clf, sample_id):
    print("Bootstrap sample #" + str(sample_id))
    sys.stdout.flush()  # This is needed when we use multiprocessing

    # Get the sampled with replacement dataset
    sample_X, sample_y, sample_p_id = resample(X, y, id_groups)

    # Classify and get the results
    #accuracies, cms = classify_loso(sample_X, sample_y, sample_p_id, clf)
    accuracies, f1s, cms = classify_loso(sample_X, sample_y, sample_p_id, clf)

    return np.mean(accuracies)




# define a function that calculate the confidence interval based on given p-value
'''
Create a confidence interval for the classifier with the given p value
Args:
    X (numpy matrix): The feature matrix with which we want to train on classifier on
    y (numpy vector): The label for each row of data point
    p_id (numpy vector): The p_id id for each row in the data (correspond to the participant ids)
    clf (sklearn classifier): The classifier that we which to train and validate with bootstrap interval
    num_resample (int): The number of resample we want to do to create our distribution
    p_value (float): The p values for the upper and lower bound
Returns:
    f1_distribution (float vector): the distribution of all the f1s
    f1_interval (float vector): a lower and upper interval on the f1s corresponding to the p value
'''
def bootstrap_interval(X, y, id_groups, clf, num_resample=1000, p_value=0.05):
    # Setup the pool of available cores
    #ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
    pool = mp.Pool(processes=4)

    # Calculate each round asynchronously
    #results = [pool.apply_async(bootstrap_classify, args=(X, y, p_id, clf, sample_id,)) for sample_id in
               #range(num_resample)]

    results = [bootstrap_classify(X, y, id_groups, clf, sample_id) for sample_id in range(num_resample)]

    
    acc_distribution = results

    # Sort the results
    acc_distribution.sort()

    # Set the confidence interval at the right index
    lower_index = floor(num_resample * (p_value / 2))
    upper_index = floor(num_resample * (1 - (p_value / 2)))
    acc_interval = acc_distribution[lower_index], acc_distribution[upper_index]

    return acc_distribution, acc_interval
