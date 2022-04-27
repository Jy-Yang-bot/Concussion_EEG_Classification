# Goal: split the dataset and proceed to classification
# Edit: JY Yang, Mar. 25
# Adapted from Liz code (Step0_DefineModelsandTests)


# import all the python packages needed to run the functions
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import permutation_test_score
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import joblib


# define a function to train the module
def classify_loso(X, y, id_groups, clf):
    """ 
    Split the dataset into train and test parts by the subjects.
    The run will be 9 in total, each time all windows of 1 subject are left out as test, the other 8 as train.
    In each fold, the sequential subject windows will be dropped
    Args:
        X (dataframe): this is the features with row being a data point
        y (dataframe): this is the label vector with row belonging to a data point
        id_groups (list): this is the p_id vector (which is a the participant id)
        clf (sklearn classifier): this is a classifier made in sklearn with fit, transform and predict functionality
    Returns:
        accuracies (list): the accuracies for each leave one out participant
        cms (list): the confusion matrix for each leave one out participant
    """
    # perform feature selection --> split train and test groups
    # prepare leave one out cross-validation procedure
    logo = LeaveOneGroupOut()
    # keep track of how many folds left
    num_folds = logo.get_n_splits(X, y, id_groups)
    
    # creating two blank list to place the accuracies and f1 scores
    accuracies = [] 
    f1s = []
    result = []
    actual_group = []
    # creating a blank 2x2 tables, where confusion matrix (true pos, false pos, etc.) output will be stored
    # specificity & sensitivity
    cms = np.zeros((2, 2)) 

    # change the X and y dataframe to numpy array
    X = X.to_numpy()
    y = y.to_numpy()
    # load in feature matrix (X), target matrix (y), and subject matrix (id_groups) and split them
    for train_index, test_index in logo.split(X, y, id_groups): 

        # split such that all rows for a single subject becomes the test set
        X_train, X_test = X[train_index], X[test_index]
        # while all rows for the remaining subjects becomes the training set
        y_train, y_test = y[train_index], y[test_index]
        
        # print out the number of folds left (from 9 to 1)
        print(f"Number of folds left: {num_folds}")
        
        # allows models to run concurrently/parallelized; loky is default
        with joblib.parallel_backend('loky'): 
            # fit model using the training dataset
            clf.fit(X_train, y_train) 
        # using the model parameters that we learned above, use testing set X data to predict y
        y_pred = clf.predict(X_test) 
        
        # get the accuracy scores and f1 scores by comparing the predicted result the test set answer of concussion group
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # draw a confusion matrix based on concussion group answer and prediction in the test set
        cm = confusion_matrix(y_test, y_pred)
        
        # for each fold we run, append the accuracy, f1 scores, and confusion matrix from that fold in a larger list
        accuracies.append(accuracy)
        f1s.append(f1)
        result.append(y_pred)
        actual_group.append(y_test)
        cms = np.add(cms, cm)
        
        # deduct 1 after each run of 1 fold
        num_folds = num_folds - 1

    #output model performance outcomes (accuracy and confusion matrix values)
    return accuracies, f1s, cms, result, actual_group


def vote_classify_loso(X, y, id_groups, clf):
    """
    Split the dataset into train and test parts by the subjects.
    The run will be 9 in total, each time all windows of 1 subject are left out as test, the other 8 as train.
    In each fold, the sequential subject windows will be dropped
    Args:
        X (dataframe): this is the features with row being a data point
        y (dataframe): this is the label vector with row belonging to a data point
        id_groups (list): this is the p_id vector (which is a the participant id)
        clf (sklearn classifier): this is a classifier made in sklearn with fit, transform and predict functionality
    Returns:
        accuracies (list): the accuracies for each leave one out participant
        cms (list): the confusion matrix for each leave one out participant
    """
    # perform feature selection --> split train and test groups
    # prepare leave one out cross-validation procedure
    logo = LeaveOneGroupOut()
    # keep track of how many folds left
    num_folds = logo.get_n_splits(X, y, id_groups)

    # creating a blank matrix, where model accuracies will eventually be stored
    result = []
    accuracies = []
    f1s = []
    ids = []
    # creating a blank 2x2 tables, where confusion matrix (true pos, false pos, etc.) output will be stored
    cms = np.zeros((2, 2))

    # change the X and y dataframe to numpy array
    X = X.to_numpy()
    y = y.to_numpy()
    # load in feature matrix (X), target matrix (y), and subject matrix (id_groups) and split them
    for train_index, test_index in logo.split(X, y, id_groups):
        # split such that all rows for a single subject becomes the test set
        X_train, X_test = X[train_index], X[test_index]
        # while all rows for the remaining subjects becomes the training set
        y_train, y_test = y[train_index], y[test_index]

        new_id = pd.DataFrame(id_groups, columns=['id'])
        id_train, id_test = new_id.loc[train_index], new_id.loc[test_index]

        # print out the number of folds left (from 9 to 1)
        print(f"Number of folds left: {num_folds}")

        # allows models to run concurrently/parallelized; loky is default
        with joblib.parallel_backend('loky'):
            # fit model using the training dataset
            clf.fit(X_train, y_train)
            # using the model parameters that we learned above, use testing set X data to predict y
        y_pred = clf.predict(X_test)

        # get the accuracy scores and f1 scores by comparing the predicted result the test set answer of concussion group
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # draw a confusion matrix based on concussion group answer and prediction in the test set
        cm = confusion_matrix(y_test, y_pred)

        # for each fold we run, append the accuracy, f1 scores, and confusion matrix from that fold in a larger list
        accuracies.append(accuracy)
        f1s.append(f1)
        result.append(y_pred)
        cms = np.add(cms, cm)
        ids.append(id_test.values.tolist())
        # deduct 1 after each run of 1 fold
        num_folds = num_folds - 1

    # give a list of non-repetitive ids and another list of the majority count of 0 / 1
    new_id = np.concatenate(ids)
    new_actual = y_test.tolist()
    flat_list = [item for sublist in new_actual for item in sublist]
    id_count = []
    vote = []
    for ids in new_id:
        if ids not in id_count:
            id_count.append(ids)

    for index in range(len(result)):
        counting = result[index].tolist()
        most_common = max(counting, key=counting.count)
        vote.append(most_common)

    # create a new dataframe with id regarding its prediction
    pred_result = pd.DataFrame(id_count, columns=["id"])
    pred_result['predict'] = vote

    # output model performance outcomes (accuracy and confusion matrix values)
    return accuracies, f1s, cms, pred_result



# define the function to calculate the accuracy scores from the voted predictions
def vote_accuracy(pred_result, df):
    """
    Calculate the voted result with the actual group of patients, giving an overall accuracies for the classifier
    Args:
        pred_result (dataframe): this is the id and predicted state (voted) of each patient
        df (dataframe): this is the all dataframe (input)
    Returns:
        vote_accuracy (list): the voted result of the overall accuracy in each hyperparameter setting
    """
    # setup an empty list to append the state data in and id
    state = []
    count = []

    # retrieve the state of each patient from the full dataset
    for index, row in df.iterrows():
        # get the group info from the initial dataset
        # starting from an empty list of id, if this id is new, append the corresponding state of it
        # if already in the count list, we had the state before, no need for appending again
        if row['id'] not in count:
            count.append(row['id'])
            state.append(row['group'])

    # add the all actual states of our participants to our dataframe
    pred_result['state'] = state

    # calculate and report for the voting accuracy in a list
    vote_accuracy = []
    v_acc = accuracy_score(pred_result['state'], pred_result['predict'])
    vote_accuracy.append(v_acc)

    return vote_accuracy, pred_result