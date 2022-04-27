# Goal: test the accuracy of performance by different modifications (feature reduction methods)
# Edit: JY Yang, Feb. 20
# Adapted from Liz code Step0_DefineModelsandTests

import pickle
import pandas as pd
import numpy as np

# define a function that load the pickle file based on the name input
def load_pickle(filename):
    '''Helper function to unpickle the pickled python obj'''
    # open and load the data of a pickle file based on the name
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    # output the file data
    return data



# define the functions of extracting data from the pickle files
# this step is the same if being placed within the PdfPages drawing function
# separating it as a distinct functions only intend to clear the code when running through different modules
'''
Input: list of values or model used in step 1 training, name of the module algorithm
Output: return two lists: 1st --> list of arrays of all accuracies in each split of training
                          2nd --> list of mean score of accuracies to each C value
'''
def data_extraction(L_iter, pkl_name):
    # group mean accuracy and f1 scores for all features
    group_all_mean_acc_allmodels = []
    group_all_mean_f1s_allmodels = []

    
    # mean scores of the accuracy and f1 scores of all models
    mean_scores_acc_allmodels = []
    mean_scores_f1s_allmodels = []
    
    # iterate through the c values in list
    for c in L_iter:
        # load the logreg pickle file (created in step 1 model training --> all features)
        group_all = load_pickle(pkl_name.format(c))
        
        # add accuracies to a single np array for boxplot
        acc_all = [np.array(group_all['accuracies']) * 100]
        # get mean for all comparisons (group and condition)
        acc_all_comparisons = np.concatenate(acc_all, axis=0)  
        acc_all_comparisons_mean = np.mean(acc_all_comparisons)
        acc_all_comparisons_mean = np.round(acc_all_comparisons_mean, decimals=2)
        
        '''
        Currently we are focusing on a pdf report of classification accuracies in decision of which module to choose from
        The f1 scores will be take care of after the final clean-up of the codes
        
        # add f1 scores to a single np array for boxplot
        f1s_all = [np.array(group_all['f1s']) * 100]
        # get mean for all comparisons (group and condition)
        f1s_all_comparisons = np.concatenate(f1s_all, axis=0)  
        f1s_all_comparisons_mean = np.mean(f1s_all_comparisons)
        f1s_all_comparisons_mean = np.round(f1s_all_comparisons_mean, decimals=2)
        '''
        
        '''
        Take care of in the finalization step, current report focuses on the accuracy scores only
        
        # save the mean comparison f1 scores with corresponding c values
        mean_scores_f1s = [c, f1s_all_comparisons_mean]
        mean_scores_f1s_allmodels.append(mean_scores_f1s)
        '''
        
        
        # define the names of the feature selection methods
        module = 'Logistic Regression'
        
        # obtain the accuracies and f1 scores again to np array
        group_all_acc = [np.array(group_all['accuracies']) * 100]
        '''
        group_all_f1s = [np.array(group_all['f1s']) * 100]
        '''
        # round the mean accuracies to 33 decimals
        group_all_mean_acc = np.mean(group_all_acc)
        group_all_mean_acc = np.round(group_all_mean_acc, decimals=3)
        '''
        group_all_mean_f1s = np.mean(group_all_f1s)
        group_all_mean_f1s = np.round(group_all_mean_f1s, decimals=3)
        '''
        # save all the accuracy scores for the boxplot mapping
        group_all_mean_acc_allmodels.append(group_all_acc)
        flat_list = [item for sublist in group_all_mean_acc_allmodels for item in sublist]
        # save all mean accuracies for the table
        mean_scores_acc_allmodels.append(group_all_mean_acc)
        '''
        group_all_mean_f1s_allmodels.append(group_all_mean_f1s)
        '''
        
    return flat_list, mean_scores_acc_allmodels

