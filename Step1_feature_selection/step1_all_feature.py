# Goal: test the accuracy of performance by applying all features in the training
# Edit: JY Yang, Feb. 4
# Adapted from Liz code Best_Params_Loops_StateComparison_BaselineVsCold

# import the modules that are used
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import io
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# import helper function modules
import import_ipynb
from helper_general_info import *
from helper_pdf_draw import *
from helper_group_classify import *
from helper_classification_script import *


# Import and read the data file
df = pd.read_csv('data_step01_jiayue.csv')
# print the information about imbalance check of the dataset
print_imbalance_info(df)



# Logistic Regression Models
C_iter = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10]
for c in C_iter:
    clf = LogisticRegression(penalty='l1', solver='saga', C=c)
    
    # apply impute and scale to the classifier
    pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler()),
        ('CLF', clf)])

    X, y, id_groups = pre_process_group_classify(df)
    # calculate the corresponding scores based on train test split (LeaveOneGroupOut)
    accuracies, f1s, cms = classify_loso(X, y, id_groups, pipe)
    

    clf_data = {
        'accuracies': accuracies,
        'f1s': f1s,
        'cms': cms,
    }

    pain_final_performance_log_file = open('concussion_perform_select_log_%s.pkl' % c, 'ab')
    pickle.dump(clf_data, pain_final_performance_log_file)
    pain_final_performance_log_file.close()
    print(sum(accuracies))



# SVM Models
kernel = ['linear', 'rbf']
# the c list is the same as logistic regression
for c in C_iter:
    for k in kernel:
        # feed in the svm module to run based on current setting of kernel and c
        clf = SVC(max_iter=10000, kernel=k, C=c)

        # apply impute and scale to the classifier
        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler()),
            ('CLF', clf)])

        X, y, id_groups = pre_process_group_classify(df)
        # calculate the corresponding scores based on train test split (LeaveOneGroupOut)
        accuracies, f1s, cms = classify_loso(X, y, id_groups, pipe)
        
        # save data into a separated pickle file
        clf_data = {
            'accuracies': accuracies,
            'f1s': f1s,
            'cms': cms,
        }
        
        # create a new pickle file to store the data
        pain_final_performance_svm_file = open('concussion_perform_select_svm_{}_{}.pkl'.format(c, k), 'ab')
        pickle.dump(clf_data, pain_final_performance_svm_file)
        pain_final_performance_svm_file.close()
        print(sum(accuracies))



# Decision Tree Models
Crit = ['gini', 'entropy']
for cr in Crit:
    # define classification module using 2 criteria
    clf = DecisionTreeClassifier(criterion=cr)
    
    # apply impute and scale to the classifier
    pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler()),
        ('CLF', clf)])

    X, y, id_groups = pre_process_group_classify(df)
    # calculate the corresponding scores based on train test split (LeaveOneGroupOut)
    accuracies, f1s, cms = classify_loso(X, y, id_groups, pipe)
    
    # save the accuracy scores of the 2 criteria training
    clf_data = {
        'accuracies': accuracies,
        'f1s': f1s,
        'cms': cms,
    }

    pain_final_performance_tree_file = open('concussion_perform_select_tree_%s.pkl' % cr, 'ab')
    pickle.dump(clf_data, pain_final_performance_tree_file)
    pain_final_performance_tree_file.close()
    print(sum(accuracies))




# LDA model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# hyperparameters are by default
clf = LinearDiscriminantAnalysis()

# apply impute and scale to the classifier
pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()),
    ('CLF', clf)])

X, y, id_groups = pre_process_group_classify(df)
# calculate the corresponding scores based on train test split (LeaveOneGroupOut)
accuracies, f1s, cms = classify_loso(X, y, id_groups, pipe)


clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

# save the dict of classification scores in the pickle file
pain_final_performance_lda_file = open('concussion_perform_select_lda.pkl', 'ab')
pickle.dump(clf_data, pain_final_performance_lda_file)
pain_final_performance_lda_file.close()
print(sum(accuracies))
