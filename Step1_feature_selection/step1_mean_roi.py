#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Goal: test the accuracy of performance by spliting the input features into averged regional values
# Edit: JY Yang, Feb. 10

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
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA


# In[2]:


# import helper function modules
import import_ipynb
from helper_general_info import *
from helper_pdf_draw import *
from helper_group_classify import *
from helper_classification_script import *


# In[3]:


# Import and read the data file
df = pd.read_csv('data_step01_jiayue_practice.csv')
# print the information about imbalance check of the dataset
print_imbalance_info(df)


# In[4]:


# perform transformation (pca) to the dataset
# since the average calculation based on the imputed data --> no NaN wanted as we calculate the average
X, y, id_groups = pre_process_group_classify(df)

column_names = list(X)

# impute and scale the X feature dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
X = impute.fit_transform(X_scaled)


# return the X dataset back to a dataframe

X = pd.DataFrame(X, columns = column_names)


# In[5]:


# modulate the X feature dataset, change it into the new X with mean values by interest areas
X = mean_area_feature(X)


# In[6]:


# to retain an explained variance of 80%, we need to have 45 components
C_iter = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10]

# Logistic Regression Models
for c in C_iter:
    clf = LogisticRegression(penalty='l1', solver='saga', C=c)
    
    
    # calculate the corresponding scores based on train test split (LeaveOneGroupOut)
    accuracies, f1s, cms = classify_loso(X, y, id_groups, clf)
    

    clf_data = {
        'accuracies': accuracies,
        'f1s': f1s,
        'cms': cms,
    }

    pain_final_performance_log_file = open('concussion_perform_select_log_mean_%s.pkl' % c, 'ab')
    pickle.dump(clf_data, pain_final_performance_log_file)
    pain_final_performance_log_file.close()
    print(sum(accuracies))


# In[7]:


# SVM Models
kernel = ['linear', 'rbf']
# the c list is the same as logistic regression
for c in C_iter:
    for k in kernel:
        # feed in the svm module to run based on current setting of kernel and c
        clf = SVC(max_iter=10000, kernel=k, C=c)
    
        # calculate the corresponding scores based on train test split (LeaveOneGroupOut)
        accuracies, f1s, cms = classify_loso(X, y, id_groups, clf)
        
        # save data into a separated pickle file
        clf_data = {
            'accuracies': accuracies,
            'f1s': f1s,
            'cms': cms,
        }
        
        # create a new pickle file to store the data
        pain_final_performance_svm_file = open('concussion_perform_select_svm_mean_{}_{}.pkl'.format(c, k), 'ab')
        pickle.dump(clf_data, pain_final_performance_svm_file)
        pain_final_performance_svm_file.close()
        print(sum(accuracies))


# In[8]:


# Decision Tree Models
Crit = ['gini', 'entropy']
for cr in Crit:
    # define classification module using 2 criteria
    clf = DecisionTreeClassifier(criterion=cr)
    
    # calculate the corresponding scores based on train test split (LeaveOneGroupOut)
    accuracies, f1s, cms = classify_loso(X, y, id_groups, clf)
    
    # save the accuracy scores of the 2 criteria training
    clf_data = {
        'accuracies': accuracies,
        'f1s': f1s,
        'cms': cms,
    }

    pain_final_performance_tree_file = open('concussion_perform_select_tree_mean_%s.pkl' % cr, 'ab')
    pickle.dump(clf_data, pain_final_performance_tree_file)
    pain_final_performance_tree_file.close()
    print(sum(accuracies))


# In[9]:


# LDA model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# hyperparameters are by default
clf = LinearDiscriminantAnalysis()

# calculate the corresponding scores based on train test split (LeaveOneGroupOut)
accuracies, f1s, cms = classify_loso(X, y, id_groups, clf)


clf_data = {
    'accuracies': accuracies,
    'f1s': f1s,
    'cms': cms,
}

# save the dict of classification scores in the pickle file
pain_final_performance_lda_file = open('concussion_perform_select_lda_mean.pkl', 'ab')
pickle.dump(clf_data, pain_final_performance_lda_file)
pain_final_performance_lda_file.close()
print(sum(accuracies))


# In[ ]:




