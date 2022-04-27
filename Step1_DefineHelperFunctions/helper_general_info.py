# goal: check for if any imbalance of the data file
# edit: JY Yang, Jan. 21

# Import the pandas
import pandas as pd

# define a function that prints the imbalance information check for the dataset in group, id, mode
# input of the csv dataset, output several strings that writes about imbalance
def print_imbalance_info(df):
    # check for the imbalance in 'id'
    print('The distribution of data in each participant:')
    print(df['id'].value_counts())


    # check for the imbalance in 'group'
    print('\n\nThe distribution of data in each group:')
    print(df['group'].value_counts())


    # check for the imbalance in 'mode'
    print('\n\nThe distribution of data in each mode:')
    print(df['mode'].value_counts())


    # count the number of data windows from each participants

    # check for the imbalance in combined 'group' and 'id'
    print('\n\nThe data distrbution by participant in each group:')
    print(df[['group','id']].value_counts())



