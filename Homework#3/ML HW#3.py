import pandas as pd
import numpy as np 
%matplotlib inline 
import matplotlib.pyplot as plt

def readdata(data):
    return pd.read_csv(data+'.csv', header = 0)

def summary_statistics(df):
    pd.set_option('display.width', 18)
    print 'Column Names:', "\n", df.columns.values
    print 'First Few Rows of Data:', "\n", df.head()
    print 'Last Few Rows of Data:', "\n", df.tail()
    print 'Summary Statistics:', "\n", df.describe
    print 'Number of Missing Values:', "\n", df.isnull().sum()
    
    for col_name in df:
        #print ('Data Type %string: %string' %(col_name, df[col_name].dtype))
        
    print 'Correlation Matrix :', "\n", df.corr().unstack()

def plot_histogram(df, hist_var):
    fig = df[hist_var].hist()
    fig.set_title('Histogram for ' + hist_var)
    plt.draw()
    plt.savefig(hist_var)
    plt.close()

def plot_bar(df, bar_var):
    fig =df.groupby(bar_var).size().plot(kind='bar')
    fig.set_xlabel(bar_var) #defines the x axis label
    fig.set_ylabel('Number of Observations') #defines y axis label
    fig.set_title(bar_var+' Distribution') #defines graph title
    plt.draw()
    plt.savefig(bar_var)
    plt.close('all')


orgin_var_names = ['SeriousDlqin2yrs','RevolvingUtilizationOfUnsecuredLines',
              'age','NumberOfTime30-59DaysPastDueNotWorse',
              'DebtRatio','MonthlyIncome',
              'NumberOfOpenCreditLinesAndLoans',
              'NumberOfTimes90DaysLate',
              'NumberOfTime60-89DaysPastDueNotWorse',
              'NumberRealEstateLoansOrLines',
              'NumberOfDependents']

#plot all histogram 
for each in origin_var_names:
    plot_histogram(df,each)
              
#log transform data that distributions are distorted by outliers 

df['RUOU_log'] = np.log(df['RevolvingUtilizationOfUnsecuredLines']+1)
df['Number30-50_log']= np.log(df['NumberOfTime30-59DaysPastDueNotWorse']+1)
df['DebtRatio_log'] = np.log(df['DebtRatio']+1)
df['Number90_log'] = np.log(df['NumberOfTimes90DaysLate']+1)
df['Number60-89_log'] = np.log(df['NumberOfTime60-89DaysPastDueNotWorse']+1)
df['NumberRealEastate_log'] = np.log(df['NumberRealEstateLoansOrLines']+1)

log_var_names = ['RUOU_log','Number30-50_log','DebtRatio_log','Number90_log','Number60-89_log',
                 'NumberRealEastate_log']
for each in log_var_names:
    plot_histogram(df,each)

#checking: for a given DataFrame, calculates how many values for 
    #each variable is null and prints the resulting table 
def print_null_freq(df):
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    return pd.crosstab(df_lng.variable, null_variables)

#Following procedure imputes missing data from K_nearest_neighbor regression
from sklearn.neighbors import KNeighborsRegressor

#define a function to split data into two categories: one with missing values, and one without
def get_nonull(df,var):
    data_nonull = df[df[var].isnull()==False]   
    return data_nonull
df_without_null = get_nonull(df,'MonthlyIncome')
def get_null(df,var):
    data_null = df[df[var].isnull()==True]
    return data_null
df_with_null = get_null(df,'MonthlyIncome')

#check for correlation 
df_without_null.corr() #select the most highly correlated data for 

#define a function to fill missing data based on kneighborhood imputation of 
#the two most correlated variables 
def fill_kneighb_miss(var1, var2, var_to_fill, num_neighbors):
    '''var1, var2 are variables most closely related to the variable that
    is going to be filled
    the function fills the missing data with k nears neighbor imputation, where k=num_neighbors
    '''
    cols = [var1,var2]
    fill_var = df_without_null[var_to_fill]
    iv_imputer = KNeighborsRegressor(n_neighbors=num_neighbors)
    iv_imputer.fit(df_without_null[cols], fill_var)
    new_values = iv_imputer.predict(df_with_null[cols])
    df_with_null[var_to_fill]=new_values
    filleddata = df_without_null.append(df_with_null)
    return filleddata
#test
fill_data = fill_kneighb_miss('Number90_log','Number60-89_log','MonthlyIncome')

#fill missing value for number of dependency, becaue the person probably do not have dependents if he/she did not report one 
filled_data = fill_data.fillna(0)

import re

def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    s2 = re.sub('-', '', s1)
    s3 = re.sub(': ', '_', s2)
    s4 = re.sub('0.1' , '0_1', s3 )
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s4).lower()

#this function distinguish categorical data between continuous data 
#****can't use eval, need to fix this part of code 
def type_variable(df):
    df.columns = [camel_to_snake(col) for col in list(df.columns.values)]
    for i in range(len(df.columns)):
        var_name = str(list(df.columns.values)[i])
        command1 = type(list(df[var_name].unique())[0]) == str
        command2 = len(list(df[var_name].unique())) >= 4
        if command1==False and command2:
            print (var_name, 'continuous')
        else:
            print (var_name,'categorical')

type_variable(filled_data)

#get dummny variables from categorical variables 
def dummy_data(df, cate_var):
    df_dummy = pd.get_dummies(df[cate_var])
    df_new = df.join(df_dummy)
    return df_new
df_dummied = dummy_data(filled_data,'SeriousDlqin2yrs')

=====================================================================
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn import preprocessing
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time


clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

grid = { 
    'RF':{'n_estimators': [1,10,100,1000], 'max_depth': [1,5,10,20], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.1,1,5,10]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
   
models_to_run=['LR','KNN','DT','RF', 'GB','SVM']

#converting pandas dataframe to numpy matrix 
y = filled_data['SeriousDlqin2yrs'].as_matrix().astype(np.int)
filled_data.drop(['SeriousDlqin2yrs','Unnamed: 0','RevolvingUtilizationOfUnsecuredLines',
              'NumberOfTime30-59DaysPastDueNotWorse',
              'DebtRatio',
              'NumberOfTimes90DaysLate',
              'NumberOfTime60-89DaysPastDueNotWorse',
              'NumberRealEstateLoansOrLines'],axis=1, inplace=True) #only keeping the log transformed values 
X = filled_data.as_matrix().astype(np.float)
X.shape



with open('parameters-table.csv', 'wb') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        w.writerow(['MODEL', 'Accuracy', 'Precision', 'Recall', 'F1'])
        for index,clf in enumerate([clfs[x] for x in models_to_run[0:1]]):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            parameter_values=grid[models_to_run[index]]
            print models_to_run[index]
            try:
                for p in ParameterGrid(parameter_values):
                    clf.set_params(**p)
                    print clf
                    y_pred = clf.fit(X_train, y_train).predict(X_test)
                    accuracy = metrics.accuracy_score(y_test, y_pred) 
                    precision = metrics.precision_score(y_test, y_pred) 
                    recall = metrics.recall_score(y_test, y_pred) 
                # f1 calculation is F1 = 2 * (precision * recall) / (precision + recall)
                    f1 = metrics.f1_score(y_test, y_pred)
                    w.writerow([clf, accuracy, precision, recall, f1])
            except:
                pass
#check the best 5 models in terms of F1 score
reader = csv.reader(open("HW3_model_comparison.csv"), delimiter=",")
import operator
sortedlist = sorted(reader, key=operator.itemgetter(4), reverse=True)

#top performing classifiers
sortedlist[0:6]


#robestness test
#coerce all the feature values into similar range
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
    '''
    This function returns the entire predicted label value, label data being replaced by the predicted value
    the stratified_k_fold function loops through the n_folds to get y_pred
    '''
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    for train_index, test_index in stratified_k_fold:
        X_train, X_test = X[train_index],X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index]=clf.predict(X_test)
    return y_pred


def evaluate_model(y, y_pred):
    '''
    Compare the label of the test data to predicted values
    and return accuracy, precision, recall, and f1 score.
    '''
    accuracy = metrics.accuracy_score(y, y_pred) 
    precision = metrics.precision_score(y, y_pred) 
    recall = metrics.recall_score(y, y_pred) 
    # f1 calculation is F1 = 2 * (precision * recall) / (precision + recall)
    f1 = metrics.f1_score(y, y_pred) 

#evaluate top five models 
#1
evaluate_model(y, stratified_cv(X,y,ensemble.GradientBoostingClassifier,init=None, learning_rate=0.001, loss='deviance',
    max_depth=3, max_features=None, max_leaf_nodes=None,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0, n_estimators=10000,
    presort='auto', random_state=None, subsample=0.1, verbose=0,
    warm_start=False))
#2
evaluate_model(y,stratified_cv(X,y,tree.DecisionTreeClassifier,class_weight=None, criterion='gini', max_depth=10,
    max_features='sqrt', max_leaf_nodes=None, min_samples_leaf=1,
    min_samples_split=10, min_weight_fraction_leaf=0.0,
    presort=False, random_state=None, splitter='best'))

#3
evaluate_model(y,stratified_cv(X,y,ensemble.GradientBoostingClassifier,init=None, learning_rate=0.001, loss='deviance',
    max_depth=5, max_features=None, max_leaf_nodes=None,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0, n_estimators=10000,
    presort='auto', random_state=None, subsample=0.1, verbose=0,
    warm_start=False))
#4
evaluate_model(y,stratified_cv(X,y,tree.DecisionTreeClassifier,class_weight=None, criterion='gini', max_depth=20,
max_features='log2', max_leaf_nodes=None, min_samples_leaf=1,
min_samples_split=10, min_weight_fraction_leaf=0.0,
presort=False, random_state=None, splitter='best'))

#5
evaluate_model(y,stratified_cv(X,y,ensemble.GradientBoostingClassifier,init=None, learning_rate=0.001, loss='deviance',    
 max_depth=3, max_features=None, max_leaf_nodes=None,
 min_samples_leaf=1, min_samples_split=2, 
 min_weight_fraction_leaf=0.0, n_estimators=10000,
 presort='auto', random_state=None, subsample=0.5, verbose=0,
 warm_start=False))
 

def plot_precision_recall(y_true, y_prob, model_name, model_params):
    '''
    Plot a precision recall curve for one model with its y_prob values.
    '''
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision_curve[:-1]
    recall = recall_curve[:-1]
    plt.clf() #clear the figure
    plt.plot(recall, precision, label='%s' % model_params)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title("Precision Recall Curve for %s" %model_name)
    plt.savefig(model_name)
    plt.legend(loc="lower right")

#plot precision_recall_curve for the top one model
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
original_params = {'learning_rate': 0.001, 'max_depth':3,
                    'n_estimators':10000,'subsample':0.1,'warm_start':False}
params = dict(original_params)

clf = ensemble.GradientBoostingClassifier(**params)
y_pred = clf.fit(X_train, y_train).predict(X_test)
plot_precision_recall(y_test, y_pred, 'GradientBoosting', 'learn_rate=0.001,max_depth=3,n_estimator=10000,subsampler=0.1')



#grid to find the best estimator and parameter 

param_grid = {
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 9, 15],
              'n_estimators': [1000, 2000, 3000],
              }

est = ensemble.GradientBoostingRegressor()


start_time = time.time()
gs_cv = grid_search.GridSearchCV(est, param_grid, n_jobs=4).fit(X_train, y_train)
end_time = time.time()

print('It took {:.2f} seconds'.format(end_time - start_time))
# best hyperparameter setting
gs_cv.best_params_
=================================================================
