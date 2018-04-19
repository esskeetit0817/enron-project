#!/usr/bin/python
#coding=utf-8

import sys
import pickle
import warnings
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, load_classifier_and_data, test_classifier
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score

# pre-processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### 将data_dict转化为DataFrame
df = pd.DataFrame(data_dict)
df = df.T

# move 'poi' column to head of list. this will be the label
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('poi')))
df = df.reindex(columns = cols)

def data_info():
    print'Number of persons:', len(df.index)
    print'Number of features:', len(df.columns)
    print'Number of data points:', len(df.index) * len(df.columns)
    print'Number of POIs:', len(df[df['poi'] == True])
    print'Number of non-POIs:', len(df[df['poi'] == False])


print '\nOriginal Dataset Info:\n',df.columns
data_info()


### Task 2: Remove outliers
### 找出数据集中NaN的数量
df.replace('NaN', np.nan, inplace = True)
features_nan_count = df.isnull().sum(axis = 0).sort_values(ascending=False)
person_nan_count = df.isnull().sum(axis = 1).sort_values(ascending=False)

print'\nTop 10 features NaN:\n',features_nan_count[:10]
print'\nTop 10 person NaN:\n\n',person_nan_count[:10]
df = df.fillna(0)   ### 将'NaN'值用0替代

### 绘制数据中salary和bonus散点图
plt.scatter( df["salary"], df["bonus"])
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

### 找出salary和bonus散点图中的异常值
print "the outlier in the salary-bonus scatter is ",df['salary'].idxmax()

df = df.drop(['LOCKHART EUGENE E','TOTAL','THE TRAVEL AGENCY IN THE PARK'], axis=0)

### 处理后的数据信息
print'\nProcessed Dataset Info:\n',df.columns
data_info()


### 删除email_address这个无用特征
df = df.drop(['email_address'], axis=1) 

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
### 创建三个新特征
df['from_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_ratio'] = df['from_poi_ratio'].fillna(0)

df['sent_to_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['sent_to_poi_ratio'] = df['sent_to_poi_ratio'].fillna(0)

df['shared_with_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']
df['shared_with_poi_ratio'] = df['shared_with_poi_ratio'].fillna(0)

df_features = df.drop('poi', axis=1)
df_labels = df['poi']

### 使用SelectKBest计算特征的得分，并排序：
k_best = SelectKBest(k='all')
k_best.fit(df_features, df_labels)


print'\nFeatures KBest Score:\n',pd.DataFrame( \
    k_best.scores_,index=df_features.columns.tolist(),columns=['score']).sort_values('score', ascending=False)

df_features = df[['exercised_stock_options', 'total_stock_value',\
    'sent_to_poi_ratio','restricted_stock', 'shared_with_poi_ratio','total_payments']]
df_labels = df['poi']




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
algorithms = [
    GaussianNB(),
    make_pipeline(MinMaxScaler(), SVC(class_weight='balanced', random_state=42)),
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    RandomForestClassifier(class_weight='balanced', random_state=42),
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(class_weight='balanced'), random_state=42)
]
params = [
    { },  # GaussianNB

    {  # SVC pipeline
        'svc__kernel' : ('linear', 'rbf', 'poly', 'sigmoid'),
        'svc__C' : [0.1, 1.0, 10, 100, 1000, 10000],
        'svc__gamma' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]
    },
    {  # DecisionTreeClassifier
        'criterion' : ('gini', 'entropy'),
        'splitter' : ('best', 'random'),
        'min_samples_split' : [7, 8, 9, 10, 11, 12, 13]
    },
    {  # RandomForestClassifier
        'n_estimators' : [2,3,4,5,6,7,8],
        'criterion' : ('gini', 'entropy'),
        'min_samples_split' : [25,30,35,40,45,50,55]
    },
    {  # AdaBoostClassifier
        'n_estimators' : [5, 10, 50],
        'learning_rate' : [0.001, 0.01, 0.1, 1.0]
    }
]
features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(df_features,df_labels,test_size=0.3, random_state=42)

dataset = df.to_dict(orient='index') #dataframe数据结构转换为list
feature_list = ['poi']+df_features.columns.tolist()

best_estimator = None  
best_score = 0  
print'\nChoose Best Model for each Algorithm (F1 Score):\n'
warnings.filterwarnings('ignore') # 忽视f1警告错误

for ii, algorithm in enumerate(algorithms):
    model = GridSearchCV(algorithm, params[ii],scoring = 'f1',cv=5) 
    model.fit(features_train,labels_train)

    best_estimator_ii = model.best_estimator_
    best_score_ii = model.best_score_

    print'\nF1 Score:',best_score_ii,'\n'
    test_classifier(best_estimator_ii, dataset, feature_list)

    if best_score_ii > best_score:
        best_estimator = best_estimator_ii
        best_score = best_score_ii

clf =  best_estimator
print'\nThe Highest F1 Score Model:',clf

### Extract features and labels from dataset for local testing


# Provided to give you a starting point. Try a variety of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
### 尝试使用的算法列表



# Example starting point. Try investigating other evaluation techniques!


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dataset = df.to_dict(orient='index')
feature_list = ['poi'] + df_features.columns.tolist()
dump_classifier_and_data(clf, dataset, feature_list )
