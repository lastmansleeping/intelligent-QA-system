import string, re
import nltk
import math
from sets import Set
from sklearn.linear_model import LogisticRegression
import time
from sklearn.externals import joblib
import gensim
import operator
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import time
from nltk.corpus import wordnet as wn
import lda
import lda.datasets
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import math
import warnings
import pickle
from sklearn.preprocessing import Imputer


#%matplotlib inline
plt.style.use('ggplot')
from sklearn.linear_model import LogisticRegression

#Train the model on Logistic Regression
def buildModel(data_type, features_df, flag):
    features = ['UnigramCount', 'BigramCount', 'TrigramCount', 'LemmaCount', 'IDFUnigramCount']
    features = features + ['SynonymCount', 'AntonymCount', 'HypernymCount', 'HyponymCount']
    features = features + ['LevenshteinEditDistance']
    if flag == 1:
        features = features + ['PositiveSimilarity', 'NegativeSimilarity']
    
    X_Train = features_df[features][features_df['Type'] == 'train']
    y_Train = features_df['Label'][features_df['Type'] == 'train']

    model = LogisticRegression()
    model = model.fit(X_Train, y_Train)

    #Test Set
    X_Test = features_df[features][features_df['Type'] == data_type]
    y_Test = features_df['Label'][features_df['Type'] == data_type]

    #y_Test_Predicted = model.predict(X_Test)
    y_Test_Predicted_Probabilites = model.predict_proba(X_Test)
    return y_Test_Predicted_Probabilites


#Calculate MRR
def getMRR(df):
    question_ids = df['QuestionID'].unique()
    rankSum = 0
    for qid in question_ids:
        proba_list = df[df['QuestionID'] == qid]['Probability'].tolist()
        label_list = df[df['QuestionID'] == qid]['Label'].tolist()
        lst = zip(proba_list, label_list)
        lst = np.array(lst)
        lst = lst[np.argsort(lst[:, 0])]
        lst = lst[::-1]
        for i in range(len(lst)):
            if lst[i][1] == 1:
                rankSum = rankSum + (1/ float(i + 1))
                break
                #print qid, i + 1
    mean_reciprocal_rank = rankSum/ float(len(question_ids))
    #mrr = rankSum/ float(len(question_ids))
    return mean_reciprocal_rank

#Calculate MAP
def getMAP(df):
    question_ids = df['QuestionID'].unique()
    average_precision_measures = []
    for qid in question_ids:
        proba_list = df[df['QuestionID'] == qid]['Probability'].tolist()
        label_list = df[df['QuestionID'] == qid]['Label'].tolist()
        lst = zip(proba_list, label_list)
        lst = np.array(lst)
        lst = lst[np.argsort(lst[:, 0])]
        lst = lst[::-1]
        no_of_positives = 0
        no_of_documents = 0
        precision_measures = []
        for i in range(len(lst)):
            no_of_documents = no_of_documents + 1
            if lst[i][1] == 1:
                no_of_positives = no_of_positives + 1
                precision = no_of_positives / float(no_of_documents)
                precision_measures.append(precision)
                #print qid, i + 1
        average_precision = np.mean(precision_measures)
        #print precision_measures, average_precision
        average_precision_measures.append(average_precision)
    mean_average_precision = np.mean(average_precision_measures)
    #mrr = rankSum/ float(len(question_ids))
    return mean_average_precision

def evaluateModel(data_type, features_df, question_type):
    flag = 1
    if question_type == 'ENTY':
        flag = 0
    y_Test_Predicted_Probabilites = buildModel(data_type, features_df, flag)
    df1 = features_df[features_df['Type'] == data_type]
    df1 = df1.reset_index(drop=True)
    df1 = df1[['QuestionID', 'SentenceID', 'Label', 'Type']]
    df2 = pd.Series(y_Test_Predicted_Probabilites[:, 1], name='Probability')
    df = pd.concat([df1, df2], axis = 1)
    """
    print "Mean Reciprocal Rank : ", getMRR(df)
    print "Mean Average Precision : ", getMAP(df)
    """
    return {'MAP' : getMAP(df), 'MRR' : getMRR(df)}
    

"""print "Dev\t:", evaluateModel('dev', features_df)
print "Test\t:", evaluateModel('test', features_df)
print "Train\t:", evaluateModel('train', features_df)"""

#Evaluating performance based on question classes
def evaluateModel2(features_df):
    evaluation_df = pd.DataFrame(columns = ['QuestionType', 'number_of_questions', 'train_MAP', 'test_MAP', 'dev_MAP', 'train_MRR', 'test_MRR', 'dev_MRR'])
    question_types = features_df['QuestionType'].unique()
    for question_type in question_types:
        row = {}
        #print question_type
        df = features_df[features_df['QuestionType'] == question_type]
        row['QuestionType'] = question_type
        row['number_of_questions'] = len(df.groupby(['QuestionID', 'Type']).size())
        for data_type in ['train', 'test', 'dev']:
            result = evaluateModel(data_type, df, question_type)
            row[data_type + '_MAP'] = result['MAP']
            row[data_type + '_MRR'] = result['MRR']
        evaluation_df = evaluation_df.append(pd.DataFrame([row]), ignore_index = True)
    return evaluation_df 

def plotEvaluation():
    features_df = pd.read_table('C:\Users\JareD\Major Project\EvenSem\Data\DataSet_Feature_Extraction_3.tsv')
    evaluation_df = evaluateModel2(features_df)
    plot_df = evaluation_df[['QuestionType', 'train_MAP', 'test_MAP', 'dev_MAP', 'train_MRR', 'test_MRR', 'dev_MRR']]
    #plot_df = evaluation_df[['QuestionType', 'dev_MAP']]
    plot = plot_df.plot(kind='bar', figsize=(18, 10))
    plot.set_title("Plot of Question Type vs. Performance")
    x_tick_labels = plot_df['QuestionType']
    x_tick_labels = [str(x) + "\n[" + str(int(evaluation_df[evaluation_df['QuestionType'] == x]['number_of_questions'])) + "]" for x in x_tick_labels]
    plot.set_xticklabels(x_tick_labels, rotation = 0)
    plt.xlabel('Question Types')
    plt.ylabel('MAP and MRR scores')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    plt.show()
    print evaluation_df


plotEvaluation()