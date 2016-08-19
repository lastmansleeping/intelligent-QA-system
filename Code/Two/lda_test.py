import lda
import lda.datasets
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import math
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import time
from sklearn.preprocessing import Imputer
import pickle

#Latent Dirichlet Allocation

def buildLDA(features_df):
    #Vectorize the sentences
    sentences = features_df['Sentence']
    vectorizer = CountVectorizer(max_features=1000, analyzer='word')
    X = vectorizer.fit_transform(sentences)
    vocabulary = vectorizer.get_feature_names()
    
    #Cluster the vectors
    model = lda.LDA(n_topics = 20, n_iter = 500, random_state = 1)
    model.fit(X)
    topic_word = model.topic_word_
    doc_topic = model.doc_topic_
    return model, vocabulary

def getTopicVector(sentence, model, vectorizer):
    #model, vocabulary = buildLDA()
    #vectorizer = CountVectorizer(analyzer='word', vocabulary = vocabulary)
    #sentence = sentence.decode('utf-8','ignore').encode("utf-8")
    #print sentence
    X = vectorizer.fit_transform([sentence])
    topic_vector = model.transform(X)
    return topic_vector

#Getting average vectors
def getAverageVectors(topic_vectors):
    average_topic_vectors = {}
    for key, value in topic_vectors.iteritems():
        average_topic_vectors[key] = {}
        average_topic_vectors[key][0] = np.squeeze(np.asarray(np.matrix(topic_vectors[key][0]).sum(axis = 0)))/len(topic_vectors[key][0])
        average_topic_vectors[key][1] = np.squeeze(np.asarray(np.matrix(topic_vectors[key][1]).sum(axis = 0)))/len(topic_vectors[key][1])
        average_topic_vectors[key][0] = normalizeVector(average_topic_vectors[key][0])
        average_topic_vectors[key][1] = normalizeVector(average_topic_vectors[key][1])
        
    return average_topic_vectors

def normalizeVector(vector):
    vector_magnitude = math.sqrt(sum([x * x for x in vector]))
    vector = [x/vector_magnitude for x in vector]
    return vector

def getLDAFeature():
    #Training the model and soft clustering sentences
    features_df = pd.read_table('C:\Users\JareD\Major Project\EvenSem\Data\DataSet_Feature_Extraction_3.tsv')
    features_df_train = features_df[features_df['Type'] == 'train']
    model, vocabulary = buildLDA(features_df_train)
    vectorizer = CountVectorizer(analyzer='word', vocabulary = vocabulary)
    #x = 1
    topic_vectors = {}
    
    for question_type in features_df_train['QuestionType'].unique():
        topic_vectors[question_type] = {1 : [], 0 : []}
        df = features_df_train[features_df_train['QuestionType'] == question_type]
        #df = df.head(10)
        for index, row in df.iterrows():
            topic_vector = getTopicVector(row['Sentence'], model, vectorizer)
            if row['Label'] == 1:
                topic_vectors[question_type][1].append(topic_vector[0])
            else:
                topic_vectors[question_type][0].append(topic_vector[0])
    
    
    #x = 2
    average_topic_vectors = getAverageVectors(topic_vectors)
    average_topic_vectors['ABBR'] = {}
    average_topic_vectors['ABBR'][0] = 20 * [0]
    average_topic_vectors['ABBR'][1] = 20 * [0]
    pickle.dump(topic_vectors, open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\Topic_Vectors.pkl", "wb"))
    pickle.dump(average_topic_vectors, open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\Average_Topic_Vectors_2.pkl", "wb"))
    pickle.dump(model, open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\LDA_Model.pkl", "wb"))
    pickle.dump(vectorizer, open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\CountVectorizer.pkl", "wb"))#print average_topic_vectors
    #x = 3
    #Getting cosine similarity
    """for key, value in average_topic_vectors.iteritems():
                    print key
                    print "0"
                    print value[0]
                    print "1"
                    print value[1]"""
    sentence = "worldwide it affects 3 5 million people and causes 100 000 130,000 deaths a year"
    topic_vector = getTopicVector(sentence, model, vectorizer)
    topic_vector = normalizeVector(topic_vector[0])
    positive_average_vector = Imputer().fit_transform(np.array(average_topic_vectors['DESC'][1]).reshape(1, -1))
    negative_average_vector = Imputer().fit_transform(np.array(average_topic_vectors['DESC'][0]).reshape(1, -1))
    topic_vector = Imputer().fit_transform(np.array(topic_vector).reshape(1, -1))
    positive_similarity = cosine_similarity(positive_average_vector, topic_vector)
    negative_similarity = cosine_similarity(negative_average_vector, topic_vector)
    #print positive_similarity, negative_similarity
    #x = 4
    return positive_similarity, negative_similarity
    
    

positive_similarity, negative_similarity = getLDAFeature()       
print positive_similarity, negative_similarity
