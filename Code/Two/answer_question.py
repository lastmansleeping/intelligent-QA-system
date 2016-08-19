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
from nltk.stem.snowball import SnowballStemmer


idfWeights = {}

def removePunctuation(inputString):
    #inputString = ' '.join(inputString.replace("'s", '').split())
    #inputString = re.sub('[%s]' % re.escape(string.punctuation), '', inputString)
    #return re.sub( '\s+', ' ', inputString).strip()
    return re.sub('[%s]' % re.escape(string.punctuation), '', inputString)

def removeStopWords(tokens):
    return [t for t in tokens if t not in nltk.corpus.stopwords.words('english')]

def getBigrams(tokens):
    bigrams = []
    for i in range(len(tokens) - 1):
        bigram = tokens[i], tokens[i + 1]
        bigrams.append(bigram)
    return bigrams

def getTrigrams(tokens):
    trigrams = []
    for i in range(len(tokens) - 2):
        trigram = tokens[i], tokens[i + 1], tokens[i + 2]
        trigrams.append(trigram)
    return trigrams
    
def getLemmas(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(t.decode('utf-8')) for t in tokens]
    
def getIDFWeights(df):
    IDFWeights = {}
    frequencies = {}
    no_of_documents = 0
    for index, row in df.iterrows():
        no_of_documents = no_of_documents + 1
        sentence = row['Sentence']
        sentence = removePunctuation(sentence)
        sentence = sentence.lower()
        sTokens = nltk.word_tokenize(sentence)
        sTokens = Set(sTokens)
        for token in sTokens:
            if frequencies.has_key(token):
                frequencies[token] = frequencies[token] + 1
            else:
                frequencies[token] = 1
    
    for index, row in df.iterrows():
        sentence = row['Sentence']
        sentence = removePunctuation(sentence)
        sentence = sentence.lower()
        sTokens = nltk.word_tokenize(sentence)
        sTokens = Set(sTokens)
        sentDict = {}
        for token in sTokens:
            idfWeight = math.log(no_of_documents) - math.log(frequencies[token])
            sentDict[token] = idfWeight
        IDFWeights[row['SentenceID']] = sentDict
    return IDFWeights

def getWordnetDict(word):
    synonyms = []
    antonyms = []
    hypernyms = []
    hyponyms = []
    
    for synset in wn.synsets(word):
        #synonyms and antonyms
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
        
        #hypernyms
        if synset.hypernyms():
            hypernyms.extend(synset.hypernyms()[0].lemma_names())
        
        #hyponyms    
        for y in [x.lemma_names() for x in synset.hyponyms()]:
            hyponyms.extend(y)
    
    wordnet_dict = {'synonyms': list(set(synonyms)), 'antonyms': list(set(antonyms)), 'hypernyms': list(set(hypernyms)), 'hyponyms': list(set(hyponyms))}    
    return wordnet_dict

def getWordnetFeatures(question, sentence):
    
    #tokenize question and sentences
    qTokens = nltk.word_tokenize(question)
    sTokens = nltk.word_tokenize(sentence)
    
    #remove stop words
    qTokens = removeStopWords(qTokens)
    sTokens = removeStopWords(sTokens)
    
    #Lemmatize tokens
    qTokens = getLemmas(qTokens)
    sTokens = getLemmas(sTokens)
    
    synonym_count = 0
    antonym_count = 0
    hypernym_count = 0
    hyponym_count = 0
    
    for q_token in qTokens:
        q_token_wordnet_dict = getWordnetDict(q_token)
        synonyms = q_token_wordnet_dict['synonyms']
        antonyms = q_token_wordnet_dict['antonyms']
        hypernyms = q_token_wordnet_dict['hypernyms']
        hyponyms = q_token_wordnet_dict['hyponyms']
        for synonym in synonyms:
            try:
                if synonym in sTokens:
                    synonym_count = synonym_count + 1
                    break
            except:
                print synonym, sTokens
                
        for antonym in antonyms:
            try:
                if antonym in sTokens:
                    antonym_count = antonym_count + 1
                    break
            except:
                print antonym, sTokens
        
        for hypernym in hypernyms:
            try:
                if hypernym in sTokens:
                    hypernym_count = hypernym_count + 1
                    break
            except:
                print hypernym, sTokens
        
        for hyponym in hyponyms:
            try:
                if hyponym in sTokens:
                    hyponym_count = hyponym_count + 1
                    break
            except:
                print hyponym, sTokens
    
    return {'SynonymCount' : synonym_count, 'AntonymCount' : antonym_count, 'HypernymCount' : hypernym_count, 'HyponymCount' : hyponym_count}

#Levenshtein Edit Distance
def getLevenshteinEditDistance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

#------------------------------------------------------------------------------------------------
#Question Classification

#Question Classification Variables
question_classifier = joblib.load('C:\Users\JareD\Major Project\EvenSem\Models\Question Classifier\question_classifier.pkl') 
word_vector_path = "C:\Users\JareD\Major Project\EvenSem\Question Classification\question-classification-master\data\glove.6B.50d.txt"
word_vector = gensim.models.Word2Vec.load_word2vec_format(word_vector_path, binary=False)
vector_dim = 50

def getQuestionType(question):
	question = removePunctuation(question)
	return question_classifier.predict(np.array(average_vector(word_vector, question.lower())).reshape(1, -1))[0]

def average_vector2(dictionary, question):
    cnt = 0
    s = [0]*vector_dim
    for w in question.split(" "):
        w = w.lower()
        cnt += 1
        try:
            # print word, word_vector[word]
            s = map(operator.add, dictionary[w], s)
        except KeyError:
            cnt -= 1
            # pass #Use random vector or skip?
#             s = map(operator.add, dictionary.seeded_vector(random_generator(50)), s)
    if cnt == 0:
        return s
    return [elem/float(cnt) for elem in s]

def average_vector(dictionary, question):
    splitted = question.split(" ")
    s = [0]*vector_dim
    cnt = 2.0
    try:
        if (len(splitted) == 0):
            return s
        else:
            s = map(operator.add, dictionary[splitted[0].lower()], s)
            if (len(splitted) <= 1):
                return s
            s = map(operator.add, dictionary[splitted[1].lower()], s)
            if (splitted[0].lower() == 'what' and splitted[1].lower() == 'is'):
                return average_vector2(dictionary, question)
#                 s = map(operator.add, dictionary[splitted[3].lower()], s)
#                 cnt += 1.0
            return [elem/cnt for elem in s]         
    except KeyError:
        return s 
#-------------------------------------------------------------------------------------------------   

#Getting LDA Features
def getTopicVector(sentence, model, vectorizer):
    #model, vocabulary = buildLDA()
    #vectorizer = CountVectorizer(analyzer='word', vocabulary = vocabulary)
    #sentence = sentence.decode('utf-8','ignore').encode("utf-8")
    #print sentence
    X = vectorizer.fit_transform([sentence])
    topic_vector = model.transform(X)
    return topic_vector

def normalizeVector(vector):
    vector_magnitude = math.sqrt(sum([x * x for x in vector]))
    vector = [x/vector_magnitude for x in vector]
    return vector

#Load LDA Models
topic_vectors = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\Topic_Vectors.pkl", "rb"))
average_topic_vectors = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\Average_Topic_Vectors_2.pkl", "rb"))
lda_model = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\LDA_Model.pkl", "rb"))
vectorizer = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\CountVectorizer.pkl", "rb"))

#Main function
def getFeatures(df, question):
    #Make a dataframe and groupby QuestionID
    columns = ['DocumentID', 'SentenceID']
    columns = columns + ['Sentence']
    features = ['UnigramCount', 'BigramCount', 'TrigramCount', 'LemmaCount', 'IDFUnigramCount']
    features = features + ['SynonymCount', 'AntonymCount', 'HypernymCount', 'HyponymCount']
    features = features + ['LevenshteinEditDistance']
    features = features + ['PositiveSimilarity', 'NegativeSimilarity']
    dfFeatures = pd.DataFrame(columns = columns + features)
    
    #Preprocess question
    question = removePunctuation(question)
    question = question.lower()
    qTokens = nltk.word_tokenize(question)
    qBigrams = getBigrams(qTokens)
    qTrigrams = getTrigrams(qTokens)
    qTokens = removeStopWords(qTokens)
    qLemmas = getLemmas(qTokens)
    question_type = getQuestionType(question)


    #Get IDF Vectors
    IDFWeights = getIDFWeights(df)
    for index, row in df.iterrows():
        #empty dictionary
        featureVector = {}
        
        #question = row['Question']
        sentence = row['Sentence']

        #Remove punctuation 
        #question = removePunctuation(question)
        sentence = removePunctuation(sentence)

        #Convert to Lower case
        #question = question.lower()
        sentence = sentence.lower()

        featureVector['DocumentID'] = row['DocumentID']
        featureVector['SentenceID'] = row['SentenceID']
        featureVector['Sentence'] = sentence
        
        #tokenize
        #qTokens = nltk.word_tokenize(question)
        sTokens = nltk.word_tokenize(sentence)
        
        #print len(qTokens), len(sTokens)
        
        #Bigram count
        #qBigrams = getBigrams(qTokens)
        sBigrams = getBigrams(sTokens)
        bigramCount = 0
        for i in qBigrams:
            try:
                if i in sBigrams:
                    bigramCount = bigramCount + 1
            except:
                print i, sTokens
        featureVector['BigramCount'] = bigramCount
        
        #Trigram count
        #qTrigrams = getTrigrams(qTokens)
        sTrigrams = getTrigrams(sTokens)
        trigramCount = 0
        for i in qTrigrams:
            try:
                if i in sTrigrams:
                    trigramCount = trigramCount + 1
            except:
                print i, sTokens
        featureVector['TrigramCount'] = trigramCount
        
        #Remove stop words
        #qTokens = removeStopWords(qTokens)
        sTokens = removeStopWords(sTokens)
        
        #Unigram Count and IDF unigram count
        IDFUnigramCount = 0
        unigramCount = 0
        for i in qTokens:
            try:
                if i in sTokens:
                    unigramCount = unigramCount + 1
                    IDFUnigramCount = IDFUnigramCount + IDFWeights[row['SentenceID']][i]
            except:
                print i, sTokens
        featureVector['UnigramCount'] = unigramCount
        featureVector['IDFUnigramCount'] = IDFUnigramCount
        
        #Parent Lemma Unigram Count
        #qLemmas = getLemmas(qTokens)
        sLemmas = getLemmas(sTokens)
        lemmaCount = 0
        for i in qLemmas:
            try:
                if i in sLemmas:
                    lemmaCount = lemmaCount + 1
            except:
                    print i, sTokens
        featureVector['LemmaCount'] = lemmaCount
        
        #Wordnet Features
        wordnet_features = getWordnetFeatures(question, sentence)
        featureVector['SynonymCount'] = wordnet_features['SynonymCount'] - lemmaCount
        featureVector['AntonymCount'] = wordnet_features['AntonymCount']
        featureVector['HypernymCount'] = wordnet_features['HypernymCount']
        featureVector['HyponymCount'] = wordnet_features['HyponymCount']
        
        #Levenshtein Distance
        levenshtein_distance = getLevenshteinEditDistance(qLemmas, sLemmas)
        featureVector['LevenshteinEditDistance'] = levenshtein_distance
        
        #Question Classification
        #question_type = getQuestionType(question)
        featureVector['QuestionType'] = question_type
        
        #LDA features
        topic_vector = getTopicVector(sentence, lda_model, vectorizer)
        topic_vector = normalizeVector(topic_vector[0])
        #print average_topic_vectors[question_type][0]
        #print average_topic_vectors[question_type][1]
        positive_average_vector = np.array(average_topic_vectors[question_type][1]).reshape(1, -1)
        negative_average_vector = np.array(average_topic_vectors[question_type][0]).reshape(1, -1)
        #print negative_average_vector
        topic_vector = np.array(topic_vector).reshape(1, -1)
        positive_similarity = cosine_similarity(positive_average_vector, topic_vector)
        #print positive_similarity
        negative_similarity = cosine_similarity(negative_average_vector, topic_vector)
        featureVector['PositiveSimilarity'] = positive_similarity[0][0]
        featureVector['NegativeSimilarity'] = negative_similarity[0][0]

        #Append features to dataframe
        dfFeatures = dfFeatures.append(pd.DataFrame([featureVector]), ignore_index = True)
    return dfFeatures, question_type
    
def getAnswers(features_df, question_type, question):
    features = ['UnigramCount', 'BigramCount', 'TrigramCount', 'LemmaCount', 'IDFUnigramCount']
    features = features + ['SynonymCount', 'AntonymCount', 'HypernymCount', 'HyponymCount']
    features = features + ['LevenshteinEditDistance']
    model = 0
    if question_type != 'ENTY':
        features = features + ['PositiveSimilarity', 'NegativeSimilarity']
        model = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\QA Classifier\qa_classifier_2.pkl", "rb"))
    else:
        model = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\QA Classifier\qa_classifier_1.pkl", "rb"))
    
    X = features_df[features]
    #y_Test_Predicted = model.predict(X_Test)
    
    y_Predicted_Probabilites = model.predict_proba(X)
    y_Predicted_Probabilites = np.array(y_Predicted_Probabilites)[:, 1]
    ranked_answer_list = zip(features_df['DocumentID'], features_df['SentenceID'], y_Predicted_Probabilites)
    #print ranked_answer_list
    ranked_answer_list = np.array(ranked_answer_list)
    ranked_answer_list = ranked_answer_list[np.argsort(ranked_answer_list[:, 2])]
    ranked_answer_list = ranked_answer_list[::-1]
    return ranked_answer_list[:3]
    #return y_Predicted_Probabilites

def getNextType(current_question_type):
    return 'DESC'
    
def predict_next_sentence(current_question_type, candidate_sentences, candidate_sentence_ids):
    lda_model = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\LDA_Model.pkl", "rb"))
    vectorizer = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\CountVectorizer.pkl", "rb"))
    average_topic_vectors = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\LDA\Average_Topic_Vectors_2.pkl", "rb"))
    next_question_type = getNextType(current_question_type)
    cosine_similarities = []
    for candidate_sentence in candidate_sentences:
        topic_vector = getTopicVector(candidate_sentence, lda_model, vectorizer)
        topic_vector = normalizeVector(topic_vector[0])
        positive_average_vector = np.array(average_topic_vectors[next_question_type][1]).reshape(1, -1)
        topic_vector = Imputer().fit_transform(np.array(topic_vector).reshape(1, -1))
        try:
        	positive_similarity = cosine_similarity(positive_average_vector, topic_vector)
        except:
        	print candidate_sentence
        	print topic_vector, positive_average_vector
        cosine_similarities.append(positive_similarity[0][0])
    
    candidate_sentences = zip(candidate_sentences, cosine_similarities, candidate_sentence_ids)
    return candidate_sentences


def getNextAnswer(current_question_type, candidate_sentences, candidate_sentence_ids):
    ranked_prediction = np.array(predict_next_sentence(current_question_type, candidate_sentences, candidate_sentence_ids))
    ranked_prediction = ranked_prediction[np.argsort(ranked_prediction[:, 1])]
    ranked_prediction = ranked_prediction[::-1]  
    #for i in ranked_prediction[:3]:
    #    print i[1], i[0]
    return ranked_prediction[:3]

#----------------------------------------------------
documents_sentences_df = pd.read_table("C:\Users\JareD\Major Project\EvenSem\Data\Documents_Sentences.tsv")
documents_clusters_df = pd.read_table("C:\Users\JareD\Major Project\EvenSem\Data\Documents_Clusters.tsv")
tfidf_vocabulary = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\IR\Tfidf_Vocabulary.pkl", "rb"))
tfidf_matrix = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\IR\Tfidf_Matrix.pkl", "rb"))
km = pickle.load(open("C:\Users\JareD\Major Project\EvenSem\Models\IR\KMeans.pkl", "rb"))
stemmer = SnowballStemmer("english")

def preprocess(inputString):
    inputString = inputString.lower()
    inputString = re.sub('[%s]' % re.escape(string.punctuation), '', inputString)
    inputString = ' '.join(inputString.replace("'s", '').split())
    return re.sub( '\s+', ' ', inputString).strip()

def queryVectorize(question):
    tokens = nltk.word_tokenize(question)
    queryTf = [stemmer.stem(token) for token in tokens]
    
    terms = tfidf_vocabulary
    tfVector = [0 for i in range(len(terms))]
    for term in queryTf:
        if term in terms:
            tfVector[terms.index(term)] = tfVector[terms.index(term)] + 1
            #print term
        #else:
            #print "----", term
    return np.array(tfVector)

def retrieveDocuments(question):
    question = preprocess(question)
    #print question
    question_vector = queryVectorize(question)
    #print question_vector
    cosine_similarities_clusters = cosine_similarity(question_vector.reshape(1, -1), km.cluster_centers_)[0]
    cosine_similarities_documents = cosine_similarity(question_vector.reshape(1, -1), tfidf_matrix)[0]
    documents_id = documents_clusters_df['DocumentID']
    ranked_list = zip(documents_id, cosine_similarities_documents)
    ranked_list = np.array(ranked_list)
    ranked_list = ranked_list[np.argsort(ranked_list[:, 1])]
    ranked_list = ranked_list[::-1]
    retrieved_documents = ranked_list[:, 0][:5]  
    print retrieved_documents
    return retrieved_documents

def loadDataset(question):
    retrieved_documents = retrieveDocuments(question)
    return documents_sentences_df[documents_sentences_df['DocumentID'].isin(retrieved_documents)]

def getFeaturesPipeline(question):
	
    df = loadDataset(question)
    print question
    #get features
    time_start = time.clock()
    features_df, question_type = getFeatures(df, question)
    ranked_answer_list = getAnswers(features_df, question_type, question)

    answer_list = []
    #Print answers
    for i in range(3):
    	print "------------------------"
        sentence_id = ranked_answer_list[i][1]
        sentence = df[df['SentenceID'] == sentence_id]['Sentence']
        print (i+1), ranked_answer_list[i][2], sentence.values[0]
        answer_list.append({'answer' : sentence.values[0], 'next_answer' : []})
        document_id = ranked_answer_list[i][0]
        sentences = features_df[features_df['DocumentID'] == document_id]['Sentence']
        sentence_ids = features_df[features_df['DocumentID'] == document_id]['SentenceID']
        print "Next Answer"
        next_answers = getNextAnswer(question_type, sentences, sentence_ids)
        for next_answer in next_answers:
        	sentence = df[df['SentenceID'] == next_answer[2]]['Sentence']
        	print sentence.values[0]
        	answer_list[i]['next_answer'].append(sentence.values[0])

    time_elapsed = (time.clock() - time_start)

    print "Time taken to answer question : " + str(time_elapsed - time_start)
    return answer_list


    
#getFeaturesPipeline("how much is 1 tablespoon of water")
#def mainPipeline():

    #get question
    

    #pre process question
    #get question type
    #get list of documents and clusters
    #make a dataframe
    #get ranked list of answers
    #for each answer, get ranked list of next answers


