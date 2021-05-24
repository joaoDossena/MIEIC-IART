from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt


import pandas as pd

# Get data from file
def readData(path, path_test):
    df = pd.read_csv(path, sep="	")
    test_df = pd.read_csv(path_test, sep="	")
    print("Data Read")
    return (df, test_df)

# --------------------------------------------------------------

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
# stemmer = PorterStemmer()

# Cleaning and tokenization of tweets
def processing(df):
    corpus = []
    for i in range(len(df)):
        # get tweet and remove usernames (@username) and links to pictures (https://t.co/link)
        tweet = re.sub('@[a-zA-Z0-9_]+|https?://t.co/[a-zA-Z0-9_]+|[^a-zA-Z]', ' ', df['Tweet text'][i])

        # to lower-case and tokenize
        tweet = tweet.lower().split()

        # Stemming and stop word removal
        # stemmed_tweet = ' '.join([stemmer.stem(w) for w in tweet if not w in set(stopwords.words('english'))])

        # Lemmatizing
        lemma_tweet = ' '.join([lemmatizer.lemmatize(w) for w in tweet if not w in set(stopwords.words('english'))])

        # corpus.append(stemmed_tweet)
        corpus.append(lemma_tweet)
    
    print("lower case lemmat", file=f)
    print("Tokenizing done!")
    return corpus

# --------------------------------------------------------------

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

def processing2(df):
    Corpus = df

    # Step - a : Remove blank rows if any.
    Corpus['Tweet text'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    Corpus['Tweet text'] = [entry.lower() for entry in Corpus['Tweet text']]
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus['Tweet text'] = [re.sub('@[a-zA-Z0-9_]+|https?://t.co/[a-zA-Z0-9_]+|[^a-zA-Z]', ' ', entry) for entry in Corpus['Tweet text']]
    # Step - c2 : Remove usernames 
    Corpus['Tweet text']= [word_tokenize(entry) for entry in Corpus['Tweet text']]
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(Corpus['Tweet text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'text_final'] = str(Final_words)
    
    # print(Corpus['text_final'])
    print("lower case lemmat", file=f)
    return Corpus



# --------------------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer

# Create bag-of-words model
def bagOfWords(df, corpus):
    vectorizer = CountVectorizer(max_features = 250) # original = 1500
    X = vectorizer.fit_transform(corpus).toarray()
    y = df.iloc[:,1].values

    print("Bag of words done!")

    print("bag_of_words: 250 max_features", file=f)
    return (X, y)

# --------------------------------------------------------------

from sklearn.model_selection import train_test_split

# Split dataset into training and test sets
def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    print("Splitted data")

    print("test_size: 0.2 | random_state: 0", file=f)
    return (X_train, X_test, y_train, y_test)

# --------------------------------------------------------------

from imblearn.over_sampling import SMOTE

# Oversampling Method
def smote(X_train, y_train):

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print("Oversampling")
    print("Oversampling with Smote", file=f)
    return (X_train, y_train)

# --------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB

# Fit Naive Bayes to the training set
def naiveBayes():
    # classifier = GaussianNB()
    # classifier.fit(X_train, y_train)

    # print("Naive Bayes done!")
    # print("Predicting test set results...")
    # y_pred = classifier.predict(X_test)


    # conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.show()

    # print(confusion_matrix(y_test, y_pred))
    # print('Accuracy: ', accuracy_score(y_test, y_pred))
    # print('Precision: ', precision_score(y_test, y_pred))
    # print('Recall: ', recall_score(y_test, y_pred))
    # print('F1: ', f1_score(y_test, y_pred))
    return


# --------------------------------------------------------------

from sklearn.svm import SVC

# SVM
def svm(X_train, X_test, y_train, y_test):

    # 'C': 1, 'kernel': 'linear'
    classifier = SVC(C=1, kernel='linear')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("SVM 'C': 1, 'kernel': 'linear'", file=f)
    print(confusion_matrix(y_test, y_pred), file=f)
    print('Accuracy: ', accuracy_score(y_test, y_pred), file=f)
    print('Precision: ', precision_score(y_test, y_pred), file=f)
    print('Recall: ', recall_score(y_test, y_pred), file=f)
    print('F1: ', f1_score(y_test, y_pred), file=f)

# --------------------------------------------------------------

from sklearn.linear_model import LogisticRegression

# Logistic Regression
def logisticRegression(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("LogisticRegression", file=f)
    print(confusion_matrix(y_test, y_pred), file=f)
    print('Accuracy: ', accuracy_score(y_test, y_pred), file=f)
    print('Precision: ', precision_score(y_test, y_pred), file=f)
    print('Recall: ', recall_score(y_test, y_pred), file=f)
    print('F1: ', f1_score(y_test, y_pred), file=f)

# --------------------------------------------------------------

from sklearn.linear_model import Perceptron

# Perceptron
def perceptron(X_train, X_test, y_train, y_test):
    classifier = Perceptron()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("Perceptron", file=f)
    print(confusion_matrix(y_test, y_pred), file=f)
    print('Accuracy: ', accuracy_score(y_test, y_pred), file=f)
    print('Precision: ', precision_score(y_test, y_pred), file=f)
    print('Recall: ', recall_score(y_test, y_pred), file=f)
    print('F1: ', f1_score(y_test, y_pred), file=f)

# --------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier

# Decision Tree
def decisionTree(X_train, X_test, y_train, y_test):

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("Decision Tree", file=f)
    print(confusion_matrix(y_test, y_pred), file=f)
    print('Accuracy: ', accuracy_score(y_test, y_pred), file=f)
    print('Precision: ', precision_score(y_test, y_pred), file=f)
    print('Recall: ', recall_score(y_test, y_pred), file=f)
    print('F1: ', f1_score(y_test, y_pred), file=f)

# --------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

# Random Forest
def randomForest(X_train, X_test, y_train, y_test):

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("Random Forest", file=f)
    print(confusion_matrix(y_test, y_pred), file=f)
    print('Accuracy: ', accuracy_score(y_test, y_pred), file=f)
    print('Precision: ', precision_score(y_test, y_pred), file=f)
    print('Recall: ', recall_score(y_test, y_pred), file=f)
    print('F1: ', f1_score(y_test, y_pred), file=f)


# --------------------------------------------------------------

from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(50, 50, 50), learning_rate= 'adaptive', max_iter=200, solver='sgd')

# Multi Layered Perceptron
def mlp(X_train, X_test, y_train, y_test):
    # 'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'sgd'
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("MLP activation='tanh', alpha=0.0001, hidden_layer_sizes=(50, 50, 50), learning_rate= 'adaptive', max_iter=200, solver='sgd'", file=f)
    print(confusion_matrix(y_test, y_pred), file=f)
    print('Accuracy: ', accuracy_score(y_test, y_pred), file=f)
    print('Precision: ', precision_score(y_test, y_pred), file=f)
    print('Recall: ', recall_score(y_test, y_pred), file=f)
    print('F1: ', f1_score(y_test, y_pred), file=f)

# --------------------------------------------------------------

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Finding the best parameters for MLP
def findMlp(X_train, X_test, y_train, y_test):
    print("'hidden_layer_sizes': [(20,20), (50,50), (50,50,50), (10,10)],", file=f)
    print("'activation': ['tanh'],", file=f)
    print("'alpha': [0.05, 0.0001],", file=f)
    print("'learning_rate': ['adaptive'],", file=f)
    print("'max_iter': [200, 500],", file=f)
    
    parameter_space = {
        'hidden_layer_sizes': [(20,20), (50,50), (50,50,50), (10,10)],
        'activation': ['tanh'],
        'solver': ['sgd'],
        'alpha': [0.05, 0.0001],
        'learning_rate': ['adaptive'],
        'max_iter': [200, 500],
    }

    clf = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)

    # Best paramete set
    print('Best parameters found:\n', clf.best_params_, file=f)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=f)

    y_true, y_pred = y_test , clf.predict(X_test)

    print('Results on the test set:', file=f)
    print(classification_report(y_true, y_pred), file=f)

# --------------------------------------------------------------

def findSVC(X_train, X_test, y_train, y_test):

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), tuned_parameters, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)

    print('Best parameters found:\n', clf.best_params_, file=f)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=f)

    y_true, y_pred = y_test , clf.predict(X_test)

    print('Results on the test set:', file=f)
    print(classification_report(y_true, y_pred), file=f)
    

# --------------------------------------------------------------

# # Simple test

# rev = input("Enter tweet: ")
# rev = re.sub('[^a-zA-Z]', ' ', rev).lower().split()
# rev = ' '.join([ps.stem(w) for w in rev])
# X = vectorizer.transform([rev]).toarray()

# # print(X.shape)
# # print(X)

# if(classifier.predict(X) == [1]):
#     print('Irony detected! (+)')
# else:
#     print('Not ironic (-)')


# --------------------------------------------------------------

def main():
    

    (df, test_df) = readData("./datasets/train/train-taskA.txt", "./datasets/test/gold_test_taskA.txt")
    Corpus = processing2(df)
    test_corpus = processing2(test_df)

    Train_X = Corpus['text_final']
    Train_Y = Corpus.iloc[:,1].values

    Test_X = test_corpus['text_final']
    Test_Y = test_corpus.iloc[:,1].values

    # (X_train, y_train) = bagOfWords(df, corpus)
    # (X_test, y_test) = bagOfWords(test_df, test_corpus)

    # Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Label'],test_size=0.2)

    Encoder = LabelEncoder()
    Train_Y  = Encoder.fit_transform(Train_Y )
    Test_Y  = Encoder.fit_transform(Test_Y )

    print("TfidfVectorizer max_features=5000", file=f)
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(Corpus['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    (Train_X_Tfidf, Train_Y) = smote(Train_X_Tfidf, Train_Y)

    # naiveBayes()

    # print("\n-----\n", file=f)

    # svm(Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y)

    # print("\n-----\n", file=f)

    # logisticRegression(X_train, X_test, y_train, y_test)

    print("\n-----\n", file=f)

    perceptron(Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y)

    # print("\n-----\n", file=f)

    # decisionTree(X_train, X_test, y_train, y_test)

    # print("\n-----\n", file=f)

    # randomForest(X_train, X_test, y_train, y_test)

    # print("\n-----\n", file=f)

    # mlp(Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y)

    # print("\n-----\n", file=f)

    # findMlp(Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y)

    # print("\n-----\n", file=f)

    # findSVC(Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y)

    print("Finished")

# Change Filename everytime xd
f = open("./logs/good_log5_perceptron_default.txt", "w")
main()
f.close()
