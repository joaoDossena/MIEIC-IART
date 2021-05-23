from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt


import pandas as pd

# Get data from file
def readData(path):
    df = pd.read_csv(path, sep="	")
    print("Data Read")
    return df

# --------------------------------------------------------------

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

corpus = []
lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
# stemmer = PorterStemmer()

# Cleaning and tokenization of tweets
def processing(df):
    for i in range(len(df)):
        # get tweet and remove usernames (@username) and links to pictures (https://t.co/link)
        tweet = re.sub('@[a-zA-Z0-9_]+|https?://t.co/[a-zA-Z0-9_]+|[^a-zA-Z]', ' ', df['Tweet text'][i])

        # to lower-case and tokenize
        tweet = tweet.lower().split()

        # Stemming and stop word removal
        stemmed_tweet = ' '.join([stemmer.stem(w) for w in tweet if not w in set(stopwords.words('english'))])

        # Lemmatizing
        lemma_tweet = ' '.join([lemmatizer.lemmatize(w) for w in tweet if not w in set(stopwords.words('english'))])

        corpus.append(stemmed_tweet)
        # corpus.append(lemma_tweet)
    
    print("lower case stemming", file=f)
    print("Tokenizing done!")

# --------------------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer

# Create bag-of-words model
def bagOfWords(df):
    vectorizer = CountVectorizer(max_features = 1500) # original = 1500
    X = vectorizer.fit_transform(corpus).toarray()
    y = df.iloc[:,1].values

    print("Bag of words done!")

    print("bag_of_words: 1500 max_features", file=f)
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
    classifier = SVC()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("SVM", file=f)
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

classifier = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 50, 50, 50), learning_rate='adaptive', solver='sgd', max_iter=1000)

# Multi Layered Perceptron
def mlp(X_train, X_test, y_train, y_test):
    # 'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'sgd'
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("MLP activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 50, 50, 50), learning_rate='adaptive', solver='sgd', max_iter=1000", file=f)
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
    parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (20,20,20,20), (50,50,50,50), (50,100,50), (100,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05, 0.01],
        'learning_rate': ['constant','adaptive', 'invscaling'],
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
    
    

    df = readData("./datasets/train/train-taskA.txt")
    processing(df)

    (X, y) = bagOfWords(df)

    (X_train, X_test, y_train, y_test) = splitData(X, y)

    (X_train, y_train) = smote(X_train, y_train)

    # naiveBayes()

    print("\n-----\n", file=f)

    svm(X_train, X_test, y_train, y_test)

    print("\n-----\n", file=f)

    logisticRegression(X_train, X_test, y_train, y_test)

    print("\n-----\n", file=f)

    perceptron(X_train, X_test, y_train, y_test)

    print("\n-----\n", file=f)

    decisionTree(X_train, X_test, y_train, y_test)

    print("\n-----\n", file=f)

    randomForest(X_train, X_test, y_train, y_test)

    print("\n-----\n", file=f)

    mlp(X_train, X_test, y_train, y_test)

    print("\n-----\n", file=f)

    # findMlp(X_train, X_test, y_train, y_test)

    print("Finished")

# Change Filename everytime xd
f = open("./logs/log1.txt", "w")
main()
f.close()
