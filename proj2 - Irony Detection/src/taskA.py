##################################################################################
# Coleta de dados
import pandas as pd 
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from imblearn.over_sampling import SMOTE


df = pd.read_csv("./datasets/train/train-taskA.txt", sep="	")

test_df = pd.read_csv("./datasets/test/gold_test_TaskA.txt", sep="	")

# print(df)
print("Reading data done!")

# tá lendo errado o dataframe: ignorando 17 índices
# indices nao lidos (inclusive): 1646-1648, 3029-3039, 3459-3461
# for i in range(1, len(df)):
# 	if(df['Tweet index'][i] != (df['Tweet index'][i-1] + 1)):
# 		print("Current", df['Tweet index'][i])
# 		print("Previous", df['Tweet index'][i-1])

##################################################################################
# Cleaning and tokenization of tweets
# TODO: tentar tokenização diferente pra levar em consideração as hashtags, links de fotos, arrobas, etc
import re
import nltk
# from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

corpus = []
lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
# stemmer = PorterStemmer()
for i in range(len(df)):
    # get tweet and remove usernames (@username) and links to pictures (https://t.co/link)
    tweet = re.sub('@[a-zA-Z0-9_]+|https?://t.co/[a-zA-Z0-9_]+|[^a-zA-Z]', ' ', df['Tweet text'][i])
    # to lower-case and tokenize
    tweet = tweet.lower().split()

    # # stemming and stop word removal
    stemmed_tweet = ' '.join([stemmer.stem(w) for w in tweet if not w in set(stopwords.words('english'))])

    #lemmatizing
    lemma_tweet = ' '.join([lemmatizer.lemmatize(w) for w in tweet if not w in set(stopwords.words('english'))])

    corpus.append(stemmed_tweet)

test_corpus = []
for i in range(len(test_df)):
    # get tweet and remove usernames (@username) and links to pictures (https://t.co/link)
    tweet = re.sub('@[a-zA-Z0-9_]+|https?://t.co/[a-zA-Z0-9_]+|[^a-zA-Z]', ' ', test_df['Tweet text'][i])
    # to lower-case and tokenize
    tweet = tweet.lower().split()

    # # stemming and stop word removal
    stemmed_tweet = ' '.join([stemmer.stem(w) for w in tweet if not w in set(stopwords.words('english'))])

    #lemmatizing
    lemma_tweet = ' '.join([lemmatizer.lemmatize(w) for w in tweet if not w in set(stopwords.words('english'))])

    test_corpus.append(stemmed_tweet)

# print(test_corpus)
print("Tokenizing done!")

##################################################################################


# Create bag-of-words model

from sklearn.feature_extraction.text import CountVectorizer

# TODO: change max_features parameter
vectorizer = CountVectorizer(max_features = 1500) # original = 1500
X = vectorizer.fit_transform(corpus).toarray()
y = df.iloc[:,1].values

X_test = vectorizer.fit_transform(test_corpus).toarray()
y_test = test_df.iloc[:,1].values


# print(vectorizer.get_feature_names())
# print(X.shape, y.shape)


print("Bag of words done!")

# ##################################################################################


# Split dataset into training and test sets

#TODO: split with different test file
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_train = X
y_train = y


# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

print("Splitting done!")

###################################################################################

# SMOTE

sm = SMOTE()

X_train, y_train = sm.fit_resample(X_train, y_train)

#################################################################################


# Fit Naive Bayes to the training set

# from sklearn.naive_bayes import GaussianNB

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


################################################################################


# SVM

# from sklearn.svm import SVC

# classifier = SVC()
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print('Accuracy: ', accuracy_score(y_test, y_pred))
# print('Precision: ', precision_score(y_test, y_pred))
# print('Recall: ', recall_score(y_test, y_pred))
# print('F1: ', f1_score(y_test, y_pred))

# # ##################################################################################


# Logistic Regression

# from sklearn.linear_model import LogisticRegression

# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print('Accuracy: ', accuracy_score(y_test, y_pred))
# print('Precision: ', precision_score(y_test, y_pred))
# print('Recall: ', recall_score(y_test, y_pred))
# print('F1: ', f1_score(y_test, y_pred))


# # ##################################################################################


# Perceptron

# from sklearn.linear_model import Perceptron

# classifier = Perceptron()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print('Accuracy: ', accuracy_score(y_test, y_pred))
# print('Precision: ', precision_score(y_test, y_pred))
# print('Recall: ', recall_score(y_test, y_pred))
# print('F1: ', f1_score(y_test, y_pred))


# # ##################################################################################


# Decision Tree

# from sklearn.tree import DecisionTreeClassifier

# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print('Accuracy: ', accuracy_score(y_test, y_pred))
# print('Precision: ', precision_score(y_test, y_pred))
# print('Recall: ', recall_score(y_test, y_pred))
# print('F1: ', f1_score(y_test, y_pred))


##################################################################################


# Random Forest

# from sklearn.ensemble import RandomForestClassifier

# classifier = RandomForestClassifier()
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print('Accuracy: ', accuracy_score(y_test, y_pred))
# print('Precision: ', precision_score(y_test, y_pred))
# print('Recall: ', recall_score(y_test, y_pred))
# print('F1: ', f1_score(y_test, y_pred))


 ##################################################################################


# Multi Layered Perceptron

from sklearn.neural_network import MLPClassifier
# 'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'adaptive', 'solver': 'sgd'
classifier = MLPClassifier(max_iter=2000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F1: ', f1_score(y_test, y_pred))


# ##################################################################################


# Finding the best parameters for MLP

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (20,20,20,20), (50,50,50,50), (50,100,50), (100,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05, 0.01],
    'learning_rate': ['constant','adaptive', 'invscaling'],
}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)

# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred = y_test , clf.predict(X_test)

from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))

# # ##################################################################################

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
