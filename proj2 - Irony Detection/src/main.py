##################################################################################
# Coleta de dados
import pandas as pd 


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

df = pd.read_csv("./datasets/train/train-taskA.txt", sep="	")
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
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

corpus = []
ps = PorterStemmer()
for i in range(len(df)):
    # get tweet and remove usernames (@username) and links to pictures (https://t.co/link)
    tweet = re.sub('@[a-zA-Z0-9_]+|https?://t.co/[a-zA-Z0-9_]+|[^a-zA-Z]', ' ', df['Tweet text'][i])
    # to lower-case and tokenize
    tweet = tweet.lower().split()
    # stemming and stop word removal
    tweet = ' '.join([ps.stem(w) for w in tweet if not w in set(stopwords.words('english'))])
    corpus.append(tweet)

# print(corpus)
print("Tokenizing done!")

##################################################################################


# Create bag-of-words model

from sklearn.feature_extraction.text import CountVectorizer

# TODO: change max_features parameter
vectorizer = CountVectorizer(max_features = 1500)
X = vectorizer.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values

# print(vectorizer.get_feature_names())
print(X.shape, y.shape)
print("\n\ny:\n\n", y)


print("Bag of words done!")

# ##################################################################################


# # Split dataset into training and test sets

# #TODO: split with different test file
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# # print(X_train.shape, y_train.shape)
# # print(X_test.shape, y_test.shape)

# print("Splitting done!")

# ##################################################################################


# # Fit Naive Bayes to the training set

# from sklearn.naive_bayes import GaussianNB

# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# print("Naive Bayes done!")
# print("Predicting test set results...")
# y_pred = classifier.predict(X_test)

# print("y_pred: ", y_pred)
# print("y_test: ", y_test)

# print("Test set results predicting done!")
# print("Generating metrics...")

# print(confusion_matrix(y_test, y_pred))
# print('Accuracy: ', accuracy_score(y_test, y_pred))
# print('Precision: ', precision_score(y_test, y_pred))
# print('Recall: ', recall_score(y_test, y_pred))
# print('F1: ', f1_score(y_test, y_pred))

# print("Metrics generated...")


# #################################################################################


# SVM

# from sklearn.svm import SVC

# classifier = SVC()
# classifier.fit(X_train, y_train)
# print("SVC done!")

# y_pred = classifier.predict(X_test)
# print("Test set results predicting done!")


# print(confusion_matrix(y_test, y_pred))
# print('Accuracy: ', accuracy_score(y_test, y_pred))
# print('Precision: ', precision_score(y_test, y_pred, average=None))
# print('Recall: ', recall_score(y_test, y_pred, average=None))
# print('F1: ', f1_score(y_test, y_pred, average=None))


# # ##################################################################################


# # # Logistic Regression

# # from sklearn.linear_model import LogisticRegression

# # classifier = LogisticRegression()
# # classifier.fit(X_train, y_train)
# # y_pred = classifier.predict(X_test)

# # print(confusion_matrix(y_test, y_pred))
# # print('Accuracy: ', accuracy_score(y_test, y_pred))
# # print('Precision: ', precision_score(y_test, y_pred))
# # print('Recall: ', recall_score(y_test, y_pred))
# # print('F1: ', f1_score(y_test, y_pred))


# # ##################################################################################


# # # Perceptron

# # from sklearn.linear_model import Perceptron

# # classifier = Perceptron()
# # classifier.fit(X_train, y_train)
# # y_pred = classifier.predict(X_test)

# # print(confusion_matrix(y_test, y_pred))
# # print('Accuracy: ', accuracy_score(y_test, y_pred))
# # print('Precision: ', precision_score(y_test, y_pred))
# # print('Recall: ', recall_score(y_test, y_pred))
# # print('F1: ', f1_score(y_test, y_pred))


# # ##################################################################################


# # # Decision Tree

# # from sklearn.tree import DecisionTreeClassifier

# # classifier = DecisionTreeClassifier()
# # classifier.fit(X_train, y_train)
# # y_pred = classifier.predict(X_test)

# # print(confusion_matrix(y_test, y_pred))
# # print('Accuracy: ', accuracy_score(y_test, y_pred))
# # print('Precision: ', precision_score(y_test, y_pred))
# # print('Recall: ', recall_score(y_test, y_pred))
# # print('F1: ', f1_score(y_test, y_pred))


# # ##################################################################################


# # # Random Forest

# # from sklearn.ensemble import RandomForestClassifier

# # classifier = RandomForestClassifier()
# # classifier.fit(X_train, y_train)
# # y_pred = classifier.predict(X_test)

# # print(confusion_matrix(y_test, y_pred))
# # print('Accuracy: ', accuracy_score(y_test, y_pred))
# # print('Precision: ', precision_score(y_test, y_pred))
# # print('Recall: ', recall_score(y_test, y_pred))
# # print('F1: ', f1_score(y_test, y_pred))


# ##################################################################################


# # Simple test

# rev = input("Enter tweet: ")
# rev = re.sub('[^a-zA-Z]', ' ', rev).lower().split()
# rev = ' '.join([ps.stem(w) for w in rev])
# X = vectorizer.transform([rev]).toarray()

# print(X.shape)
# print(X)

# if(classifier.predict(X) == [1]):
#     print('Irony detected! (+)')
# else:
#     print('Not ironic (-)')

