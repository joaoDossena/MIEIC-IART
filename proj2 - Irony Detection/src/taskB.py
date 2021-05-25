import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
# %matplotlib inline 
sns.set(color_codes=True)

# Get data from file
def readData(path_train, path_test):
    df = pd.read_csv(path_train, sep="\t")
    test_df = pd.read_csv(path_test, sep="\t")
    print("Data Read Successfully")
    return (df, test_df)


(train_df, test_df) = readData("./datasets/train/train-taskB.txt", "./datasets/test/gold_test_taskB.txt")

# train_df.shape
# test_df.shape
# train_df.head()
# # checking if there are any null values present in the dataset
# train_df.isnull().sum()

# train_df.groupby('Label')['Tweet text'].count()
# train_df.groupby('Label')['Tweet text'].count().plot.bar(ylim=0)
# plt.show()



import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
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


def processing(Corpus):
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
    print("Processed")
    return Corpus


corpus = processing(train_df)
test_corpus = processing(test_df)




from sklearn.feature_extraction.text import CountVectorizer

# Create bag-of-words model
def bagOfWords(df, corpus):
    vectorizer = CountVectorizer(max_features = 250) # original = 1500
    X = vectorizer.fit_transform(corpus).toarray()
    y = df.iloc[:,1].values

    print("Bag of words done!")

    print("bag_of_words: 250 max_features")
    return (X, y)



from imblearn.over_sampling import SMOTE

# Oversampling Method
def smote(X_train, y_train):

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print("Oversampling with Smote")
    return (X_train, y_train)


Train_X = corpus['text_final']
Train_Y = corpus.iloc[:,1].values

Test_X = test_corpus['text_final']
Test_Y = test_corpus.iloc[:,1].values


print("TfidfVectorizer max_features=5000")
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# Oversampling
# (Train_X_Tfidf, Train_Y) = smote(Train_X_Tfidf, Train_Y)
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from matplotlib import pyplot
# Oversampling - Random Over Sampler
oversample = RandomOverSampler(sampling_strategy='all')
(Train_X_Tfidf_RandOS, Train_Y_RandOS) = oversample.fit_resample(Train_X_Tfidf, Train_Y)

# Oversampling - SMOTE
# (Train_X_Tfidf_SMOTE, Train_Y_SMOTE) = smote(Train_X_Tfidf, Train_Y)

# # Undersampling - Random Under Sampler
# from imblearn.under_sampling import RandomUnderSampler
# undersample = RandomUnderSampler()
# (Train_X_Tfidf_RandUS, Train_Y_RandUS) = undersample.fit_resample(Train_X_Tfidf, Train_Y)

# counter = Counter(Train_Y_RandOS)
# for k,v in counter.items():
# 	per = v / len(Train_Y) * 100
# 	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# pyplot.title("Random Over Sampler")
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()

# counter = Counter(Train_Y_RandUS)
# for k,v in counter.items():
# 	per = v / len(Train_Y) * 100
# 	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# pyplot.title("Random Under Sampler")
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()

# counter = Counter(Train_Y_SMOTE)
# for k,v in counter.items():
# 	per = v / len(Train_Y) * 100
# 	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# pyplot.title("SMOTE")
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import time

from sklearn.svm import SVC

# SVM
def svm(X_train, X_test, y_train, y_test):
    classifier = SVC()

    time_0 = time.time()
    classifier.fit(X_train, y_train)
    training_time = str(round(time.time()-time_0,3))
    
    time_1 = time.time()
    y_pred = classifier.predict(X_test)
    predict_time = str(round(time.time()-time_1,3))
    
    print("Training Time: " + training_time)
    print("Predict Time: " + predict_time)
    
    print(classification_report(y_test, y_pred))
    
    conf_matrix = confusion_matrix(y_true=Test_Y, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

    return (classification_report(y_test, y_pred, output_dict=True), training_time, predict_time, y_test, y_pred)

from sklearn.neural_network import MLPClassifier
# Finding the best parameters for MLP
def findMlp(X_train, X_test, y_train, y_test):
    # print("'hidden_layer_sizes': [(20,20), (50,50), (50,50,50), (10,10)],", file=f)
    # print("'activation': ['tanh'],", file=f)
    # print("'alpha': [0.05, 0.0001],", file=f)
    # print("'learning_rate': ['adaptive'],", file=f)
    # print("'max_iter': [200, 500],")
    
    parameter_space = {
        'hidden_layer_sizes': [(15,), (10,), (7,), (20,)],
        'activation': ['tanh'],
        'solver': ['sgd'],
        'alpha': [0.05],
        'learning_rate': ['adaptive'],
    }

    clf = GridSearchCV(MLPClassifier(), parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)

    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    y_true, y_pred = y_test , clf.predict(X_test)

    print('Results on the test set:')
    print(classification_report(y_true, y_pred))


# # normal (mau)
# (svc_for_graph, svc_train_time, svc_predict_time, svc_y_test, svc_y_pred) = svm(Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y)

# # Random Oversampling
# (svc_for_graph, svc_train_time, svc_predict_time, svc_y_test, svc_y_pred) = svm(Train_X_Tfidf_RandOS, Test_X_Tfidf, Train_Y_RandOS, Test_Y)

# # Random Undersampling
# (svc_for_graph, svc_train_time, svc_predict_time, svc_y_test, svc_y_pred) = svm(Train_X_Tfidf_RandUS, Test_X_Tfidf, Train_Y_RandUS, Test_Y)

# # Smote (idk)
# (svc_for_graph, svc_train_time, svc_predict_time, svc_y_test, svc_y_pred) = svm(Train_X_Tfidf_SMOTE, Test_X_Tfidf, Train_Y_SMOTE, Test_Y)


findMlp(Train_X_Tfidf_RandOS, Test_X_Tfidf, Train_Y_RandOS, Test_Y)