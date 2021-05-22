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

df = pd.read_csv("./datasets/train/train-taskB.txt", sep="	")
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

# print(corpus)
print("Tokenizing done!")

##################################################################################

# Create bag-of-words model

from sklearn.feature_extraction.text import CountVectorizer

# TODO: change max_features parameter
vectorizer = CountVectorizer(max_features = 1500) # original = 1500
X = vectorizer.fit_transform(corpus).toarray()
y = df.iloc[:,1].values

# print(vectorizer.get_feature_names())
print(X.shape, y.shape)

print("Bag of words done!")
