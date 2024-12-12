import pandas as pd
import numpy as np
import string
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from joblib import dump


S = set(stopwords.words('english'))
lm = WordNetLemmatizer()

def text_preprocess(text):
    text = text.lower()
    t = word_tokenize(text)
    t = [word for word in t if word not in string.punctuation]
    t = [word for word in t if word not in S]
    t = [lm.lemmatize(word) for word in t]
    return t

def convert_text_to_word2vec(tokens, model):
    vectors = np.array([model.wv[word] for word in tokens if word in model.wv])
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(model.vector_size)

def data_preprocess(dataset):
    dataset.columns = ['tweet_id', 'entity', 'sentiment', 'content']

    dataset['content'] = dataset['content'].fillna("").astype(str).apply(text_preprocess)


    le_entity = LabelEncoder()
    le_sentiment = LabelEncoder()
    dataset['entity'] = le_entity.fit_transform(dataset['entity'].astype(str))
    dataset['sentiment'] = le_sentiment.fit_transform(dataset['sentiment'].astype(str))


    model = Word2Vec(dataset['content'], vector_size=50, window=2, min_count=1, workers=4)
    dataset['content'] = dataset['content'].apply(lambda x: convert_text_to_word2vec(x, model))

    return dataset


train = pd.read_csv('twitter_training.csv')
valid = pd.read_csv('twitter_validation.csv')

train_preprocessed = data_preprocess(train)
valid_preprocessed = data_preprocess(valid)

dump(train_preprocessed, 'train_preprocessed.joblib')
dump(valid_preprocessed, 'valid_preprocessed.joblib')
