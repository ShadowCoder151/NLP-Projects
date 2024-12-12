from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec

S = set(stopwords.words('english'))
# st = PorterStemmer()
lm = WordNetLemmatizer()

def text_preprocess(text):
	# text = text.lower()
	t = word_tokenize(text)
	t = [word for word in t if word not in S]
	t = [word for word in t if word not in string.punctuation]
	# t = [st.stem(word) for word in t]
	t = [lm.lemmatize(word) for word in t]
	return t



def convert_text_to_word2vec(t, model):
    if not t:
        return np.zeros(model.vector_size)
    
    V = np.array([model.wv[word] for word in t if word in model.wv])
    return np.mean(V, axis=0) if V.size > 0 else np.zeros(model.vector_size)



def data_preprocess(dataset: pd.DataFrame):
    dataset.columns = ['tweet_id', 'entity', 'sentiment', 'content']
    
    dataset['content'] = dataset['content'].fillna("").astype(str)
    dataset['content'] = dataset['content'].apply(lambda x: text_preprocess(x))

    l1 = LabelEncoder()
    dataset['entity'] = l1.fit_transform(dataset['entity'])

    l2 = LabelEncoder()
    dataset['sentiment'] = l2.fit_transform(dataset['sentiment'])
    model = Word2Vec(dataset['content'], vector_size=50, window=2, min_count=1, workers=4)
    
    dataset['content'] = dataset['content'].apply(lambda x: convert_text_to_word2vec(x, model))
    
    return dataset

train = pd.read_csv('C:/Users/Welcome/Desktop/NLP-Projects/Sentiment Analysis/twitter_training.csv')
valid = pd.read_csv('C:/Users/Welcome/Desktop/NLP-Projects/Sentiment Analysis/twitter_validation.csv')


train_mod = data_preprocess(train)
valid_mod = data_preprocess(valid)

dump(train_mod, 'C:/Users/Welcome/Desktop/NLP-Projects/Sentiment Analysis/train_preprocessed.joblib')
dump(valid_mod, 'C:/Users/Welcome/Desktop/NLP-Projects/Sentiment Analysis/valid_preprocessed.joblib')





    
	
    


