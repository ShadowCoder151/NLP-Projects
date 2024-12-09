from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

# def download_resources():
#     nltk.download('wordnet')
#     nltk.download('omw-1.4')

def text_preprocessing(text):
    t1 = text.lower()

    t2 = word_tokenize(text)   # Tokenization

    stop_words = set(stopwords.words('english'))
    t3 = [word for word in t2 if word not in stop_words] # Stop word removal

    st = PorterStemmer()
    t4 = [st.stem(word) for word in t3] # Stemming

    lemma = WordNetLemmatizer()

    t5 = [lemma.lemmatize(word) for word in t4] # Lemmatization

    t6 = [word for word in t5 if word not in string.punctuation] # Remove punctuation marks

    return t6

text = "The diligent student, with remarkable focus and unwavering determination, consistently tackled challenging mathematical problems, demonstrating resilience, intelligence, and a genuine passion for learning, even when faced with seemingly insurmountable obstacles, ultimately achieving success and inspiring others through sheer hard work, perseverance, and an unrelenting belief in self-improvement."
print(text_preprocessing(text))








