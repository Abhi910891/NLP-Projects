import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer() 

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords 
    tokens = [word for word in tokens if word not in stop_words] 
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word , pos = 'v') for word in tokens]
     
    # Join tokens back to string
    clean_text = " ".join(tokens) 
    return clean_text