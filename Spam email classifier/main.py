# ==============================
# Imports
# ==============================
import pandas as pd
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# Download NLTK resources
# ==============================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# ==============================
# Load Dataset
# ==============================
df = pd.read_csv('spam_ham_dataset.csv')

# If dataset has unnamed index column, drop it
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

print(df.head())
print(df.columns)


# ==============================
# Text Preprocessing
# ==============================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords, punctuation and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    
    return " ".join(cleaned_tokens)


# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)


# ==============================
# Feature Extraction
# ==============================
vectorizer = CountVectorizer(max_features=3000)

X = vectorizer.fit_transform(df['clean_text'])
y = df['label']   # 0 = ham, 1 = spam


# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# Train Model
# ==============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# ==============================
# Evaluation
# ==============================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==============================
# Test with Custom Message
# ==============================
def predict_spam(message):
    cleaned_message = preprocess_text(message)
    vector = vectorizer.transform([cleaned_message])
    prediction = model.predict(vector)
    
    return "Spam" if prediction[0] == 1 else "Ham"


# Example
msg = "Congratulations! You have won a free mobile phone"
print("\nMessage:", msg)
print("Prediction:", predict_spam(msg))
