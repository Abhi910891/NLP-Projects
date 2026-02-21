from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def text_classification(tfidf):
    X = tfidf
    y = [1]
    clf = MultinomialNB()
    clf.fit(X, y)
    return clf.predict(X)