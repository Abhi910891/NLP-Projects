from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def statistical_features(text):
    docs = [text]

    bow_vec = CountVectorizer()
    bow = bow_vec.fit_transform(docs).toarray()

    tfidf_vec = TfidfVectorizer()
    tfidf = tfidf_vec.fit_transform(docs).toarray()

    return bow, tfidf, bow_vec.get_feature_names_out()