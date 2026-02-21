from gensim.models import Word2Vec

def word_embeddings(tokens):
    sentences = [tokens]
    model = Word2Vec(sentences, vector_size=50, window=3, min_count=1)
    vectors = {w: model.wv[w].tolist() for w in tokens}
    return vectors