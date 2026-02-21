import streamlit as st

from modules.preprocessing import preprocess
from modules.lingusitic import linguistic_analysis
from modules.statistical import statistical_features
from modules.embedding import word_embeddings
from modules.ml_tasks import sentiment_analysis, text_classification
from modules.transformers_tasks import transformer_tasks
from modules.visualization import word_freq_plot

st.title("🧠 Universal NLP Analyzer")

text = st.text_area("Enter text")

if st.button("Analyze") and text:

    tokens, filtered, stems, lemmas = preprocess(text)
    pos, ner, dep = linguistic_analysis(text)
    bow, tfidf, vocab = statistical_features(text)
    vectors = word_embeddings(tokens)
    sentiment = sentiment_analysis(text)
    cls = text_classification(tfidf)
    summary, answer, generated = transformer_tasks(text)

    st.subheader("Tokens")
    st.write(tokens)

    st.subheader("POS Tags")
    st.write(pos)

    st.subheader("Named Entities")
    st.write(ner)

    st.subheader("Sentiment")
    st.write(sentiment)

    st.subheader("Summary")
    st.write(summary)

    st.subheader("QA Answer")
    st.write(answer)

    st.subheader("Generated Text")
    st.write(generated)

    st.subheader("Word Frequency")
    fig = word_freq_plot(tokens)
    st.pyplot(fig)