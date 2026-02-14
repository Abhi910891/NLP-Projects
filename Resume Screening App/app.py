import pickle
import re 
import nltk
import streamlit as st 
import numpy

nltk.download("punkt")
nltk.download("stopwords")

# loading model 
clf = pickle.load(open("clf.pkl","rb"))  # rb= read binary mode
tfidf = pickle.load(open("tfidf.pkl","rb"))
lb = pickle.load(open("lb.pkl","rb"))


def cleanResume(txt):
    cleanTxt = re.sub(r"http\S+\s*", "", txt)            # Remove URLs
    cleanTxt = re.sub(r"@\S+", "", cleanTxt)             # Remove @mentions
    cleanTxt = re.sub(r"#\S+", "", cleanTxt)             # Remove hashtags
    cleanTxt = re.sub(r"RT|cc", "", cleanTxt)            # Remove RT and cc
    cleanTxt = re.sub(r"[{}]".format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), " ", cleanTxt)  # Remove punctuations
    cleanTxt = re.sub(r"[^\x00-\x7f]","",cleanTxt)
    cleanTxt = re.sub(r"\s+", " ", cleanTxt)             # Replace multiple spaces with one
    return cleanTxt


# web app 

def main():
    st.title("Resume Screening App")
    upload_file =  st.file_uploader("Upload Resume" , type = ["pdf" , "txt"])
    

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # if UTF-8 decoding fails , try decoding with 'latin-1'
            resume_text = resume_bytes.decode("latin-1")
        
        cleaned_resume = cleanResume(resume_text)
        cleaned_resume = tfidf.transform([cleaned_resume])
        vectorized_text = cleaned_resume.toarray()
        predict_category = clf.predict(vectorized_text)

        predicted_category_name = lb.inverse_transform(predict_category)

        st.write(predicted_category_name[0])
        st.write(predict_category[0])

    
if __name__ == "__main__":
    main()