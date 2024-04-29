import streamlit as st
import pickle
import re
import nltk
import sklearn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import docx2txt
from nltk.tokenize import WhitespaceTokenizer

nltk.download('punkt')
nltk.download('stopwords')

# load the trained model and  tokenization:
# clf = pickle.load(open('uclf.pkl','rb'))
# word_vectorizer = pickle.load(open('word_vectorizer.pkl','rb'))


def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def predict(cleaned_resume):
    input_features = word_vectorizer.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    return prediction_id

def catego(prediction):
       category = {
            0: "Advocate",
            1: "Arts",
            2: "Automation Testing",
            3: "Blockchain",
            4: "Business Analyst",
            5: "Civil Engineer",
            6: "Data Science",
            7: "Database Developer",
            8: "DevOps Engineer",
            9: "DotNet Developer",
            10: "ETL Developer",
            11: "Electrical Engineer",
            12: "HR	",
            13: "Hadoop	",
            14: "Health and fitness",
            15: "Java Developer",
            16: "Mechanical Engineering	",
            17: "Network Security Engineer	",
            18: "Operations Manager",
            19: "PMO",
            20: "Python Developer",
            21: "SAP Developer",
            22: "Sales Manager",
            23: "Testing",
            24: "Web Designing",
        }
       category_name = category.get(prediction, "Unknown")
       return category_name



#web app
def main():
    st.title("Upload Resume for screen")
    uploaded_file = st.file_uploader("upload resume", type=['pdf'])


    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')


        cleaned_resume = clean_resume(resume_text)
        prediction = predict(cleaned_resume)
        st.write(prediction)

        category_name = catego(prediction)
        st.write("Predicted Category:", category_name)

        st.title(" Upload job Description")
        job_pred = st.text_input("upload your Job description", "")
        joblen = len(job_pred)

        if joblen == 0:
          st.write("Please Upload Description")
        else:
          job_role = clean_resume(job_pred)
          jobprediction = predict(job_role)
          st.write(jobprediction)

          job_category = catego(jobprediction)
          st.write("job Predicted Category:", job_category)

          text_tokenizer= WhitespaceTokenizer()
          remove_characters= str.maketrans("", "", "±§!@#$%^&*()-_=+[]}{;'\:,./<>?|")
          cv = CountVectorizer()

          resume_docx = cleaned_resume
          job_description = job_pred
          #takes the texts in a list
          text_docx= [resume_docx, job_description]
          #creating the list of words from the word document
          words_docx_list = text_tokenizer.tokenize(resume_docx)
          #removing speacial charcters from the tokenized words
          words_docx_list=[s.translate(remove_characters) for s in words_docx_list]
          #giving vectors to the words
          count_docx = cv.fit_transform(text_docx)
          #using the alogorithm, finding the match between the resume/cv and job description
          similarity_score_docx = cosine_similarity(count_docx)
          match_percentage_docx= round((similarity_score_docx[0][1]*100),2)

          st.subheader("Matching percentage is :")
          st.write(match_percentage_docx)




if __name__ == "__main__":
 main()
