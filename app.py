import streamlit as st
import spacy
import pdfminer
from pdfminer.high_level import extract_text
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    return extract_text(pdf_file)

def analyze_resume(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def calculate_similarity(resume_text, jd_text):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0] * 100

st.title("AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste the Job Description here")

if uploaded_file and job_desc:
    resume_text = extract_text_from_pdf(uploaded_file)
    cleaned_resume = analyze_resume(resume_text)
    score = calculate_similarity(cleaned_resume, job_desc)
    st.subheader("Resume Match Score:")
    st.write(f"{score:.2f}%")
if st.button("Analyze"):
    if resume and job_desc:
        score = get_similarity(resume, job_desc)
        st.success(f"✅ Match Score: {score}%")
    else:
        st.warning("Please enter both resume and job description.")
