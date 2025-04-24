import streamlit as st
import spacy
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    if pdf_file is not None:
        return extract_text(pdf_file)
    return ""

# Clean resume text using NLP
def analyze_resume(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Calculate similarity score
def calculate_similarity(resume_text, jd_text):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0] * 100

# Streamlit UI
st.title("AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste the Job Description here")

# Button to trigger analysis
if st.button("Analyze"):
    if uploaded_file is not None and job_desc.strip():
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            cleaned_resume = analyze_resume(resume_text)
            score = calculate_similarity(cleaned_resume, job_desc)
        st.success(f"✅ Resume Match Score: {score:.2f}%")
    else:
        st.warning("⚠️ Please upload a resume and enter a job description.")
