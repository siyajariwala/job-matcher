import streamlit as st
from sentence_transformers import SentenceTransformer
import math
import pdfplumber

st.title("📄 Resume ↔ Job Match Score")
st.write("Upload your resume and paste a job description to see how well they match semantically.")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(x**2 for x in b))
    return dot_product / (magnitude_a * magnitude_b)

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# UI Elements
resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
job_description = st.text_area("Paste the job description here", height=250)

if st.button("Calculate Match Score"):
    if resume_file and job_description:
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(resume_file)

            job_embedding = model.encode(job_description)
            resume_embedding = model.encode(resume_text)

            score = cosine_similarity(job_embedding, resume_embedding)
            percentage = score * 100

        st.metric("Match Score", f"{percentage:.1f}%")

        if score > 0.7:
            st.success("Strong match! 🎯")
        elif score > 0.5:
            st.warning("Moderate match — consider tailoring your resume")
        else:
            st.error("Weak match — resume may need significant tailoring")
    else:
        st.warning("Please upload a resume and paste a job description first!")