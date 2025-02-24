import streamlit as st
import pickle
from docx import Document  
import PyPDF2 
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('clf.pkl', 'rb') as model_file:
    svc_model = pickle.load(model_file)  # Load the trained classifier

with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)  # Load the TF-IDF vectorizer

with open('encoder.pkl', 'rb') as encoder_file:
    le = pickle.load(encoder_file)  # Load the label encoder

# Function to clean and preprocess resume text
def clean_resume_text(text):
    text = re.sub('http\S+\s', ' ', text)  
    text = re.sub('RT|cc', ' ', text) 
    text = re.sub('#\S+\s', ' ', text)  
    text = re.sub('@\S+', ' ', text)  
    text = re.sub('[%s]' % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), ' ', text) 
    text = re.sub(r'[^\x00-\x7f]', ' ', text)  
    text = re.sub('\s+', ' ', text)  
    return text.strip()

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + '\n'
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(file):
    doc = Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from a TXT file with encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')  
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')  
    return text

# Function to handle file upload and text extraction
def process_uploaded_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

# Function to predict the category of a resume
def predict_category(resume_text):
    cleaned_text = clean_resume_text(resume_text)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

# Function to rank resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Predictor", page_icon="📄", layout="wide")
    st.title("AI-Powered Resume Screening System")
    st.markdown("Enter a job description to rank resumes.")
    
    job_description = st.text_area("Enter the job description")
    
    uploaded_files = st.file_uploader("Upload resumes for ranking", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
    if uploaded_files and job_description:
        resumes = [process_uploaded_file(file) for file in uploaded_files]
        scores = rank_resumes(job_description, resumes)
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)
        st.subheader("Ranked Resumes")
        st.write(results)
        
        for file, text in zip(uploaded_files, resumes):
            st.subheader(f"Extracted Text from {file.name}")
            st.text_area("Extracted Resume Text", text, height=300)

if __name__ == "__main__":
    main()


# Run the Streamlit app
if __name__ == "__main__":
    main()
