import streamlit as st
import pickle
from docx import Document  
import PyPDF2 
import re

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
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text) 
    text = re.sub(r'[^\x00-\x7f]', ' ', text)  
    text = re.sub('\s+', ' ', text)  
    return text.strip()


# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
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


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Predictor", page_icon="📄", layout="wide")

    st.title("AI-Powered Resume Screening System")
    st.markdown("Upload a resume in PDF, DOCX, or TXT format to predict its job category.")

    # File upload section
    uploaded_file = st.file_uploader("Choose a Resume File", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            
            resume_text = process_uploaded_file(uploaded_file)
            st.success("Text extracted successfully from the uploaded resume.")

            if st.checkbox("Show Extracted Text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Prediction Result")
            category = predict_category(resume_text)
            st.write(f"The predicted job category for the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")


# Run the Streamlit app
if __name__ == "__main__":
    main()