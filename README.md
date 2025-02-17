# AI-Powered-Resume-Screening-System-
# Overview
The AI-Powered Resume Screening System is a web application designed to automate the process of categorizing resumes based on their content. Built using Python and Streamlit, this system leverages machine learning to predict the job category of a resume by analyzing its text. It supports resumes in PDF, DOCX, and TXT formats, making it versatile for various use cases.

This project is ideal for recruiters, HR professionals, and organizations looking to streamline their resume screening process and improve efficiency.

# Features
Resume Text Extraction: Extracts text from PDF, DOCX, and TXT files.

Text Cleaning: Preprocesses and cleans resume text for better analysis.

Job Category Prediction: Uses a pre-trained machine learning model to predict the job category of a resume.

User-Friendly Interface: Built with Streamlit for an intuitive and interactive web interface.

Flexible File Support: Handles multiple file formats for convenience.

# How It Works
Upload a Resume: Users can upload a resume in PDF, DOCX, or TXT format.

Text Extraction: The system extracts text from the uploaded file.

Text Cleaning: The extracted text is cleaned and preprocessed to remove unnecessary elements (e.g., URLs, special characters).

Prediction: The cleaned text is passed through a pre-trained machine learning model to predict the job category.

Result Display: The predicted job category is displayed to the user.

# Technologies Used
Python: Core programming language.

Streamlit: For building the web application interface.

Scikit-learn: For machine learning model training and prediction.

PyPDF2: For extracting text from PDF files.

python-docx: For extracting text from DOCX files.

Pickle: For loading pre-trained models and vectorizers.

Regex (re): For text cleaning and preprocessing.

# Usage
Open the app in your browser after running the Streamlit command.
Upload a resume file (PDF, DOCX, or TXT).
View the extracted text (optional).
See the predicted job category for the uploaded resume.

# Pre-Trained Models
The system uses the following pre-trained models and files:
clf.pkl: Trained classifier for job category prediction. (Skipped this file in the repository due to size issue)
tfidf.pkl: TF-IDF vectorizer for text transformation.
encoder.pkl: Label encoder for decoding predicted categories.

Ensure these files are present in the project directory for the system to function correctly.
Enjoy using the AI-Powered Resume Screening System! 🚀
