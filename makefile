from dotenv import load_dotenv

load_dotenv()
import base64
import streamlit as st
import os
import io
from PIL import Image 
import pdf2image
import google.generativeai as genai
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd  # Assuming interview questions are stored in a CSV file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# Load dataset containing sample interview questions
interview_questions_df = pd.read_csv("interview_questions.csv") 

# Train an NLP model on the interview questions dataset
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(interview_questions_df['Question'])
y = interview_questions_df['Category']  # Assuming there's a category column for different types of questions
clf = LinearSVC()
clf.fit(X, y)

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get Gemini response using NLP model
def get_gemini_response_nlp(input_text, resume_text):
    # Tokenize input text and resume text
    input_tokens = nltk.word_tokenize(input_text.lower())
    resume_tokens = nltk.word_tokenize(resume_text.lower())

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform text data
    tfidf_matrix = tfidf_vectorizer.fit_transform([input_text, resume_text])

    # Compute cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

    # If the similarity between input text and resume text is high, provide positive response
    if similarity[0][1] > 0.5:
        return "The candidate's profile aligns well with the job description."
    else:
        return "The candidate's profile does not seem to align well with the job description."

# # Function to get sample interview questions
# def get_sample_interview_questions():
#     sample_questions = [
#         "Tell me about yourself.",
#         "What interests you about this role?",
#         "Can you describe a challenging project you worked on and how you overcame obstacles?",
#         "How do you handle conflicts or disagreements with team members?",
#         "Where do you see yourself in five years?",
#         "Do you have any questions for us?"
#     ]
#     return "\n".join(sample_questions)


def similarity(text1, text2):
    # Vectorize input texts
    text1_vector = vectorizer.transform([text1])
    text2_vector = vectorizer.transform([text2])
    
    # Compute cosine similarity between the vectors
    similarity_score = cosine_similarity(text1_vector, text2_vector)[0][0]
    return similarity_score

def get_sample_interview_questions(job_description):
    # Calculate similarity scores between the job description and each question
    similarity_scores = interview_questions_df['Question'].apply(lambda question: similarity(job_description, question))
    
    # Sort questions based on similarity scores
    relevant_questions_indices = similarity_scores.argsort()[-5:][::-1]  # Select top 5 most similar questions
    relevant_questions = interview_questions_df.iloc[relevant_questions_indices]['Question']
    
    return relevant_questions.tolist()

# Function to setup PDF content
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Convert PDF to images
        images = pdf2image.convert_from_bytes(uploaded_file.read())

        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
input_text = st.text_area("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage Match")
submit4 = st.button("Sample Interview Questions")

input_prompt1 = """
You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
Your task is to evaluate the resume against the provided job description and give me the percentage match if the resume matches
the job description. First, the output should come as a percentage, then keywords missing, and lastly final thoughts.
"""

# Handle button actions
if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.write("Please upload the resume")

elif submit3:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.write("Please upload the resume")

elif submit4:
    if uploaded_file is not None:
        resume_text = ""  # Extract text from the resume and store it here
        sample_questions = get_sample_interview_questions()
        st.subheader("Sample Interview Questions:")
        st.write(sample_questions)
    else:
        st.write("Please upload the resume")
