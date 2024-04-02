import streamlit as st
import psycopg2
from psycopg2 import Error
import bcrypt
import os
# from dotenv import load_dotenv
from PIL import Image, ImageOps
import base64
import io
import pdf2image
import nltk
import docx
import requests
from openai import OpenAI
import PyPDF2
from io import BytesIO

import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline

from sklearn.decomposition import TruncatedSVD



# Load NLTK resources
nltk.download('punkt')


# Function to connect to PostgreSQL database
def connect_to_db():
    try:
        connection = psycopg2.connect(
            user="postgres",
            password="Saranya@426",
            host="localhost",
            port=5432,
            database="chatbot"
        )
        print("connected to database")
        return connection
    except Error as e:
        # st.error("Error while connecting to PostgreSQL database")
        print(f"Error while connecting to PostgreSQL database: {e}")
        return None 

# Function to create user table if it doesn't exist
def create_user_table():
    try:
        connection = connect_to_db()
        cursor = connection.cursor()
        create_table_query = '''CREATE TABLE IF NOT EXISTS users
                                (first_name VARCHAR(50) NOT NULL,
                                last_name VARCHAR(50) NOT NULL,
                                username VARCHAR(50) UNIQUE NOT NULL,
                                password VARCHAR(200) NOT NULL);'''
        cursor.execute(create_table_query)
        connection.commit()
        st.success("User table created successfully")
    except Error as e:
        st.error(f"Error while creating user table: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to register a new user
def register_user(first_name, last_name, username, password):
    try:
        # Validate input lengths
        if len(first_name) > 50 or len(last_name) > 50 or len(username) > 50 or len(password) > 50:
            st.error("One or more input values exceed the maximum length of 50 characters.")
            return
        
        connection = connect_to_db()
        cursor = connection.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        insert_query = '''INSERT INTO users (first_name, last_name, username, password)
                          VALUES (%s, %s, %s, %s);'''
        cursor.execute(insert_query, (first_name, last_name, username, hashed_password))
        connection.commit()
        st.success("User registered successfully")
        print("user register sucessfully")
    except Error as e:
        print("firstname,lastname,username,password",first_name, last_name, username, hashed_password)
        st.error(f"Error while registering user: {e}")
        print("Error registering  user")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to login user
def login_user(username, password):
    try:
        connection = connect_to_db()
        cursor = connection.cursor()
        select_query = '''SELECT password FROM users WHERE username = %s;'''
        cursor.execute(select_query, (username,))
        hashed_password = cursor.fetchone()[0]
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            st.success("Login successful")
            return True
        else:
            st.error("Invalid username or password")
            return False
    except TypeError:
        st.error("Invalid username or password")
        return False
    except Error as e:
        st.error(f"Error while logging in: {e}")
        return False
    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to logout user
def logout_user():
    st.session_state.pop("username", None)
    st.success("Logout successful")

# Initialize Streamlit app
st.set_page_config(page_title="Career CraftsMan-Chatbot")
st.header("Career CraftsMan-Chatbot")

# Display login page
login_page = st.sidebar.checkbox("Login")
if login_page:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state["username"] = username  # Store username in session state

# Display registration page
registration_page = st.sidebar.checkbox("Register")
if registration_page:
    st.subheader("Register")
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    if st.button("Register"):
        register_user(first_name, last_name, new_username, new_password)

# Display logout button
if "username" in st.session_state:
    logout_button = st.sidebar.checkbox("Logout")
    if logout_button:
        logout_user()

# # # Check if the user is logged in
# if "username" not in st.session_state:
#     # Display the login form
#     st.subheader("Login")
#     login_username = st.text_input("Username")
#     login_password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         if login_user(login_username, login_password):
#             st.session_state["username"] = login_username  # Store username in session state
#             st.success("Login successful")
#             st.button("Logout")  # Display logout button after successful login
#         else:
#             # If login fails, display registration form
            
#             st.subheader("Register")
#             first_name = st.text_input("First Name")
#             last_name = st.text_input("Last Name")
#             register_username = st.text_input("New Username")
#             register_password = st.text_input("New Password", type="password")
#             if st.button("Register"):
#                 register_user(first_name, last_name, register_username, register_password)
#     else:
#         # If login button not clicked, don't display registration form
#         st.subheader("Don't have an account?")
#         st.write("Please register to access the chatbot.")
#         st.button("Register")
# else:
#     # Display the logout button
#     st.subheader(f"Welcome, {st.session_state['username']}!")  # Display welcome message
#     if st.button("Logout"):
#         logout_user()


# Function to extract text from Word resume
def extract_resume_text_word(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
            
        return text
    except Exception as e:
        st.error(f"Error extracting text from Word resume: {e}")
        return None



def extract_resume_text_pdf(uploaded_file):
    try:
        # Open the PDF file
        with open(uploaded_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            
            # Initialize an empty string to store the text
            text = ""
            
            # Loop through each page in the PDF
            for page_num in range(pdf_reader.numPages):
                # Extract text from the current page
                page = pdf_reader.getPage(page_num)
                text += page.extractText() + "\n"
            
            return text
    except Exception as e:
        st.error(f"Error extracting text from PDF resume: {e}")
        return None
    


def extract_resume_text_pdf(uploaded_file):
    try:
        # Create a BytesIO object to read the uploaded file content
        pdf_file = BytesIO(uploaded_file.read())
        
        # Open the PDF file
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Initialize an empty string to store the text
        text = ""
        
        # Loop through each page in the PDF
        for page_num in range(len(pdf_reader.pages)):
            # Extract text from the current page
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        # Close the BytesIO object
        pdf_file.close()
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF resume: {e}")
        return None

client = OpenAI(
    # This is the default and can be omitted
    api_key='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'
)
def get_resume_summary(resume_text):
    try:
        prompt = "Generate the summary of the resume:" +  resume_text

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        # print("response",response)
        
        if response and response.choices and len(response.choices) > 0:
                summary = response.choices[0].message.content
                # print("summary in func",summary)
                return summary
        else:
            return "Error: Failed to generate summary"
    except Exception as e:
        return f"Error: {e}"

# Function to calculate percentage match between job description and resume
def calculate_percentage_match(job_description, resume):
    stop_words = set(stopwords.words('english'))
    job_description_tokens = [word.lower() for word in word_tokenize(job_description) if word.isalnum() and word.lower() not in stop_words]
    resume_tokens = [word.lower() for word in word_tokenize(resume) if word.isalnum() and word.lower() not in stop_words]
    common_words = set(job_description_tokens).intersection(resume_tokens)
    percentage_match = (len(common_words) / len(job_description_tokens)) * 100
    return round(percentage_match, 2)

# Function to generate resume points based on job description
def generate_resume_points(job_description):
    try:
        prompt = "Generate resume points for the job description:" + input_text

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        # print("response",response)
        
        if response and response.choices and len(response.choices) > 0:
                summary = response.choices[0].message.content
                # print("summary in func",summary)
                return summary
        else:
            return "Error: Failed to generate summary"
    except Exception as e:
        return f"Error: {e}"


    
def fetch_sample_questions(job_description):
    try:
        prompt = "Generate sample interview question and answers for the job description:" + job_description

        response = client.chat.completions.create(
            n=1,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        
        if response and response.choices and len(response.choices) > 0:
                summary = response.choices[0].message.content
                # print("summary in func",summary)
                return summary
        else:
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []

    
def train_model_with_samples(sample_questions):
    try:
        # Ensure sample_questions is not empty
        if not sample_questions:
            raise ValueError("No sample questions provided.")

        training_data = [sample_questions]

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('lsa', TruncatedSVD(n_components=min(5, len(training_data)), random_state=42)),
        ])

        # Fit the pipeline on the questions
        pipeline.fit(training_data)

        # Check if the model is trained
        if pipeline is not None:
            print("Model is trained successfully.")
            return pipeline
        else:
            print("Model is not trained")
            return None
        
    except Exception as e:
        print(f"Error training model with samples: {e}")
        return None



def generate_sample_questions_and_answers(model, input_text):
    try:
        # Use the trained model to transform the input text
        transformed_text = model['tfidf'].transform([input_text])

        # Use the transformed text to generate sample interview questions
        sample_questions = model['lsa'].transform(transformed_text)

        # Find the dominant topic for each sample question
        dominant_topics = np.argmax(sample_questions, axis=1)

        # Format the sample questions and answers
        formatted_questions = []
        for i, topic_idx in enumerate(dominant_topics):
            formatted_questions.append(f"Question {i + 1}: {input_text}?")

        # Join the formatted questions into a single string
        sample_questions_string = '\n\n'.join(formatted_questions)
        # print(sample_questions_string)
        return sample_questions_string
    
    except Exception as e:
        print(f"Error generating sample questions and answers: {e}")
        return ""


# Display main functionality
if "username" in st.session_state:
    st.sidebar.write(f"Logged in as: {st.session_state['username']}")
    input_text = st.text_area("Job Description: ", key="input")
    uploaded_file = st.file_uploader("Upload your resume...", type=["pdf","docx"])

    if uploaded_file is not None:
        st.write("File Uploaded Successfully")

    submit1 = st.button("Tell me about the Resume")
    submit2 = st.button("Percentage Match")
    submit3 = st.button("Sample Interview Questions")
    submit4 = st.button("Resume Points Update")

    # Handle button actions
    if submit1:
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                resume_text = extract_resume_text_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = extract_resume_text_word(uploaded_file)
            else:
                st.write("Unsupported file format. Please upload a PDF or Word document.")      

            if resume_text:
                # st.write("Resume Text:")
                print("resume text",resume_text)
                summary = get_resume_summary(resume_text)
                print("summary in submit1",summary)
                
                if summary:
                    st.subheader("Summary of the Resume:")
                    st.write(summary)
            else:
                st.write("Error: Unable to extract text from the resume")
        else:
            st.write("Please upload the resume")

    elif submit2:
        if input_text and uploaded_file:
            # Extract text from uploaded resume
            # resume_text = extract_resume_text(uploaded_file)

            if uploaded_file.type == "application/pdf":
                resume_text = extract_resume_text_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = extract_resume_text_word(uploaded_file)
            else:
                st.write("Unsupported file format. Please upload a PDF or Word document.")      

            
            if resume_text:
                # Calculate percentage match
                percentage_match = calculate_percentage_match(input_text, resume_text)
                st.write(f"Percentage Match: {percentage_match}%")
            else:
                st.write("Error: Unable to extract text from the resume")
        else:
            st.write("Please provide both the job description and upload the resume")

    
    elif submit3:
        # Fetch sample questions
        sample_questions = fetch_sample_questions(input_text)
        # print(type(sample_questions))
        if sample_questions:
            # Display sample questions and answers
            st.write("Sample Questions and Answers:")
            st.write(sample_questions)

            # Train model with fetched sample questions
            model = train_model_with_samples(sample_questions)
            print("model", model)

            if model:
                st.success("Model trained successfully.")

                # Generate sample questions using the trained model
                generated_questions = generate_sample_questions_and_answers(model, input_text)
                if generated_questions:
                    # st.write("Generated Sample Questions:")
                    # st.write(generated_questions)
                    print("Generated Sample Questions")
                else:
                    # st.error("Failed to generate sample questions.")
                    print("Failed to generate sample questions")
            else:
                st.error("Failed to train the model.")
        else:
            st.error("Failed to fetch sample questions.")

    
    
    elif submit4:
        if input_text:
            # Generate resume points based on job description
            resume_points = generate_resume_points(input_text)
            # Display the generated resume points
            st.write("Generated Resume Points:")
            st.write(resume_points)
        else:
            st.write("Please enter a job description before generating resume points.")

    

