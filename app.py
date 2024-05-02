import streamlit as st
import psycopg2
from psycopg2 import Error
import bcrypt
import os
# from dotenv import load_dotenv
# from PIL import Image, ImageOps
# import base64
# import io
# import pdf2image
import nltk
import docx
# import requests
from openai import OpenAI
import PyPDF2
from io import BytesIO

# import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer


import json

# # import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.pipeline import Pipeline

# from sklearn.decomposition import TruncatedSVD

# import openai


# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from transformers import GPT2Tokenizer



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
    # api_key='Your_API_KEY'
    api_key=''
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
def generate_resume_points(input_text):
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
    
    
import json
import os
from sklearn.metrics import accuracy_score, precision_score, f1_score
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import matplotlib.pyplot as plt

def fine_tune_gpt2_model(job_description, jsonl_file):
    try:
        # Check if the model is already fine-tuned
        if not is_fine_tuned_model_exists():
            # Load pre-trained GPT-2 tokenizer and model
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

            # Set pad_token_id explicitly
            tokenizer.pad_token_id = tokenizer.eos_token_id 

            model = GPT2LMHeadModel.from_pretrained("gpt2")

            # Prepare dataset
            samples = []
            print("Reading JSONL file:", jsonl_file)
            with open(jsonl_file, "r", encoding="utf-8") as file:
                for line in file:
                    sample = json.loads(line.strip())
                    if sample["job_title"] == job_description:
                        question = sample["question"]
                        # Ensure all questions have the same length
                        question = question[:tokenizer.model_max_length - 2]  # Subtract 2 for special tokens
                        answer = sample["answer"]
                        samples.append({"question": question, "answer": answer})

            if not samples:
                print(f"No samples found for job description '{job_description}'")
                return False

            # Fine-tune the model using sample questions and answers
            train_dataset = [{"input_ids": tokenizer.encode(sample["question"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True),
                            "labels": tokenizer.encode(sample["answer"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True)}
                            for sample in samples]
            
            training_args = TrainingArguments(
                output_dir="C:/Users/saran/OneDrive/Desktop/botproject/output",
                overwrite_output_dir=True,
                num_train_epochs=5,
                per_device_train_batch_size=8,  # Increase batch size
                save_steps=10_000,
                save_total_limit=2,
                gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
                fp16=True,  # Use mixed precision training if possible
                prediction_loss_only=True,  # Speed up training by only computing loss
                logging_steps=500,  # Log training progress less frequently
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )

            trainer.train()

            # Save the fine-tuned model
            trainer.save_model("C:/Users/saran/OneDrive/Desktop/botproject/fine_tuned_gpt2_model")

            # Save the fine-tuned tokenizer
            tokenizer.save_pretrained("C:/Users/saran/OneDrive/Desktop/botproject/fine_tuned_gpt2_model")

            # Evaluate the fine-tuned model
            eval_results = evaluate_fine_tuned_model(trainer, train_dataset)

            # Plot evaluation metrics
            plot_evaluation_metrics(eval_results)

            return True
        else:
            print("Model is already fine-tuned. Skipping fine-tuning.")
            return True
        
    except FileNotFoundError as e:
        print(f"Error: Input file not found at path '{jsonl_file}'")
        return False
    
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return False

def evaluate_fine_tuned_model(trainer, dataset):
    # Evaluate the fine-tuned model on the dataset
    eval_result = trainer.evaluate(eval_dataset=dataset)
    predictions = trainer.predict(eval_dataset=dataset)

    # Compute evaluation metrics
    labels = [example["labels"] for example in dataset]
    predicted_labels = predictions.predictions.argmax(-1)
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels, average='weighted')
    f1 = f1_score(labels, predicted_labels, average='weighted')

    eval_result['accuracy'] = accuracy
    eval_result['precision'] = precision
    eval_result['f1'] = f1

    return eval_result

def plot_evaluation_metrics(eval_results):
    # Extract evaluation metrics
    accuracy = eval_results['accuracy']
    precision = eval_results['precision']
    f1 = eval_results['f1']

    # Define labels and values for the metrics
    metrics = ['Accuracy', 'Precision', 'F1 Score']
    values = [accuracy, precision, f1]

    # Plot the metrics
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'orange'])
    plt.title('Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.ylim(0, 1)  # Set y-axis limit to range [0, 1] for accuracy, precision, and F1 score
    plt.show()


def generate_sample_questions_and_answers(job_description):
    try:
        # Load fine-tuned GPT-2 tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2_model")
        model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2_model")

        # Define prompt for generating sample questions and answers
        prompt = f"Generate sample interview questions and answers based on the following job description:\n\n{job_description}\n\nQuestion:\n1.\nAnswer:\n1."

        # Generate sample questions and answers
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=300, num_return_sequences=5, do_sample=True)

        sample_questions = []
        sample_answers = []
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            print("Generated Text:", text)  # Debugging output

            parts = text.split("Answer:", 1)
            if len(parts) == 2:
                question = parts[0].strip()
                answer = parts[1].strip()

                # Remove prompt from question and answer
                sample_questions.append(question)
                sample_answers.append(answer)
                                
            else:
                print("Error: Unable to split text into question and answer.")
                print("Text:", text)

        return sample_questions, sample_answers
    except Exception as e:
        print(f"Error generating sample questions and answers: {e}")
        return [], []

def is_fine_tuned_model_exists():
    # Check if the fine-tuned model directory exists
    model_directory = "C:/Users/saran/OneDrive/Desktop/botproject/fine_tuned_gpt2_model"  # Path to the directory where the fine-tuned model is saved
    return os.path.exists(model_directory)

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


    # elif submit3:
    #     # Fetch sample questions
    #     sample_questions = fetch_sample_questions(input_text)
        
    #     if sample_questions:
    #         # Save sample questions to a text file
    #         sample_questions_file = "./sample_questions.txt"
    #         save_sample_questions_to_file(sample_questions.split("\n"), sample_questions_file)
    #         # Fine-tune GPT-2 model using sample questions
    #         print("Fine-tuning GPT-2 model...")
    #         if fine_tune_gpt2_model(sample_questions_file):
    #             st.write("Fine-tuning complete.")
    #             #st.write("Sample Questions and Answers:")
    #             #st.write(sample_questions)
    #             # Generate sample questions and answers
    #             sample_questions, sample_answers = generate_sample_questions_and_answers(input_text)

    #             if sample_questions and sample_answers:
    #                 st.write("Sample Interview Questions and Answers:")
    #                 for i, (question, answer) in enumerate(zip(sample_questions, sample_answers), start=1):
    #                     st.write(f"Question {i}: {question}")
    #                     st.write(f"Answer {i}: {answer}")
    #             else:
    #                 print("Failed to generate sample questions and answers.")
    #         else:
    #             print("Error occurred during fine-tuning.")
    #     else:
    #         print("Failed to fetch sample questions.")

    elif submit3:
        data_jsonl_file = "C:/Users/saran/OneDrive/Desktop/botproject/data.jsonl"
        print("Data JSONL File Path:", data_jsonl_file)
        sample_questions = fetch_sample_questions(input_text)
        st.write("Sample Questions and Answers:")
        st.write(sample_questions)
    # Fetch sample questions and answers
        if fine_tune_gpt2_model(input_text, data_jsonl_file):
            print("Fine-tuning complete.")
            # Generate sample questions and answers
            sample_questions, sample_answers = generate_sample_questions_and_answers(input_text)

            if sample_questions and sample_answers:
                st.write("Sample Interview Questions and Answers:")
                for i, (question, answer) in enumerate(zip(sample_questions, sample_answers), start=1):
                    print(f"Question {i}: {question}")
                    print(f"Answer {i}: {answer}")
            else:
                print("Failed to generate sample questions and answers.")
        else:
            print("Error occurred during fine-tuning.")

    elif submit4:
        if input_text:
            # Generate resume points based on job description
            resume_points = generate_resume_points(input_text)
            # Display the generated resume points
            st.write("Generated Resume Points:")
            st.write(resume_points)
        else:
            st.write("Please enter a job description before generating resume points.")


