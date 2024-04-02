# careercraftsman_chatbot_project

About the app:
The app launched with the provided code is a Streamlit-based chatbot designed to assist users with various tasks related to career development and job search. Here's an overview of the app's functionality and potential use cases:

User Authentication:
    •	Users can register for an account by providing their first name, last name, username, and password.
    •	Registered users can log in using their credentials, and their session state is maintained throughout their 
      interaction with the app.
      
Resume Analysis:
    •	Users can upload their resumes (in PDF or DOCX format) to the app.
    •	The app extracts text from the uploaded resume and uses OpenAI's GPT-3.5 model to generate a summary of the resume's 
      content.
    •	Additionally, users can provide a job description, and the app calculates the percentage match between the job 
      description and the uploaded resume. This feature helps users tailor their resumes to specific job requirements.
      
Interview Preparation:
    •	Users can input a job description or select a job category.
    •	The app utilizes GPT-3.5 to generate sample interview questions and answers tailored to the provided job description 
       or category. This feature helps users prepare for job interviews by simulating common interview scenarios.
       
Resume Point Generation:
    •	Users can input a job description, and the app generates key resume points based on the job requirements. These resume 
      points highlight relevant skills, experiences, or achievements that users may want to include in their resumes when 
      applying for the specified job.
      
User Interaction:
    •	The app provides a user-friendly interface for interacting with different features.
    •	Users can navigate between login, registration, and logout pages using checkboxes in the sidebar.
    •	Input fields and buttons allow users to provide input, trigger actions, and view results.
    
NLP and ML Techniques:
    •	The app leverages Natural Language Processing (NLP) techniques such as text summarization, keyword extraction, and 
      question generation.
    •	It also utilizes machine learning models from the Scikit-learn library for tasks like TF-IDF vectorization and latent 
      semantic analysis.
      
Overall, the app serves as a comprehensive tool for individuals seeking assistance with resume analysis, interview preparation, and career advancement. It combines advanced NLP models with user-friendly features to provide valuable insights and guidance to users in their professional endeavors.

