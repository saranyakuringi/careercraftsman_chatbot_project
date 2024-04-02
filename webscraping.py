# from bs4 import BeautifulSoup
# import csv

# html_code = '''
# <h3 style="text-align:justify"><a id="what-is-go"></a><span style="font-size:18px">1. What is Go?</span></h3>
# <p style="text-align:justify">Go is also known as <a title="Golang Resources" href="https://mindmajix.com/golang" target="_blank">GoLang</a>, it is a general-purpose programming language designed at Google and developed by Robert Griesemer, ken Thomson and Rob Pike. It is a statistically typed programming language used for building fast and reliable applications.</p>
# '''

# # Parse the HTML content
# soup = BeautifulSoup(html_code, "html.parser")

# # Extract question and answer
# question_element = soup.find("span")
# question_number, question_text = question_element.text.split('. ', 1)
# answer_element = soup.find("p")
# answer_text = answer_element.text.strip()


# csv_filename = 'interview_questions.csv'
# # Write to CSV
# with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Question', 'Answer', 'Source URL'])
#     writer.writerow([question_text, answer_text, "https://mindmajix.com/go-interview-questions"])


# import requests
# from bs4 import BeautifulSoup
# import csv

# # URL of the webpage to scrape
# url = "https://mindmajix.com/go-interview-questions"

# # Send a GET request to the URL
# response = requests.get(url)

# # Parse the HTML content
# soup = BeautifulSoup(response.content, "html.parser")

# # Extract all question and answer pairs
# question_elements = soup.find_all("h3")
# answer_elements = soup.find_all("p")

# # Write to CSV
# csv_filename = 'interview_questions_1.csv'
# with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Question', 'Answer', 'Source URL'])
    
#     # Iterate through question elements
#     for i, question_element in enumerate(question_elements):
#         question_text = question_element.text.strip()
        
#         # Find the immediate following answer
#         # Since not all questions have corresponding answers, we need to handle that case
#         try:
#             answer_text = answer_elements[i].text.strip()
#         except IndexError:
#             answer_text = "No answer available"
        
#         writer.writerow([question_text, answer_text, url])

# print(f"Data written to '{csv_filename}'")



import requests
from bs4 import BeautifulSoup
import csv

# URL of the webpage to scrape
url = "https://mindmajix.com/go-interview-questions"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Find all h3 and p tags
question_tags = soup.find_all("h3")
answer_tags = soup.find_all("p")[9:]  # Skipping the first 9 <p> tags

# Write to CSV
csv_filename = 'interview_questions.csv'
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Question', 'Answer', 'Source URL'])
    
    # Iterate through the question and answer tags
    for i in range(len(question_tags)):
        question_text = question_tags[i].text.strip()
        answer_text = answer_tags[i].text.strip()
        writer.writerow([question_text, answer_text, url])

print(f"Data written to '{csv_filename}'")
