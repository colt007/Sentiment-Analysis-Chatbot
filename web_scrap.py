
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import wikipediaapi


nltk.download('punkt')
nltk.download('stopwords')

def scrape_web(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.text for para in paragraphs])
    return text

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ''
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def get_wikipedia_content(title):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(title)
    if page.exists():
        return page.text
    else:
        return "The page does not exist."

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

def chatbot_response(user_input):
    if user_input.startswith("scrape"):
        url = user_input.split(" ")[1]
        content = scrape_web(url)
    elif user_input.startswith("pdf"):
        pdf_path = user_input.split(" ")[1]
        content = extract_text_from_pdf(pdf_path)
    elif user_input.startswith("wiki"):
        title = ' '.join(user_input.split(" ")[1:])
        content = get_wikipedia_content(title)
    else:
        return "Invalid input. Please start your query with 'scrape', 'pdf', or 'wiki'."
    
    preprocessed_content = preprocess_text(content)
    response = ' '.join(preprocessed_content[:100])  # Summarize for simplicity
    return response

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")
