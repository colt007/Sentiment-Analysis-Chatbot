import re
import random
import numpy as np
import pandas as pd
import nltk
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset from the Excel file
data = pd.read_excel("test.xlsx")  # Replace with your file path

# Preprocess data and convert emotions to numerical labels
X = data["text"]
y = data["emotion"]

# Preprocessing function with advanced techniques
def preprocess_text_advanced(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = mark_negation(tokens)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Tokenization and padding
max_words = 1000  # Maximum number of words to keep
tfidf_vectorizer_advanced = TfidfVectorizer(max_features=max_words)
X_tfidf = tfidf_vectorizer_advanced.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a logistic regression model with advanced preprocessing
model_advanced = LogisticRegression()
model_advanced.fit(X_train, y_train)
# Mapping of emotion labels to numerical values
emotion_mapping = {"positive": 1, "neutral": 0, "negative": -1}


emotional_responses = {
    "positive": [
        "I'm glad you're feeling positive!",
        "That sounds great!",
        "Keep up the good vibes!",
        "It's wonderful to hear that!",
        "Positivity is contagious!",
        "Your positive energy is inspiring!",
    ],
    "negative": [
        "I'm here to listen. Is there anything I can do?",
        "Remember that tough times don't last forever.",
        "Stay strong, things will get better.",
        "I'm sorry to hear that. Remember, I'm here to chat.",
        "You're not alone. We all face challenges sometimes.",
    ],
    "neutral": [
        "Tell me more. I'm here to listen.",
        "Your input is noted. How else can I assist you?",
        "It's okay to have neutral feelings. Let's chat.",
        "Let's explore more about your thoughts and feelings.",
        "Feel free to share more. I'm here for you.",
    ],
}

# Predict sentiment using the advanced model
def predict_sentiment_advanced(text):
    preprocessed_input = preprocess_text_advanced(text)
    input_tfidf = tfidf_vectorizer_advanced.transform([preprocessed_input])
    sentiment = model_advanced.predict(input_tfidf)[0]
    return sentiment

# Predict emotional category based on sentiment
def predict_emotion_advanced(sentiment):
    if sentiment == "positive":
        return "positive"
    elif sentiment == "neutral":
        return "neutral"
    else:
        return "negative"

# Function to extract named entities using NLTK
def extract_named_entities(text):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    named_entities = nltk.ne_chunk(tagged_tokens)
    return named_entities

# Function to select emotional response based on named entities
def select_emotional_response(emotion, named_entities):
    if emotion == "positive":
        responses = emotional_responses["positive"]
    elif emotion == "negative":
        responses = emotional_responses["negative"]
    else:
        responses = emotional_responses["neutral"]
    
    # Customize response based on named entities (simplified example)
    for entity in named_entities:
        if isinstance(entity, nltk.tree.Tree):
            if entity.label() == "PERSON":
                return random.choice(responses) + f" Are you talking about {entity.leaves()[0][0]}?"
    
    return random.choice(responses)
explicitly_named = False  # Flag to track if the bot has been explicitly named
global bot_name 
bot_name="Sentiment Analysis ChatBot"

# Function to get bot name from user
def get_bot_name():
    global bot_name
    bot_name = input("Sentiment Analysis Chatbot: What would you like to name me?")
    return bot_name

def handle_user_input():
    user_response = user_input.get()
    if user_response.lower() in ['exit', 'bye', 'gotta go', 'goodbye', ':(']:
        chat_history.insert(tk.END, f"{bot_name}: Goodbye!\n")
        chat_history.see(tk.END)  # Scroll to the bottom
        user_input.delete(0, tk.END)  # Clear the input field
        root.after(1000, root.quit)  # Delay and then exit the Tkinter event loop
        return
    
    chat_history.insert(tk.END, f"You: {user_response}\n")
    
    # Predict sentiment and emotion of user response
    user_sentiment_advanced = predict_sentiment_advanced(user_response)
    user_emotion_advanced = predict_emotion_advanced(user_sentiment_advanced)
    
    # Display predicted emotion
    chat_history.insert(tk.END, f"Sentiment Analysis Chatbot: The predicted emotion of your response is {user_emotion_advanced}.\n")
    
    user_input.delete(0, tk.END)
    
    sentiment_advanced = predict_sentiment_advanced(user_response)
    emotion_advanced = predict_emotion_advanced(sentiment_advanced)
    named_entities = extract_named_entities(user_response)
    response = select_emotional_response(emotion_advanced, named_entities)
    chat_history.insert(tk.END, f"Sentiment Analysis Chatbot: {response}\n")
    
    chat_history.see(tk.END)

# Create the main GUI window
root = tk.Tk()
root.title("Sentiment Analysis ChatBot")

# Create and configure chat history text box
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=25)
chat_history.config
chat_history.pack()

# Create user input entry field
user_input = tk.Entry(root, width=50)
user_input.pack()

# Create submit button
submit_button = tk.Button(root, text="Submit", command=handle_user_input)
submit_button.pack()

# Run the Tkinter event loop
root.mainloop()