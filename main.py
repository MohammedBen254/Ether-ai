from flask import Flask, render_template, request, jsonify
import random
import numpy as np
import json
import re
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
import nltk
app = Flask(__name__)

lemmatizer = WordNetLemmatizer()    

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load data
file_path = 'app/french.json'
data = load_json_data(file_path)

# Extract patterns and tags
patterns = []
tags = []
responses_dict = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    responses_dict[intent['tag']] = intent['responses']

# Data cleaning function
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


# Clean the patterns
patterns = [clean_text(pattern) for pattern in patterns]

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
word_index = tokenizer.word_index

# Padding sequences
maxlen = 200
X = pad_sequences(sequences, maxlen=maxlen)

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Load the model
model = tf.keras.models.load_model("app/Model.h5")

# Function to predict the intent of a given text
def predict_intent(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)
    intent = label_encoder.inverse_transform([np.argmax(pred)])
    return intent[0]

# Function to get a response
def get_response(text):
    intent = predict_intent(text)
    response = random.choice(responses_dict[intent])
    return response

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['user_input']
    response = get_response(user_input)
    return jsonify({'bot_response': response})

if __name__ == '__main__':
    app.run(debug=True)