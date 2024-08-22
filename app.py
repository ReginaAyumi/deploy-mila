from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import json
import random
from nltk import word_tokenize
from nltk.stem import PorterStemmer

# Load the model and tokenizer
model_path = "./indobert-finetuned-classification"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Load intents and responses
with open('pregnant-chat.json') as f:
    intents = json.load(f)

# Create DataFrame for responses
def create_response_df():
    data = pd.DataFrame({
        'Tag': [],
        'Response': [],
    })
    return data

def extract_responses(json_file, data):
    for intent in json_file['intents']:
        for response in intent['responses']:
            tag_response = [intent['tag'], response]
            data.loc[len(data.index)] = tag_response
    return data

response_df = create_response_df()
response_df = extract_responses(intents, response_df)

# Define labels
labels = response_df['Tag'].unique().tolist()
labels = [s.strip() for s in labels]

# Create label mappings
num_labels = len(labels)
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

# Flask app initialization
app = Flask(__name__)

# Function to preprocess input text
stemmer = PorterStemmer()
ignore_words = ['?', '!', ',', '.']

def preprocessing_pattern(pattern):
    words = word_tokenize(pattern.lower())
    stemmed_words = [stemmer.stem(word) for word in words if word not in ignore_words]
    return " ".join(stemmed_words)

# Function to predict intent/tag
def predict_intent(text):
    preprocessed_text = preprocessing_pattern(text)
    inputs = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    predicted_tag = id2label[predicted_class_id]
    return predicted_tag

# Function to get response based on predicted tag
def get_response(tag):
    possible_responses = response_df[response_df['Tag'] == tag]['Response'].tolist()
    if possible_responses:
        return random.choice(possible_responses)
    else:
        return "Maaf, saya tidak mengerti pertanyaan Anda."

# Flask route to handle POST request
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('text')
    predicted_tag = predict_intent(user_input)
    response = get_response(predicted_tag)
    return jsonify({"tag": predicted_tag, "response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
