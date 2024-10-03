from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import requests

# Load the model and tokenizer
model_path = "reginaayumi/mila-chatbot"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Load intents from pregnant.json
with open('pregnant.json') as f:
    intents = json.load(f)

# Extract tags from the intents
tags = [intent['tag'] for intent in intents['intents']]

# Create label mappings
id2label = {i: tag for i, tag in enumerate(tags)}
label2id = {tag: i for i, tag in enumerate(tags)}

# Flask app initialization
app = Flask(__name__)

# Preprocessing input text
stemmer = PorterStemmer()
ignore_words = ['?', '!', ',', '.']

def preprocessing_pattern(pattern):
    words = word_tokenize(pattern.lower())
    stemmed_words = [stemmer.stem(word) for word in words if word not in ignore_words]
    return " ".join(stemmed_words)

# Predict intent using IndoBERT
def predict_intent(text):
    # Preprocess text
    preprocessed_text = preprocessing_pattern(text)
    
    # Tokenize input text using tokenizer
    inputs = tokenizer(preprocessed_text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    
    # Perform inference with the model
    outputs = model(**inputs)
    
    # Get predicted class (numerical label)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    
    # Map numerical label to string tag
    predicted_tag = id2label[predicted_class_id]
    
    return predicted_tag


# Ollama API for response generation
def get_ollama_response(user_input):
    LLM_API_BASE_URL = "https://api.groq.com/openai/v1"
    LLM_API_KEY = "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"
    model = "llama-3.1-8b-instant"
    
    url = f"{LLM_API_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are a knowledgeable assistant specialized in pregnancy health. Answer the user's pregnancy-related questions politely and concisely, while providing helpful information. Always remind the user that every pregnancy is different for each person and that they should consult with a doctor or healthcare provider for personalized medical advice. Answer in plain text (concisely, maximum 3 sentences) and not in Markdown format."},
            {"role": "user", "content": user_input}
        ],
        "model": model,
        "max_tokens": 400,
        "temperature": 0
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "Maaf, saya tidak bisa memberikan jawaban saat ini."

# Flask route to handle POST request
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('text')
    
    # Classify input into a tag
    predicted_tag_id = predict_intent(user_input)
    
    # Get response from Ollama API
    ollama_response = get_ollama_response(user_input)

    return jsonify({"tag": predicted_tag_id, "response": ollama_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
