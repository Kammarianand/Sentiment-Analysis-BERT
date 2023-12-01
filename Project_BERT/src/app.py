import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Define the sentiment labels
sentiment_labels = {0:"positive", 1:"neutral", 2:"negative"}
message_types = {'Negative': 'error', 'Neutral': 'info', 'Positive': 'success'}

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create a function to check if input is valid
def is_valid_input(text):
    return bool(text.strip())  # Check if the text is not empty after stripping whitespace

# Create a function to predict sentiment
def predict_sentiment(text, model, tokenizer):
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(output.logits, dim=1).numpy().squeeze()
        prediction = np.argmax(probabilities)

    return sentiment_labels[prediction], probabilities

# Create the Streamlit user interface
st.title('Sentiment Analysis with BERT')

input_text = st.text_input('Enter your text here:')

if st.button('Analyze'):
    if not is_valid_input(input_text):
        st.error('Please enter valid text.')
    else:
        prediction, probabilities = predict_sentiment(input_text, model, tokenizer)
        formatted_probabilities = [f"{prob * 100:.2f}%" for prob in probabilities]

        st.write('Sentiment:', prediction)
        st.write('Prediction Probabilities:', {label: prob for label, prob in zip(sentiment_labels.values(), formatted_probabilities)})
