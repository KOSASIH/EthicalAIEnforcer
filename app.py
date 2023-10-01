import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained sentiment analysis model
model = load_model('path_to_model.h5')

# Define a function to preprocess the input text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Remove stopwords and special characters
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join the preprocessed tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Perform sentiment analysis on the input text
def perform_sentiment_analysis(text):
    preprocessed_text = preprocess_text(text)
    
    # Convert the preprocessed text into input format for the model
    input_text = np.array([preprocessed_text])
    
    # Predict the sentiment using the pre-trained model
    prediction = model.predict(input_text)
    
    # Map the predicted sentiment to the corresponding label
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_sentiment = sentiment_labels[np.argmax(prediction)]
    
    return predicted_sentiment

# Example usage
input_text = "I really enjoyed the movie. It was fantastic!"
predicted_sentiment = perform_sentiment_analysis(input_text)
print(f"Predicted Sentiment: {predicted_sentiment}")
