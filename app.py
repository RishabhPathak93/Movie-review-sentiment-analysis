import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Paths to model and preprocessor
MODEL_PATH = "gru_sentiment_model.h5"
PREPROCESSOR_PATH = "preprocessor.joblib"

# Load the model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# Function to preprocess user input
def preprocess_review(review, word_index, max_len):
    words = review.lower().split()
    # Convert words to integers (use 2 for unknown words, max index is MAX_FEATURES-1)
    encoded_review = [word_index.get(word, 2) for word in words if word_index.get(word, 2) < 10000]
    # Pad the sequence
    return pad_sequences([encoded_review], maxlen=max_len)

# Streamlit UI
st.title("Sentiment Analysis with GRU")
st.write("This app predicts whether a movie review is **positive** or **negative**.")

# Input area for user
user_input = st.text_area("Enter a movie review:", "Type your review here...")

if st.button("Analyze Sentiment"):
    max_features = preprocessor["max_features"]
    max_len = preprocessor["max_len"]
    word_index = preprocessor["word_index"]

    # Preprocess the user input
    processed_input = preprocess_review(user_input, word_index, max_len)

    if processed_input.size == 0:
        st.write("Please enter a valid review with recognizable words.")
    else:
        # Predict sentiment
        prediction = model.predict(processed_input)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"

        # Display the result
        st.write(f"**Sentiment**: {sentiment}")
        st.write(f"**Confidence**: {prediction:.2f}")
