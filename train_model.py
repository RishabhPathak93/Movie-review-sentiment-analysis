import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tqdm import tqdm
import joblib

# Constants
MAX_FEATURES = 10000  # Vocabulary size
MAX_LEN = 200         # Max sequence length
MODEL_PATH = "gru_sentiment_model"
PREPROCESSOR_PATH = "preprocessor.joblib"

# Load IMDB data
print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

# Padding sequences
print("Padding sequences...")
X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)

# Build the GRU model
print("Building the model...")
model = Sequential([
    Embedding(input_dim=MAX_FEATURES, output_dim=128, input_length=MAX_LEN),
    GRU(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with a progress bar
batch_size = 32
epochs = 15
print("Training the model...")
with tqdm(total=epochs) as pbar:
    for epoch in range(epochs):
        model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=1, verbose=0)
        pbar.update(1)

# Evaluate the model
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the model
print(f"Saving model to {MODEL_PATH}...")
# Save the model in HDF5 format
model.save(f"{MODEL_PATH}.h5")

# Save preprocessing objects using Joblib
preprocessing_data = {"max_features": MAX_FEATURES, "max_len": MAX_LEN, "word_index": imdb.get_word_index()}
print(f"Saving preprocessing data to {PREPROCESSOR_PATH}...")
joblib.dump(preprocessing_data, PREPROCESSOR_PATH)

print("Training and saving complete!")
