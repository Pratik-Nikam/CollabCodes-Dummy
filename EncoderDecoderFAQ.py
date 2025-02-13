import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------

# Assume your FAQ CSV has two columns: 'question' and 'answer'
data = pd.read_csv('faq_data.csv')  # Replace with your data file

# For the answer text, add special tokens to mark the start and end of each sequence.
data['answer'] = data['answer'].apply(lambda x: 'START_ ' + x + ' _END')

# ---------------------------
# 2. Tokenize the Text
# ---------------------------

# Create separate tokenizers for questions and answers.
num_words = 10000  # Adjust vocabulary size as needed

question_tokenizer = Tokenizer(num_words=num_words, filters='')
question_tokenizer.fit_on_texts(data['question'])
answer_tokenizer = Tokenizer(num_words=num_words, filters='')
answer_tokenizer.fit_on_texts(data['answer'])

# Convert texts to sequences
encoder_input_sequences = question_tokenizer.texts_to_sequences(data['question'])
decoder_input_sequences = answer_tokenizer.texts_to_sequences(data['answer'])

# Determine maximum sequence lengths for padding
max_encoder_seq_length = max(len(seq) for seq in encoder_input_sequences)
max_decoder_seq_length = max(len(seq) for seq in decoder_input_sequences)

# Pad sequences so they have the same length
encoder_input_data = pad_sequences(encoder_input_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(decoder_input_sequences, maxlen=max_decoder_seq_length, padding='post')

# Prepare decoder target data by shifting decoder_input_data one timestep to the left
num_decoder_tokens = len(answer_tokenizer.word_index) + 1  # +1 for reserved index 0
decoder_target_data = np.zeros((len(data), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, seq in enumerate(decoder_input_sequences):
    for t in range(1, len(seq)):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_target_data[i, t-1, seq[t]] = 1.0

# ---------------------------
# 3. Build the Seq2Seq Model
# ---------------------------

latent_dim = 256  # Dimensionality for the encoding space

# Encoder
encoder_inputs = Input(shape=(None,), name="encoder_inputs")
enc_emb = Embedding(input_dim=len(question_tokenizer.word_index)+1, output_dim=latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
dec_emb_layer = Embedding(input_dim=len(answer_tokenizer.word_index)+1, output_dim=latent_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn encoder_input_data & decoder_input_data into decoder_target_data
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------
# 4. Train the Model
# ---------------------------

epochs = 50
batch_size = 64

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# ---------------------------
# 5. Build Inference Models
# ---------------------------
# To generate an answer, we need to build models for inference.

# Encoder model for inference: given an input sequence, output the internal states.
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model for inference:
# We'll need to define input placeholders for the states and then run the decoder LSTM step by step.
decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_input_h")
decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_input_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

# ---------------------------
# 6. Define a Function for Decoding Sequences
# ---------------------------
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence with only the start token.
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = answer_tokenizer.word_index['start_']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # Map token index to word
        sampled_word = None
        for word, index in answer_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break

        if sampled_word is None:
            break

        if sampled_word == '_end' or len(decoded_sentence.split()) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        # Update states
        states_value = [h, c]
    return decoded_sentence.strip()

# ---------------------------
# 7. Test the Model on a New Question
# ---------------------------
# Pick a random question from the data and generate an answer.
test_index = np.random.choice(len(encoder_input_data))
input_seq = encoder_input_data[test_index:test_index+1]
decoded_answer = decode_sequence(input_seq)
print("Question:", data['question'].iloc[test_index])
print("Predicted Answer:", decoded_answer)
